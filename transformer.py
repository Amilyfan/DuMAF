import torch                                    # 导入 PyTorch 主库
from torch import nn, einsum                    # nn: 神经网络模块; einsum: 爱因斯坦求和（高效张量运算）
from torch.nn import Module, ModuleList         # Module: 所有网络层的基类; ModuleList: 可存储子模块的列表
import torch.nn.functional as F                 # 函数式API（激活函数、softmax等）

from einops import rearrange, repeat            # einops: 张量维度重排/重复的简洁工具库

# ============================================================
# 手动实现: 替代 hyper_connections 和 discrete_continuous_embed_readout
# ============================================================

class ResidualWrapper(Module):
    """简单残差连接包装器, 替代 hyper_connections
       作用: 将子模块(branch)的输出与输入相加, 实现 x + branch(x) 的残差学习"""
    def __init__(self, branch):
        super().__init__()                      # 调用父类 Module 的初始化
        self.branch = branch                    # 保存被包装的子模块（如 Attention 或 FeedForward）

    def forward(self, x):
        result = self.branch(x)                 # 将输入 x 送入子模块, 得到输出
        if isinstance(result, tuple):           # 如果子模块返回的是元组（如 Attention 返回 (output, attn_weights)）
            return (x + result[0],) + result[1:]  # 只对第一个元素做残差相加, 其余元素原样传递
        return x + result                       # 非元组情况: 直接做残差相加 x + branch(x)


class SimpleEmbed(Module):
    """
    替代 discrete_continuous_embed_readout.Embed
    为分类特征和数值特征分别创建嵌入, 并拼接为 token 序列
    核心思想: 将表格中每一列特征都映射为一个 dim 维的 token, 供 Transformer 处理
    """
    def __init__(self, dim, num_discrete=(), num_continuous=0):
        """
        参数:
            dim:            每个 token 的嵌入维度
            num_discrete:   元组, 每个元素是对应分类特征的类别数量, 如 (3, 5, 2)
            num_continuous:  数值型特征的列数
        """
        super().__init__()                      # 调用父类初始化
        self.categ_embeddings = ModuleList([     # 为每个分类特征创建独立的 Embedding 层
            nn.Embedding(num_cat, dim) for num_cat in num_discrete
            # nn.Embedding(num_cat, dim): 查找表, 将整数 ID 映射为 dim 维向量
            # num_cat: 该列分类特征的类别总数（含特殊token）
        ])
        self.num_continuous = num_continuous     # 记录数值型特征的列数
        if num_continuous > 0:                  # 如果存在数值型特征
            # 为每个数值特征创建可学习的权重向量, 形状 (num_continuous, dim)
            self.numerical_weights = nn.Parameter(torch.randn(num_continuous, dim))
            # 为每个数值特征创建可学习的偏置向量, 形状 (num_continuous, dim)
            self.numerical_biases = nn.Parameter(torch.randn(num_continuous, dim))
            # 数值特征的嵌入公式: token = x * weight + bias （逐特征线性投影）

    def forward(self, inputs):
        x_categ, x_numer = inputs               # 解包输入: x_categ=(b, C_cat) 分类特征, x_numer=(b, C_num) 数值特征
        tokens = []                             # 用于收集所有特征对应的 token
        # 每个分类特征独立 embedding, 作为一个 token
        for i, embed in enumerate(self.categ_embeddings):  # 遍历每个分类特征列的 Embedding 层
            tokens.append(embed(x_categ[:, i]).unsqueeze(1))  # x_categ[:,i] 取第i列 → Embedding → (b, dim) → unsqueeze → (b, 1, dim)
        # 数值特征: 逐特征线性投影, 每个数值特征也作为一个 token
        if self.num_continuous > 0:             # 如果有数值型特征
            x_num = x_numer.unsqueeze(-1)       # (b, C_num) → (b, C_num, 1), 增加最后一维用于广播乘法
            num_tokens = x_num * self.numerical_weights + self.numerical_biases  # (b, C_num, 1) * (C_num, dim) + (C_num, dim) → (b, C_num, dim)
            tokens.append(num_tokens)           # 将所有数值特征的 token 一次性加入列表
        return torch.cat(tokens, dim=1)         # 沿 token 维度拼接: (b, C_cat + C_num, dim), 即总共 C_cat+C_num 个 token

# ===================== 前馈网络和注意力机制 =====================

class GEGLU(Module):
    """GEGLU 激活函数: 一种门控线性单元变体
       公式: GEGLU(x) = x_1 * GELU(x_2), 其中 x 沿最后一维对半分为 x_1 和 x_2
       相比普通 GELU, 门控机制让网络能学习"哪些信息该通过、哪些该抑制" """
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)        # 将输入沿最后一维对半切分: x=(b,n,d), gates=(b,n,d)
        return x * F.gelu(gates)                # 用 GELU 激活门控信号, 再与 x 逐元素相乘

def FeedForward(dim, mult = 4, dropout = 0.):
    """构建前馈网络（FFN）模块
       结构: LayerNorm → Linear(扩展2倍给GEGLU) → GEGLU → Dropout → Linear(压缩回dim)
       参数:
           dim:     输入和输出维度
           mult:    隐藏层扩展倍数（默认4倍）
           dropout: Dropout 概率
    """
    return nn.Sequential(
        nn.LayerNorm(dim),                      # 层归一化: 对最后一维做归一化, 稳定训练
        nn.Linear(dim, dim * mult * 2),         # 线性层: dim → dim*mult*2 (乘2是因为 GEGLU 会对半切分)
        GEGLU(),                                # GEGLU 激活: dim*mult*2 → dim*mult (对半切分后门控)
        nn.Dropout(dropout),                    # 随机丢弃: 防止过拟合
        nn.Linear(dim * mult, dim)              # 线性层: dim*mult → dim, 压缩回原始维度
    )

class Attention(Module):
    """多头自注意力机制 (Multi-Head Self-Attention)
       核心: 让每个 token 能"关注"序列中其他所有 token, 学习特征间的交互关系"""
    def __init__(
        self,
        dim,                                    # 输入/输出的特征维度
        heads = 8,                              # 注意力头数（并行计算多组注意力）
        dim_head = 64,                          # 每个注意力头的维度
        dropout = 0.                            # 注意力权重的 Dropout 概率
    ):
        super().__init__()                      # 调用父类初始化
        inner_dim = dim_head * heads            # 所有头的总维度 = 每头维度 × 头数
        self.heads = heads                      # 保存头数, 供 forward 中使用
        self.scale = dim_head ** -0.5           # 缩放因子 = 1/sqrt(dim_head), 防止点积值过大导致 softmax 梯度消失

        self.norm = nn.LayerNorm(dim)           # Pre-LN: 注意力之前先做层归一化

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # 一次性生成 Q、K、V 三个矩阵: dim → inner_dim*3
        self.to_out = nn.Linear(inner_dim, dim, bias = False)      # 输出投影: inner_dim → dim, 将多头结果合并回原维度

        self.dropout = nn.Dropout(dropout)      # 注意力权重的 Dropout

    def forward(self, x):
        """
        输入: x = (batch, seq_len, dim)  即 (b, n, dim)
        输出: out = (b, n, dim), attn = (b, heads, n, n)
        """
        h = self.heads                          # 取出头数, 方便后续使用

        x = self.norm(x)                        # Pre-LayerNorm: 先归一化再计算注意力, (b, n, dim)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)  # 线性投影后沿最后一维切成3份: 各为 (b, n, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # 重排维度: (b, n, heads*dim_head) → (b, heads, n, dim_head), 分离出多头
        q = q * self.scale                      # Q 乘以缩放因子 1/sqrt(dim_head), 控制点积量级

        sim = einsum('b h i d, b h j d -> b h i j', q, k)  # 计算注意力分数: Q·K^T, 结果 (b, heads, n, n)
        # sim[b,h,i,j] 表示第b个样本、第h个头中, 第i个token对第j个token的关注程度

        attn = sim.softmax(dim = -1)            # 沿最后一维做 softmax: 将分数归一化为注意力权重（概率分布）
        dropped_attn = self.dropout(attn)       # 对注意力权重施加 Dropout

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)  # 注意力加权求和: attn × V, 结果 (b, heads, n, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)  # 重排回原始形状: (b, heads, n, dim_head) → (b, n, inner_dim), 多头拼接
        out = self.to_out(out)                  # 输出投影: (b, n, inner_dim) → (b, n, dim)

        return out, attn                        # 返回注意力输出和注意力权重（权重可用于可视化分析）

# ===================== Transformer 主体 =====================

class Transformer(Module):
    """Transformer 编码器: 堆叠多层 [Attention + FeedForward], 每层都有残差连接"""
    def __init__(
        self,
        dim,                                    # token 的嵌入维度
        depth,                                  # Transformer 层数（堆叠深度）
        heads,                                  # 每层注意力的头数
        dim_head,                               # 每个注意力头的维度
        attn_dropout,                           # 注意力层的 Dropout 概率
        ff_dropout,                             # 前馈网络的 Dropout 概率
        num_residual_streams = 4                # 残差流数量（本实现中未使用, 仅保留接口兼容）
    ):
        super().__init__()                      # 调用父类初始化

        self.layers = ModuleList([])            # 存储所有 Transformer 层

        for _ in range(depth):                  # 循环创建 depth 层
            self.layers.append(ModuleList([     # 每层包含两个子模块:
                ResidualWrapper(branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                # 1. 带残差连接的多头注意力: x = x + Attention(x)
                ResidualWrapper(branch = FeedForward(dim, dropout = ff_dropout)),
                # 2. 带残差连接的前馈网络:   x = x + FFN(x)
            ]))

    def forward(self, x, return_attn = False):
        """
        输入: x = (b, seq_len, dim)
        输出: x = (b, seq_len, dim), 以及可选的注意力权重
        """
        post_softmax_attns = []                 # 收集每层的注意力权重, 用于可视化/分析

        for attn, ff in self.layers:            # 逐层处理: attn=残差注意力, ff=残差前馈
            x, post_softmax_attn = attn(x)      # 注意力层: 输出残差后的 x 和 softmax 后的注意力权重
            post_softmax_attns.append(post_softmax_attn)  # 保存本层的注意力权重

            x = ff(x)                           # 前馈网络层: 对注意力输出做非线性变换

        if not return_attn:                     # 如果不需要返回注意力权重
            return x                            # 只返回最终的 token 表示

        return x, torch.stack(post_softmax_attns)  # 返回 token 表示 + 所有层的注意力权重堆叠 (depth, b, heads, n, n)

# ===================== 数值特征嵌入器（备用, 本模型中由 SimpleEmbed 替代）=====================

class NumericalEmbedder(Module):
    """独立的数值特征嵌入器（与 SimpleEmbed 中数值部分逻辑相同）
       将每个数值特征通过 x * weight + bias 映射为 dim 维 token"""
    def __init__(self, dim, num_numerical_types):
        """参数: dim=嵌入维度, num_numerical_types=数值特征列数"""
        super().__init__()                      # 调用父类初始化
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))  # 可学习权重 (num_features, dim)
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))   # 可学习偏置 (num_features, dim)

    def forward(self, x):
        """输入: x = (batch, num_features), 输出: (batch, num_features, dim)"""
        x = rearrange(x, 'b n -> b n 1')       # (b, n) → (b, n, 1), 增加维度用于广播
        return x * self.weights + self.biases   # (b, n, 1) * (n, dim) + (n, dim) → (b, n, dim), 每个数值特征→一个token

# ===================== FT-Transformer 主模型 =====================

class FTTransformer(Module):
    """FT-Transformer (Feature Tokenizer + Transformer)
       核心思想: 将表格数据的每列特征都转为一个 token, 然后用 Transformer 学习特征间交互
       论文: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)"""
    def __init__(
        self,
        *,                                      # 星号后的参数必须用关键字传入
        categories,                             # 元组: 每个分类特征的类别数, 如 (3, 5, 2) 表示3个分类特征
        num_continuous,                         # 整数: 数值型特征的列数
        dim,                                    # 整数: 每个 token 的嵌入维度（Transformer 的隐藏维度）
        depth,                                  # 整数: Transformer 层数（堆叠深度）
        heads,                                  # 整数: 多头注意力的头数
        dim_head = 16,                          # 整数: 每个注意力头的维度
        dim_out = 1,                            # 整数: 输出维度（分类数, 二分类设为2）
        num_special_tokens = 2,                 # 整数: 特殊 token 数量（用于填充/未知类别, 避免 ID 冲突）
        attn_dropout = 0.,                      # 浮点: 注意力层 Dropout 概率
        ff_dropout = 0.,                        # 浮点: 前馈网络 Dropout 概率
        num_residual_streams = 4                # 整数: 残差流数量（保留接口, 实际未用）
    ):
        super().__init__()                      # 调用父类初始化
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        # 断言: 每个分类特征的类别数必须 > 0
        assert len(categories) + num_continuous > 0, 'input shape must not be null'
        # 断言: 至少要有一个特征（分类或数值）

        # ---- 分类特征相关计算 ----

        self.num_categories = len(categories)           # 分类特征的列数
        self.num_unique_categories = sum(categories)    # 所有分类特征的类别总数（用于统计）

        # ---- 创建分类嵌入表 ----

        self.num_special_tokens = num_special_tokens    # 保存特殊 token 数量, forward 中要用
        total_tokens = self.num_unique_categories + num_special_tokens  # 总 token 数（含特殊 token）

        # ---- 构建嵌入层 ----

        categories_with_special = tuple(c + num_special_tokens for c in categories)
        # 每个分类特征的嵌入表大小 = 原始类别数 + 特殊 token 数
        # 例如原来有3个类别, 加2个特殊token后, Embedding 表大小为5

        self.embedding = SimpleEmbed(dim, num_discrete = categories_with_special, num_continuous = num_continuous)
        # 创建嵌入层: 将所有分类特征和数值特征分别映射为 dim 维 token

        # ---- CLS token ----

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # 可学习的 [CLS] token, 形状 (1, 1, dim)
        # 作用: 作为全局聚合 token, Transformer 处理后用它的输出做最终分类

        # ---- Transformer 编码器 ----

        self.transformer = Transformer(
            dim = dim,                          # token 维度
            depth = depth,                      # 层数
            heads = heads,                      # 注意力头数
            dim_head = dim_head,                # 每头维度
            attn_dropout = attn_dropout,        # 注意力 Dropout
            ff_dropout = ff_dropout,            # 前馈网络 Dropout
            num_residual_streams = num_residual_streams  # 残差流（保留接口）
        )

        # ---- 分类头 ----

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),                  # 层归一化: 稳定 CLS token 的输出分布
            nn.ReLU(),                          # ReLU 激活: 引入非线性
            nn.Linear(dim, dim_out)             # 线性层: dim → dim_out, 输出分类 logits
        )
        # 论文中的分类头结构: Linear(ReLU(LayerNorm(cls_token)))

    def forward(self, x_categ, x_numer, return_attn = False, return_features = False):
        """
        前向传播
        参数:
            x_categ:         分类特征, (batch, num_categories), 整数类型
            x_numer:         数值特征, (batch, num_continuous), 浮点类型
            return_attn:     是否同时返回注意力权重（用于可视化）
            return_features: 是否返回 CLS token 特征（用于双分支融合）
        返回:
            logits:      分类结果, (batch, dim_out)
            attns:       (可选) 各层注意力权重, (depth, batch, heads, seq_len, seq_len)
            features:    (可选) CLS token 特征, (batch, dim)
        """
        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'
        # 断言: 输入的分类特征列数必须与模型定义时一致

        x_categ = x_categ + self.num_special_tokens
        # 将分类特征的 ID 整体偏移, 避免与特殊 token（ID=0,1）冲突
        # 例如原始编码 [0,1,2] → 偏移后 [2,3,4], 留出 0,1 给特殊 token

        x = self.embedding((x_categ, x_numer))
        # 嵌入: 分类特征→Embedding查表, 数值特征→线性投影
        # 输出: (batch, num_categories + num_continuous, dim) 即每个特征一个 token

        # ---- 拼接 [CLS] token ----

        b = x.shape[0]                          # 取 batch 大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)  # 将 CLS token 复制 batch 份: (1,1,dim) → (b,1,dim)
        x = torch.cat((cls_tokens, x), dim = 1)  # 在 token 序列最前面拼接 CLS: (b, 1+C_cat+C_num, dim)

        # ---- Transformer 编码 ----

        x, attns = self.transformer(x, return_attn = True)
        # 经过 depth 层 [Attention + FFN] 处理
        # x: (b, 1+C_cat+C_num, dim), attns: (depth, b, heads, seq_len, seq_len)

        # ---- 提取 [CLS] token 的输出 ----

        x = x[:, 0]                            # 取序列中第0个位置, 即 [CLS] token 的最终表示: (b, dim)

        # ---- 如果只需要特征（用于双分支融合），在分类头之前返回 ----

        if return_features:                     # 双分支融合模式
            return x                            # 返回 CLS token 特征: (b, dim)

        # ---- 分类头: Linear(ReLU(LayerNorm(cls))) ----

        logits = self.to_logits(x)              # (b, dim) → LayerNorm → ReLU → Linear → (b, dim_out)

        if not return_attn:                     # 如果不需要注意力权重
            return logits                       # 只返回分类 logits

        return logits, attns                    # 返回 logits 和注意力权重
