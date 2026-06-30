import torch
import torch.nn as nn
from transformer import FTTransformer
from mamba import (
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
)


class ScoringSystem(nn.Module):
    """
    单个打分系统: Linear -> GELU -> Dropout -> Linear -> Sigmoid
    输入 concat 特征 (B, 2*fusion_dim), 输出权重 w (B, fusion_dim), 每通道独立 ∈ (0,1).
    w 用于加权融合: fusion = vision * w + (1 - w) * tabular
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, in_dim)
        return torch.sigmoid(self.fc(x))   # (B, out_dim), 元素独立 ∈ (0,1)


class DualBranchModel(nn.Module):
    """
    双分支融合模型 (多 Scorer + 每路独立 MLP 非线性聚合):
      分支1 (图像分支): VisionMamba + 表格辅助 -> 192 -> 投影到 fusion_dim
      分支2 (表格分支): FTTransformer (区分类别/数值) -> ft_dim -> 投影到 fusion_dim
      融合:
        concat([v, t]) -> N 个独立 Scorer -> 权重 w_i
        fusion_i = v * w_i + (1 - w_i) * t
        out_i    = mlp_i(fusion_i)              # ← 关键: 每路独立非线性, 打破线性退化
        fused    = mean_i(out_i)                # N 路求平均
        logits   = classifier(fused)

    与 model2.py 的区别:
      model2.py 中 fused = mean_i(v*w_i + (1-w_i)*t) = v*mean(w_i) + (1-mean(w_i))*t,
      数学上等价于单个 Scorer (mean(w_i)), N 路在求和阶段被代数坍缩.
      本文件在每路输出上插入独立 MLP(mlp_i), 求和不能再提出, N 路真正起作用.
    """
    def __init__(
        self,
        # ---- FTTransformer 参数 ----
        categories,
        num_continuous,
        ft_dim=32,
        ft_depth=4,
        ft_heads=4,
        ft_dim_head=8,
        ft_attn_dropout=0.1,
        ft_ff_dropout=0.1,
        # ---- VisionMamba 参数 ----
        vim_embed_dim=192,
        vim_num_classes=2,
        text_feature_dim=37,
        # ---- 融合参数 ----
        fusion_dim=32,
        num_scorers=3,
        num_classes=2,
    ):
        super().__init__()

        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.num_scorers = num_scorers

        # =============== 分支1: VisionMamba (图像 + 表格辅助) ===============
        self.vision_branch = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
            pretrained=False,
            num_classes=vim_num_classes,
            use_text_features=True,
            text_feature_dim=text_feature_dim,
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vim_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # =============== 分支2: FTTransformer (表格, 区分类别/数值) ===============
        self.tabular_branch = FTTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=ft_dim,
            depth=ft_depth,
            heads=ft_heads,
            dim_head=ft_dim_head,
            dim_out=num_classes,
            attn_dropout=ft_attn_dropout,
            ff_dropout=ft_ff_dropout,
        )
        self.tabular_proj = nn.Sequential(
            nn.Linear(ft_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # =============== N 个独立打分系统 ===============
        # 输入: concat(vision, tabular) = 2*fusion_dim
        # 输出: 权重 w_i (fusion_dim 维)
        self.scorers = nn.ModuleList([
            ScoringSystem(in_dim=fusion_dim * 2, out_dim=fusion_dim)
            for _ in range(num_scorers)
        ])

        # =============== N 个独立 fusion MLP (每路非线性, 打破线性退化) ===============
        # 输入/输出均为 fusion_dim 维
        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
            )
            for _ in range(num_scorers)
        ])

        # =============== 分类头 ===============
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, num_classes),
        )

        print(f"\n{'=' * 80}")
        print("DualBranchModel (多Scorer + 每路独立MLP非线性聚合) 初始化完成")
        print(f"   分支1 (VisionMamba):   embed_dim={vim_embed_dim} -> {fusion_dim}")
        print(f"   分支2 (FTTransformer): dim={ft_dim} -> {fusion_dim}")
        print(f"   打分系统: {num_scorers} 个独立 Scorer, 输入={fusion_dim*2}, 输出权重={fusion_dim}")
        print(f"   每路融合: mlp_i(v*w_i + (1-w_i)*t), {num_scorers} 路求平均 -> {fusion_dim} -> {num_classes} classes")
        print(f"{'=' * 80}\n")

    def forward(self, images, features, x_categ, x_numer):
        """
        参数:
            images:   (B, 3, H, W)
            features: (B, NUM_FEATURE_COLS) 完整表格特征, 给 VisionMamba 做辅助
            x_categ:  (B, num_cat_cols) 分类特征(整数)
            x_numer:  (B, num_num_cols) 数值特征(浮点)
        返回:
            logits: (B, num_classes)
        """
        # ---- 分支1: VisionMamba (图像 + 表格辅助) ----
        vision_feat = self.vision_branch(images, text_features=features, return_features=True)
        vision_feat = self.vision_proj(vision_feat)        # (B, fusion_dim)

        # ---- 分支2: FTTransformer (表格, 区分类别/数值) ----
        tabular_feat = self.tabular_branch(x_categ, x_numer, return_features=True)
        tabular_feat = self.tabular_proj(tabular_feat)     # (B, fusion_dim)

        # ---- concat 作为打分系统的输入 ----
        concat_feat = torch.cat([vision_feat, tabular_feat], dim=1)  # (B, 2*fusion_dim)

        # ---- N 个打分系统生成权重并加权融合, 每路独立 MLP 非线性 ----
        fused = torch.zeros_like(vision_feat)              # (B, fusion_dim)
        for scorer, mlp in zip(self.scorers, self.fusion_mlps):
            w_i = scorer(concat_feat)                      # (B, fusion_dim), ∈(0,1)
            fusion_i = vision_feat * w_i + (1 - w_i) * tabular_feat   # (B, fusion_dim)
            fused = fused + mlp(fusion_i)                  # ← 关键: 独立非线性后累加

        fused = fused / self.num_scorers                   # 求平均

        # ---- 分类 ----
        logits = self.classifier(fused)                    # (B, num_classes)
        return logits
