import torch
import torch.nn as nn
from ft_transformer import FTTransformer
from model_mamba7CBAM4mutiscalek import (
    vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
)


class ScoringSystem(nn.Module):
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
        return torch.sigmoid(self.fc(x))


class DualBranchModel(nn.Module):
    DEPTH = None
    NUM_SCORERS = None
    FUSION_DIM = None

    def __init__(
        self,
        categories,
        num_continuous,
        ft_dim=32,
        ft_depth=None,
        ft_heads=4,
        ft_dim_head=8,
        ft_attn_dropout=0.1,
        ft_ff_dropout=0.1,
        vim_embed_dim=192,
        vim_num_classes=2,
        text_feature_dim=37,
        fusion_dim=None,
        num_scorers=None,
        num_classes=2,
        *,
        vim_depth=None,
    ):
        super().__init__()

        if fusion_dim is None:
            fusion_dim = self.FUSION_DIM if self.FUSION_DIM is not None else 32
        if type(fusion_dim) is not int or fusion_dim not in (32, 192):
            raise ValueError(f"fusion_dim must be 32 or 192, got {fusion_dim!r}")
        if self.FUSION_DIM is not None and fusion_dim != self.FUSION_DIM:
            raise ValueError(
                f"{type(self).__name__} requires fusion_dim={self.FUSION_DIM}, "
                f"got {fusion_dim}"
            )

        if self.NUM_SCORERS is not None:
            if num_scorers is not None and num_scorers != self.NUM_SCORERS:
                raise ValueError(
                    f"{type(self).__name__} requires num_scorers={self.NUM_SCORERS}, "
                    f"got {num_scorers}"
                )
            num_scorers = self.NUM_SCORERS
        elif num_scorers is None:
            num_scorers = 3

        if self.DEPTH is not None:
            for name, value in (("ft_depth", ft_depth), ("vim_depth", vim_depth)):
                if value is not None and value != self.DEPTH:
                    raise ValueError(
                        f"{type(self).__name__} requires {name}={self.DEPTH}, got {value}"
                    )
            ft_depth = vim_depth = self.DEPTH
        else:
            ft_depth = 4 if ft_depth is None else ft_depth
            vim_depth = 4 if vim_depth is None else vim_depth

        self.ft_depth = ft_depth
        self.vim_depth = vim_depth
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        self.num_scorers = num_scorers

        self.vision_branch = vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(
            pretrained=False,
            depth=vim_depth,
            num_classes=vim_num_classes,
            use_text_fusion=True,
            text_feature_dim=text_feature_dim,
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vim_embed_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

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

        self.scorers = nn.ModuleList([
            ScoringSystem(in_dim=fusion_dim * 2, out_dim=fusion_dim)
            for _ in range(num_scorers)
        ])

        self.fusion_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_dim, fusion_dim),
            )
            for _ in range(num_scorers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, num_classes),
        )

        print(f"\n{'=' * 80}")
        print(f"{type(self).__name__} (多Scorer + 每路独立MLP非线性聚合) 初始化完成")
        print(f"   分支1 (VisionMamba):   depth={vim_depth}, embed_dim={vim_embed_dim} -> {fusion_dim}")
        print(f"   分支2 (FTTransformer): depth={ft_depth}, dim={ft_dim} -> {fusion_dim}")
        print(f"   打分系统: {num_scorers} 个独立 Scorer, 输入={fusion_dim*2}, 输出权重={fusion_dim}")
        print(f"   每路融合: mlp_i(v*w_i + (1-w_i)*t), {num_scorers} 路求平均 -> {fusion_dim} -> {num_classes} classes")
        print(f"{'=' * 80}\n")

    def forward(self, images, features, x_categ, x_numer):
        vision_feat = self.vision_branch(images, text_features=features, return_features=True)
        vision_feat = self.vision_proj(vision_feat)        # (B, fusion_dim)

        tabular_feat = self.tabular_branch(x_categ, x_numer, return_features=True)
        tabular_feat = self.tabular_proj(tabular_feat)     # (B, fusion_dim)

        concat_feat = torch.cat([vision_feat, tabular_feat], dim=1)  # (B, 2*fusion_dim)

        fused = torch.zeros_like(vision_feat)              # (B, fusion_dim)
        for scorer, mlp in zip(self.scorers, self.fusion_mlps):
            w_i = scorer(concat_feat)                      # (B, fusion_dim), ∈(0,1)
            fusion_i = vision_feat * w_i + (1 - w_i) * tabular_feat   # (B, fusion_dim)
            fused = fused + mlp(fusion_i)

        fused = fused / self.num_scorers

        logits = self.classifier(fused)                    # (B, num_classes)
        return logits


class DualBranchModelA(DualBranchModel):

    DEPTH = 4


class DualBranchModelB(DualBranchModel):

    DEPTH = 8


class DualBranchModelC(DualBranchModel):

    DEPTH = 12


class DualBranchModelA_3(DualBranchModelA):

    NUM_SCORERS = 3


class DualBranchModelA_4(DualBranchModelA):

    NUM_SCORERS = 4


class DualBranchModelB_3(DualBranchModelB):

    NUM_SCORERS = 3


class DualBranchModelB_4(DualBranchModelB):

    NUM_SCORERS = 4


class DualBranchModelC_3(DualBranchModelC):

    NUM_SCORERS = 3


class DualBranchModelC_4(DualBranchModelC):

    NUM_SCORERS = 4


class DualBranchModelA_3_32(DualBranchModelA_3):

    FUSION_DIM = 32


class DualBranchModelA_3_192(DualBranchModelA_3):

    FUSION_DIM = 192


class DualBranchModelA_4_32(DualBranchModelA_4):

    FUSION_DIM = 32


class DualBranchModelA_4_192(DualBranchModelA_4):

    FUSION_DIM = 192


class DualBranchModelB_3_32(DualBranchModelB_3):

    FUSION_DIM = 32


class DualBranchModelB_3_192(DualBranchModelB_3):

    FUSION_DIM = 192


class DualBranchModelB_4_32(DualBranchModelB_4):

    FUSION_DIM = 32


class DualBranchModelB_4_192(DualBranchModelB_4):

    FUSION_DIM = 192


class DualBranchModelC_3_32(DualBranchModelC_3):

    FUSION_DIM = 32


class DualBranchModelC_3_192(DualBranchModelC_3):

    FUSION_DIM = 192


class DualBranchModelC_4_32(DualBranchModelC_4):

    FUSION_DIM = 32


class DualBranchModelC_4_192(DualBranchModelC_4):

    FUSION_DIM = 192
