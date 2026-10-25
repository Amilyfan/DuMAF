# Copyright (c) 2015-present, Facebook, Inc.
#
#
import math
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import random

from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from mamba_simple62 import Mamba

from cbam_module import CBAMFor2D, CBAM
# =====================================

from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

__all__ = [
    'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2',
]

# class SerpentineChannelPatchEmbedUltraOptimized(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=448, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
#                   cbam_reduction_ratio=16, cbam_kernel_size=7):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten
#
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
# ==================================================================================
# ==================================================================================
class SerpentineChannelPatchEmbedUltraOptimized(nn.Module):

    def __init__(self, img_size=448, patch_size=16, stride=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True,
                 use_image_cbam=False,
                 use_patch_cbam=False,
                 cbam_reduction_ratio=16, cbam_kernel_size=7):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        self.use_image_cbam = use_image_cbam
        self.use_patch_cbam = use_patch_cbam

        self.grid_size = (
            (self.img_size[0] - self.patch_size[0]) // self.stride[0] + 1,
            (self.img_size[1] - self.patch_size[1]) // self.stride[1] + 1
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * in_chans
        self._precompute_vectorized_indices()

        patch_dim = self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def _precompute_vectorized_indices(self):
        grid_h, grid_w = self.grid_size

        row_indices = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w)
        col_indices = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w)

        reverse_mask = torch.arange(grid_h).view(-1, 1) % 2 == 1
        reverse_mask = reverse_mask.expand(grid_h, grid_w)

        col_indices = torch.where(
            reverse_mask,
            grid_w - 1 - col_indices,
            col_indices
        )

        serpentine_rows = row_indices.reshape(-1)
        serpentine_cols = col_indices.reshape(-1)

        num_spatial = grid_h * grid_w

        spatial_indices = torch.arange(num_spatial)
        spatial_indices_repeated = spatial_indices.repeat_interleave(self.in_chans)

        base_channels = torch.arange(self.in_chans)
        channel_indices = base_channels.repeat(num_spatial)

        should_reverse = spatial_indices_repeated % 2 == 1
        channel_indices = torch.where(
            should_reverse,
            self.in_chans - 1 - channel_indices,
            channel_indices
        )

        row_indices_final = serpentine_rows.repeat_interleave(self.in_chans)
        col_indices_final = serpentine_cols.repeat_interleave(self.in_chans)

        self.register_buffer('gather_channel_idx', channel_indices.long())
        self.register_buffer('gather_row_idx', row_indices_final.long())
        self.register_buffer('gather_col_idx', col_indices_final.long())

        print(f"\n{'=' * 80}")
        print(f"🚀 蛇形扫描Patch Embedding - 在切片前应用CBAM版本")
        print(f"{'=' * 80}")
        print(f"  图像尺寸: {self.img_size}")
        print(f"  Patch尺寸: {self.patch_size}")
        print(f"  网格尺寸: {grid_h} x {grid_w}")
        print(f"  总Patch数: {self.num_patches}")
        print(f"  处理流程: 输入图像 → Image CBAM → 切片 → Patch CBAM (可选) → 投影")
        print(f"  ✅ 完全向量化（无循环）")
        print(f"  ✅ 预计算索引（forward时0开销）")
        print(f"{'=' * 80}\n")

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_chans and H == self.img_size[0] and W == self.img_size[1], \
            f"输入形状 {x.shape} 与期望的 (B, {self.in_chans}, {self.img_size[0]}, {self.img_size[1]}) 不匹配"


        patches = self._extract_patches_ultra_optimized(x)  # (B, num_patches, patch_dim)
        # =====================================================

        patches = self.proj(patches)  # (B, num_patches, embed_dim)
        patches = self.norm(patches)
        # =====================================================

        return patches

    def _extract_patches_ultra_optimized(self, x):
        B, C, H, W = x.shape
        patch_h, patch_w = self.patch_size
        stride_h, stride_w = self.stride
        patch_dim = patch_h * patch_w

        patches = x.unfold(2, patch_h, stride_h).unfold(3, patch_w, stride_w)

        patches = patches.contiguous().view(B, C, self.grid_size[0], self.grid_size[1], -1)

        batch_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, self.num_patches)

        ordered_patches = patches[
                          batch_idx,
                          self.gather_channel_idx.unsqueeze(0).expand(B, -1),
                          self.gather_row_idx.unsqueeze(0).expand(B, -1),
                          self.gather_col_idx.unsqueeze(0).expand(B, -1),
                          :
                          ]  # (B, num_patches, patch_dim)

        return ordered_patches


# ==================================================================================
# ==================================================================================
class Block(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self,
            hidden_states: Tensor,
            residual: Optional[Tensor] = None,
            inference_params=None,
            auxiliary_states: Optional[Tensor] = None
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, auxiliary_states=auxiliary_states)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        if_bimamba=False,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
        adaptive_fusion=True,
        use_text_fusion=False,
):
    if if_bimamba:
        bimamba_type = "v2"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        adaptive_fusion=adaptive_fusion,
        use_text_fusion=use_text_fusion,
        **ssm_cfg,
        **factory_kwargs
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# ==================================================================================
# ==================================================================================
class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=448,
            patch_size=16,
            stride=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            ssm_cfg=None,
            drop_rate=0.,
            drop_path_rate=0.1,#0.1
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
            ft_seq_len=None,
            pt_hw_seq_len=14,
            if_bidirectional=False,
            final_pool_type='none',
            if_abs_pos_embed=False,
            if_rope=False,
            if_rope_residual=False,
            flip_img_sequences_ratio=-1.,
            if_bimamba=False,
            bimamba_type="none",
            if_cls_token=True,
            if_divide_out=False,
            init_layer_scale=None,
            use_double_cls_token=False,
            use_middle_cls_token=False,
            adaptive_fusion=True,
            use_text_fusion=False,
            text_feature_dim=37,
            use_image_cbam=True,
            use_patch_cbam=False,
            use_cbam=False,
            cbam_position='after_patch_embed',
            cbam_reduction_ratio=16,
            cbam_kernel_size=7,
            use_channel_attention=True,
            use_spatial_attention=True,
            **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.use_text_fusion = use_text_fusion
        self.embed_dim = embed_dim

        self.use_image_cbam = use_image_cbam
        self.use_patch_cbam = use_patch_cbam
        self.use_cbam = use_cbam
        self.cbam_position = cbam_position
        # ==============================

        self.patch_embed = SerpentineChannelPatchEmbedUltraOptimized(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
            use_image_cbam=use_image_cbam,
            use_patch_cbam=use_patch_cbam,
            cbam_reduction_ratio=cbam_reduction_ratio,
            cbam_kernel_size=cbam_kernel_size
        )
        num_patches = self.patch_embed.num_patches

        if self.use_text_fusion:
            self.projection = nn.Linear(text_feature_dim, 64, bias=False)
            self.attention = nn.MultiheadAttention(
                embed_dim=64,
                num_heads=4,
                dropout=0,
                batch_first=True
            )
            self.text_proj = nn.Sequential(
                nn.Linear(text_feature_dim,embed_dim * 2),
                nn.LayerNorm(embed_dim*2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),

            )
            # self.text_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            # trunc_normal_(self.text_pos_embed, std=.02)

        # ========== CLS Token ==========
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                self.num_tokens = 1

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim))
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if ft_seq_len is None else ft_seq_len
            )

        # ========== Mamba Blocks ==========
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    adaptive_fusion=adaptive_fusion,
                    use_text_fusion=use_text_fusion,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.use_multiscale_cbam = kwargs.get('use_multiscale_cbam', True)
        if self.use_multiscale_cbam:
            self.cbam_scale1 = CBAMFor2D(in_channels=channels, reduction_ratio=cbam_reduction_ratio,
                                         kernel_size=cbam_kernel_size)
            self.cbam_scale2 = CBAMFor2D(in_channels=channels, reduction_ratio=cbam_reduction_ratio,
                                         kernel_size=cbam_kernel_size)
            self.cbam_scale3 = CBAMFor2D(in_channels=channels, reduction_ratio=cbam_reduction_ratio,
                                         kernel_size=cbam_kernel_size)
            print(f"\n{'=' * 80}")
            print(f"  ✅ 多尺度CBAM已启用 (作用在原始图像的不同尺度)")
            print(f"     - Scale 1: 128×128 区域 (右上角)")
            print(f"     - Scale 2: 256×256 区域 (右上角)")
            print(f"     - Scale 3: 448×448 区域 (完整图像)")
            print(f"     - 融合方式: out = out + upsample_add(upsample_add(f1, f2), f3)")
            print(f"{'=' * 80}\n")
        # =============================================

        if use_cbam:
            if cbam_position in ['after_patch_embed', 'both']:
                self.cbam_patch = CBAM(
                    in_channels=embed_dim,
                    reduction_ratio=cbam_reduction_ratio,
                    kernel_size=cbam_kernel_size,
                    use_channel_attention=use_channel_attention,
                    use_spatial_attention=use_spatial_attention
                )
                print(f"  ✅ 序列CBAM已启用 (after_patch_embed位置)")

            if cbam_position in ['after_blocks', 'both']:
                self.cbam_final = CBAM(
                    in_channels=embed_dim,
                    reduction_ratio=cbam_reduction_ratio,
                    kernel_size=cbam_kernel_size,
                    use_channel_attention=use_channel_attention,
                    use_spatial_attention=use_spatial_attention
                )
                print(f"  ✅ 序列CBAM已启用 (after_blocks位置)")
        # ==========================================

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=.02)
                trunc_normal_(self.cls_token_tail, std=.02)
            else:
                trunc_normal_(self.cls_token, std=.02)

        self.apply(
            partial(
                self._init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def _upsample_add(self, x, y):
        # _, _, H, W = y.size()
        # return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y
        B, C, H_small, W_small = x.size()
        _, _, H_large, W_large = y.size()

        scale_h = H_large // H_small
        scale_w = W_large // W_small

        upsampled = torch.zeros(B, C, H_large, W_large, device=x.device, dtype=x.dtype)

        upsampled[:, :, ::scale_h, ::scale_w] = x

        return upsampled+ y#0.4 0.001

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}
        #return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail", "text_pos_embed"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def _init_weights(
            self,
            module,
            n_layer,
            initializer_range=0.02,
            rescale_prenorm_residual=True,
            n_residuals_per_layer=1,
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)

    def forward_features(self, x, text_features=None, inference_params=None):
        B = x.shape[0]

        if self.use_multiscale_cbam:
            B, C, H, W = x.shape
            center_h, center_w = H // 2, W // 2

            size1 = 56
            start_h1 = center_h - size1 // 2  # 224 - 32 = 192
            start_w1 = center_w - size1 // 2
            feature1 = x[:, :, start_h1:start_h1 + size1, start_w1:start_w1 + size1]  # [B, 3, 64, 64]

            size2 = 112
            start_h2 = center_h - size2 // 2  # 224 - 75 = 149
            start_w2 = center_w - size2 // 2
            feature2 = x[:, :, start_h2:start_h2 + size2, start_w2:start_w2 + size2]  # [B, 3, 150, 150]

            feature3 = x  # [B, 3, 448, 448]


            f1_out = self.cbam_scale1(feature1)  # [B, 3, 128, 128]
            f2_out = self.cbam_scale2(feature2)  # [B, 3, 256, 256]
            f3_out = self.cbam_scale3(feature3)  # [B, 3, 448, 448]

            x = self._upsample_add(
                self._upsample_add(f1_out, f2_out),
                f3_out
            )
        # ====================================================================

        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # ====================================================================

        if self.use_cbam and self.cbam_position in ['after_patch_embed', 'both']:
            x = self.cbam_patch(x)
        # =====================================================

        auxiliary_features = None
        if self.use_text_fusion and text_features is not None:

            # projected = self.projection(text_features)
            # projected2 = projected.unsqueeze(1)
            # attn_output, _ = self.attention(projected2, projected2, projected2)
            # final_text_feature = attn_output.squeeze(1)+projected
            text_emb1 = self.text_proj(text_features)#final_text_feature
            num_patches = x.shape[1]
            text_emb = text_emb1.unsqueeze(1).expand(-1, num_patches, -1)
            #text_emb = text_emb + self.text_pos_embed
            auxiliary_features = text_emb

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, x.shape[1] + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]
                if auxiliary_features is not None:
                    aux_cls_head = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
                    aux_cls_tail = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
                    auxiliary_features = torch.cat((aux_cls_head, auxiliary_features, aux_cls_tail), dim=1)
            else:
                cls_token = self.cls_token.expand(B, -1, -1)
                token_position = 0
                x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]
                if auxiliary_features is not None:
                    aux_cls = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
                    auxiliary_features = torch.cat((aux_cls, auxiliary_features), dim=1)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual,
                inference_params=inference_params,
                auxiliary_states=auxiliary_features if self.use_text_fusion else None
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        if self.use_cbam and self.cbam_position in ['after_blocks', 'both']:
            hidden_states = self.cbam_final(hidden_states)
        # ==============================================

        if self.final_pool_type == 'none':
            if self.if_cls_token:
                if self.use_double_cls_token:
                    return (hidden_states[:, token_position[0]] + hidden_states[:, token_position[1]]) / 2
                else:
                    return hidden_states[:, token_position]
            else:
                return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)[0]
        else:
            raise NotImplementedError

    def forward(self, x, text_features=None, return_features=False, inference_params=None):
        x = self.forward_features(x, text_features, inference_params)
        if return_features:
            return x
        x = self.head(x)
        return x


@register_model
def vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=kwargs.pop("depth", 4),
        d_state=16,
        stride=16,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type='mean',
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="v2",
        if_cls_token=True,
        if_divide_out=True,
        use_middle_cls_token=False,
        use_image_cbam=False,
        use_patch_cbam=False,
        use_cbam=False,
        cbam_position='after_patch_embed',
        use_multiscale_cbam=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="to.do",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
