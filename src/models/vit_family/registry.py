"""Vision Transformer Family Registry mapping names and aliases to model builders."""

from typing import Any, Callable, Dict
import torch.nn as nn

from src.models.vit_family.swin import (
    SwinTransformerModel,
    build_swin_s,
    build_swin_t,
)
from src.models.vit_family.vit import (
    VisionTransformerModel,
    build_deit_b16,
    build_deit_s16,
    build_vit_b16,
    build_vit_s16,
    build_vit_ti16,
)

VIT_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    # ViT-B/16
    "vit_b16": build_vit_b16,
    "vit-b/16": build_vit_b16,
    "vit_b_16": build_vit_b16,
    "vit_base_patch16_224": build_vit_b16,
    "vit_base": build_vit_b16,

    # ViT-S/16
    "vit_s16": build_vit_s16,
    "vit-s/16": build_vit_s16,
    "vit_s_16": build_vit_s16,
    "vit_small_patch16_224": build_vit_s16,
    "vit_small": build_vit_s16,

    # ViT-Ti/16
    "vit_ti16": build_vit_ti16,
    "vit-ti/16": build_vit_ti16,
    "vit_ti_16": build_vit_ti16,
    "vit_tiny_patch16_224": build_vit_ti16,
    "vit_tiny": build_vit_ti16,

    # Swin-T
    "swin_t": build_swin_t,
    "swin-t": build_swin_t,
    "swin_tiny": build_swin_t,
    "swin_tiny_patch4_window7_224": build_swin_t,

    # Swin-S
    "swin_s": build_swin_s,
    "swin-s": build_swin_s,
    "swin_small": build_swin_s,
    "swin_small_patch4_window7_224": build_swin_s,

    # DeiT-S/16
    "deit_s16": build_deit_s16,
    "deit-s/16": build_deit_s16,
    "deit_small": build_deit_s16,
    "deit_small_patch16_224": build_deit_s16,

    # DeiT-B/16
    "deit_b16": build_deit_b16,
    "deit-b/16": build_deit_b16,
    "deit_base": build_deit_b16,
    "deit_base_patch16_224": build_deit_b16,
}
