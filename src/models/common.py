"""Common model abstractions, metadata extraction, and head replacement utilities."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculate total, trainable, and non-trainable parameter counts.

    Args:
        model: PyTorch model module.

    Returns:
        Dictionary containing 'total_params', 'trainable_params', and 'non_trainable_params'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
    }


def estimate_model_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Estimate model floating point operations (GFLOPs) using forward hook profiling or fvcore/ptflops if available.

    Args:
        model: PyTorch neural network module.
        input_size: (B, C, H, W) tensor dimensions.
        device: Device to place tensor on.

    Returns:
        Estimated operations in GFLOPs (1 GFLOP = 1e9 FLOPs).
    """
    from typing import Union
    dev = torch.device(device) if device is not None else torch.device("cpu")
    model_eval = model.to(dev).eval()
    
    # Try fvcore FlopCountAnalysis first if available
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(*input_size, device=dev)
        flops = FlopCountAnalysis(model_eval, dummy_input).total()
        return round(float(flops) / 1e9, 3)
    except Exception:
        pass

    # Analytical fallback using layer hook estimation for standard Conv2d, Linear, LayerNorm, BatchNorm
    total_flops = 0.0
    hooks = []

    def conv_hook(module: nn.Conv2d, input: Any, output: torch.Tensor) -> None:
        nonlocal total_flops
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        flops_per_instance = 2 * kernel_ops
        num_instances = out_h * out_w * module.out_channels
        total_flops += flops_per_instance * num_instances

    def linear_hook(module: nn.Linear, input: Any, output: torch.Tensor) -> None:
        nonlocal total_flops
        total_flops += 2 * module.in_features * module.out_features

    for m in model_eval.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    dummy_input = torch.randn(*input_size, device=dev)
    with torch.no_grad():
        try:
            model_eval(dummy_input)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    if total_flops > 0:
        return round(total_flops / 1e9, 3)

    # Heuristic fallback based on parameter count if hooks could not track custom layers
    params = sum(p.numel() for p in model.parameters())
    # Standard rule of thumb: ~2 FLOPs per param per pixel reduction factor
    return round((params * 2.0 * (224 * 224 / (16 * 16))) / 1e9, 3)


def replace_classification_head(
    model: nn.Module,
    num_classes: int,
    head_attr_names: Optional[List[str]] = None,
) -> nn.Module:
    """Safely replace classification head for arbitrary torchvision / timm model.

    Args:
        model: Target PyTorch model.
        num_classes: New target class count.
        head_attr_names: Candidate attribute names for classifier head.

    Returns:
        Model with replaced classifier head.
    """
    if head_attr_names is None:
        head_attr_names = ["fc", "classifier", "head", "heads"]

    for attr in head_attr_names:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                in_feat = layer.in_features
                setattr(model, attr, nn.Linear(in_feat, num_classes))
                return model
            elif isinstance(layer, nn.Sequential) and len(layer) > 0:
                last_idx = len(layer) - 1
                if isinstance(layer[last_idx], nn.Linear):
                    in_feat = layer[last_idx].in_features
                    layer[last_idx] = nn.Linear(in_feat, num_classes)
                    return model

    # Timm standard reset_classifier if present
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        return model

    return model
