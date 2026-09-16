"""Computational complexity and inference latency profiling for CNNs and Vision Transformers."""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.common import count_parameters, estimate_model_flops
from src.utils.device import get_device, synchronize_device


def profile_model_hardware(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: Optional[Union[str, torch.device]] = None,
    warmup_iters: int = 10,
    measure_iters: int = 30,
    batch_size_alt: int = 32,
) -> Dict[str, Any]:
    """Profile model computational complexity, memory footprint, latency percentiles, and throughput.

    Args:
        model: PyTorch model in eval mode.
        input_size: Standard input tensor shape (1, 3, 224, 224).
        device: PyTorch device ('cuda' or 'cpu').
        warmup_iters: Number of unmeasured warmup forward passes.
        measure_iters: Number of timed forward passes.
        batch_size_alt: Alternative batch size for multi-sample throughput profiling.

    Returns:
        Structured hardware profiling dictionary.
    """
    dev = get_device(device)
    model = model.to(dev).eval()

    # 1. Parameter counting & Memory Size
    params_dict = count_parameters(model)
    total_params = params_dict["total_params"]
    trainable_params = params_dict["trainable_params"]
    non_trainable_params = params_dict["non_trainable_params"]
    param_size_mb = round(
        sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024), 2
    )

    # 2. FLOPs and MACs
    flops_g = estimate_model_flops(model, input_size=input_size, device=dev)
    macs_g = round(flops_g / 2.0, 3) if flops_g is not None else None

    # 3. Latency Profiling (Batch Size 1)
    dummy_bs1 = torch.randn(*input_size, device=dev)
    bs1_latencies: List[float] = []

    # Reset peak memory stats if CUDA
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_iters):
            _ = model(dummy_bs1)
            synchronize_device(dev)

        # Timed iterations
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            _ = model(dummy_bs1)
            synchronize_device(dev)
            t1 = time.perf_counter()
            bs1_latencies.append((t1 - t0) * 1000.0)  # ms

    lat_p50 = round(float(np.percentile(bs1_latencies, 50)), 2)
    lat_p95 = round(float(np.percentile(bs1_latencies, 95)), 2)
    lat_p99 = round(float(np.percentile(bs1_latencies, 99)), 2)
    lat_bs1_mean = round(float(np.mean(bs1_latencies)), 2)

    # 4. Latency Profiling (Batch Size 32 or Available Max)
    lat_bs_alt_mean = None
    throughput = round(1000.0 / lat_bs1_mean, 2) if lat_bs1_mean > 0 else 0.0

    try:
        dummy_alt = torch.randn(batch_size_alt, input_size[1], input_size[2], input_size[3], device=dev)
        alt_latencies: List[float] = []
        with torch.no_grad():
            for _ in range(max(2, warmup_iters // 2)):
                _ = model(dummy_alt)
                synchronize_device(dev)
            for _ in range(max(5, measure_iters // 2)):
                t0 = time.perf_counter()
                _ = model(dummy_alt)
                synchronize_device(dev)
                t1 = time.perf_counter()
                alt_latencies.append((t1 - t0) * 1000.0)
        lat_bs_alt_mean = round(float(np.mean(alt_latencies)), 2)
        if lat_bs_alt_mean > 0:
            throughput = round((batch_size_alt * 1000.0) / lat_bs_alt_mean, 2)
    except Exception:
        pass

    # 5. Peak VRAM / Memory
    peak_vram_mb = 0.0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_size_mb": param_size_mb,
        "flops_g": flops_g,
        "macs_g": macs_g,
        "batch_size_1_latency_ms": lat_bs1_mean,
        "batch_size_32_latency_ms": lat_bs_alt_mean,
        "latency_p50_ms": lat_p50,
        "latency_p95_ms": lat_p95,
        "latency_p99_ms": lat_p99,
        "throughput_img_per_sec": throughput,
        "peak_vram_mb": peak_vram_mb,
        "device": str(dev),
        "warmup_iterations": warmup_iters,
        "measurement_iterations": measure_iters,
        "input_resolution": list(input_size),
    }
