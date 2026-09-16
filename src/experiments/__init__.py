"""Experiments package for CNN vs. ViT Benchmarking Suite."""

def __getattr__(name: str):
    if name == "run_cifar100_suite":
        from src.experiments.run_cifar100 import run_cifar100_suite
        return run_cifar100_suite
    elif name == "run_comparison_suite":
        from src.experiments.run_comparison import run_comparison_suite
        return run_comparison_suite
    elif name == "run_data_scaling_suite":
        from src.experiments.run_data_scaling import run_data_scaling_suite
        return run_data_scaling_suite
    elif name == "execute_full_benchmark":
        from src.experiments.run_full_benchmark import execute_full_benchmark
        return execute_full_benchmark
    elif name == "run_training_experiment":
        from src.experiments.run_single import run_training_experiment
        return run_training_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run_training_experiment",
    "run_cifar100_suite",
    "run_data_scaling_suite",
    "run_comparison_suite",
    "execute_full_benchmark",
]
