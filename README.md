# Empirically Derived Architecture-Selection Framework for CNNs vs Vision Transformers

[![Target Journal: npj Computer Vision](https://img.shields.io/badge/Target%20Journal-npj%20Computer%20Vision-blue.svg)](https://www.nature.com/npjcompumats/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Reproducibility Standard](https://img.shields.io/badge/Protocol-Strict%20Reproducibility-green.svg)](REPRODUCIBILITY.md)

This repository contains the complete experimental framework and decision pipeline for the research study:

> **"Empirically Derived Architecture-Selection Framework for CNNs vs Vision Transformers Under Data, Computational, Robustness, and Deployment Constraints"**

---

## 🎯 Research Objective

The objective of this research is **not** simply to declare whether CNNs or Vision Transformers (ViTs) are generically superior. Instead, it seeks to answer:

> *"Under what combinations of data availability, visual domain, pretraining regime, robustness requirements, computational budgets, and deployment constraints should a practitioner choose one architecture family or specific model over another?"*

The system produces an **empirically learned architecture-selection model** trained on controlled multi-condition benchmarks and externally validated using **Leave-One-Dataset-Out (LODO)** cross-validation.

---

## 🔬 Experimental Factors

1. **Architecture Families & Models**:
   - **CNN**: ResNet-50, DenseNet-121, EfficientNetV2-S, ConvNeXt-Tiny
   - **Vision Transformer (ViT)**: ViT-B/16, Swin-T
2. **Datasets & Visual Domains**:
   - **CIFAR-100**: General objects (low native resolution upsampled to 224×224)
   - **Oxford Flowers-102**: Fine-grained natural flora
   - **Stanford Cars**: Fine-grained vehicles
   - **Food-101**: Fine-grained food dishes
   - **Describable Textures Dataset (DTD)**: Surface texture patterns
   - **EuroSAT**: Remote sensing / satellite multi-spectral RGB
   - **SUN397**: Scene understanding (indoor & outdoor)
3. **Training-Data Availability**:
   - Strictly nested, class-stratified subsets: **5%, 10%, 25%, 50%, 75%, 100%**
4. **Pretraining Regimes**:
   - `from_scratch` (random initialization) vs `imagenet1k_pretrained` (ImageNet-1K weights)
5. **Random Seeds**:
   - `[42, 123, 2024, 3407, 9999]`
6. **Robustness & Corruptions**:
   - 15 image corruptions at 5 severity levels (Gaussian noise, blur, fog, contrast, JPEG compression, etc.)
7. **Hardware & Deployment Profiles**:
   - Latency (batch sizes 1 and 32), throughput (img/sec), peak memory (MB), FLOPs, parameter count.

---

## 📁 Repository Layout

```
CNN_vs_ViT/
├── configs/                  # YAML configurations
│   ├── datasets.yaml         # Dataset registry, domains, paths, licenses
│   ├── models.yaml           # Model definitions and family mappings
│   ├── training.yaml         # Standardized training recipe and augmentations
│   ├── benchmark.yaml        # Experimental grid and orchestrator settings
│   ├── robustness.yaml       # Image corruption parameters and metrics
│   ├── hardware.yaml         # Latency, memory, and FLOPs profiling settings
│   └── experiment.yaml       # Master experiment configuration
│
├── src/                      # Source modules
│   ├── datasets/             # Unified dataset adapters, registry, subsets, validator
│   ├── models/               # Model factory and architecture registries
│   ├── training/             # Trainer, checkpointing, and optimization loops
│   ├── evaluation/           # Macro-F1, Accuracy, ECE, Brier, calibration
│   ├── profiling/            # FLOPs, latency, and memory profiling
│   ├── robustness/           # Corruption engine and robustness evaluation
│   ├── statistics/           # Bootstrapping, ANOVA, mixed-effects models
│   ├── selection/            # Pareto frontiers, utility functions, LODO validation
│   └── utils/                # Seeding, logging, system profiling, config loaders
│
├── scripts/                  # Executable CLI workflows
│   ├── validate_datasets.py  # Validate local dataset integrity & zero-leakage
│   ├── generate_manifests.py # Pre-generate deterministic subset manifests
│   ├── dataset_stats.py      # Output Table 1 dataset characteristics
│   ├── train.py              # Single model training entrypoint
│   ├── run_benchmark.py      # Benchmark orchestrator
│   ├── run_robustness.py     # Robustness evaluation pipeline
│   ├── profile_models.py     # Hardware and computational profiler
│   ├── run_statistics.py     # Statistical hypothesis testing & interaction models
│   ├── train_selector.py     # Empirical architecture selector training
│   ├── run_lodo.py           # Leave-one-dataset-out cross-validation
│   └── generate_figures.py   # Publication-quality figures (14 figures)
│
├── tests/                    # Comprehensive unit tests
├── manifests/                # Saved subset CSV manifests
├── checkpoints/              # Model weights and training checkpoints
├── results/                  # Structured Parquet, JSON, and CSV metrics
├── figures/                  # Publication-ready figures
├── tables/                   # Publication-ready tables
│
├── README.md                 # Project overview and quickstart
├── DATA_PREPARATION.md       # Manual dataset download & layout guide
├── EXPERIMENT_PROTOCOL.md    # Detailed scientific protocol
├── REPRODUCIBILITY.md        # Determinism and reproducibility guarantees
├── ENVIRONMENT.md            # Hardware & software environment specification
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project packaging specification
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository and install dependencies
pip install -r requirements.txt
```

### 2. Validate Datasets & Generate Subset Manifests

```bash
# Verify local dataset directories and split integrity
python scripts/validate_datasets.py

# Generate Table 1: Dataset Characteristics
python scripts/dataset_stats.py

# Pre-generate deterministic nested subset manifests
python scripts/generate_manifests.py
```

### 3. Run Unit Tests

```bash
python -m unittest discover tests
```

---

## 📖 Documentation Links

- [DATA_PREPARATION.md](DATA_PREPARATION.md): Guide on downloading and placing datasets.
- [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md): Experimental factors, hypotheses, and analysis plan.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): Deterministic seeding, subset containment proofs, and manifest structures.
- [ENVIRONMENT.md](ENVIRONMENT.md): Computing environment, software dependencies, and hardware profiling.
# CNN-or-ViT
