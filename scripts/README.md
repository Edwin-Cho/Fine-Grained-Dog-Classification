# Ablation Study Scripts

> **English** | [한국어](README.ko.md)

This directory contains training and comparison scripts for the ablation study comparing BN-Only fine-tuning vs Full Fine-tuning strategies.

## 📂 Files

### Training Scripts

#### `train_simple.py` - BN-Only Fine-tuning (Proposed Method)
- **Strategy**: Only BatchNormalization layers trainable
- **Parameters**: 1.2M (4.7%)
- **Training Time**: ~2.7h
- **Results**: 72.72% validation accuracy

```bash
cd scripts
python train_simple.py
```

#### `train_full_finetuning.py` - Full Fine-tuning (Baseline)
- **Strategy**: All layers trainable
- **Parameters**: 24.7M (99.8%)
- **Training Time**: ~4.0h
- **Results**: 73.19% validation accuracy

```bash
cd scripts
python train_full_finetuning.py
```

### Analysis Scripts

#### `compare_bn_vs_full.py` - Results Comparison
Generates publication-ready comparison figures:
- 4-panel comparison (accuracy, parameters, overfitting, efficiency)
- Training-validation gap analysis
- Summary statistics table

```bash
cd scripts
python compare_bn_vs_full.py
```

## 📊 Output

All scripts save results to `../ablation_results/`:

```
ablation_results/
├── bn_vs_full_comparison.png       # Main comparison figure
├── train_val_comparison.png        # Overfitting analysis
├── bn_only/
│   ├── bn_only_best.h5            # Trained model
│   ├── training_history.png       # Learning curves
│   └── results.npy                # Experiment data
└── full_finetuning/
    ├── best_model.h5              # Trained model
    ├── training_history.png       # Learning curves
    └── results.npy                # Experiment data
```

## 🔬 Key Findings

| Metric | BN-Only | Full FT | Improvement |
|--------|---------|---------|-------------|
| **Trainable Params** | 1.2M | 24.7M | **-95.3%** |
| **Val Accuracy** | 72.72% | 73.19% | -0.47%p |
| **Train-Val Gap** | -3.8% | +22.7% | **-26.5%p** |
| **Efficiency Score** | 62.2 | 3.0 | **+20x** |

### Key Insights

1. **Parameter Optimization** ⭐
   - 95.3% parameter reduction with nearly identical performance
   - 0.47%p accuracy difference is statistically insignificant

2. **Overfitting Prevention** 🔥
   - BN-Only: Train-Val gap -3.8% (validation > training)
   - Full FT: Train-Val gap +22.7% (severe overfitting)
   - Frozen backbone acts as implicit regularizer

3. **Resource Efficiency** 💡
   - 20x efficiency improvement
   - Training possible on consumer laptops (4GB VRAM)
   - 33% training time reduction

## 📝 Notes

- All scripts use the same dataset: `../Dataset_Stanford/Stanford_Images`
- Training uses data augmentation (rotation, shift, zoom, flip)
- Early stopping with patience=10 to prevent overfitting
- Results are saved automatically after training

## 🚀 Quick Start

Run complete ablation study:

```bash
cd scripts

# 1. Train BN-Only (2.7h)
python train_simple.py

# 2. Train Full FT (4.0h)
python train_full_finetuning.py

# 3. Generate comparison figures
python compare_bn_vs_full.py
```

**Total time**: ~7 hours on Apple M4 Pro GPU.

## 💻 System Requirements

### BN-Only Fine-tuning
- **GPU Memory**: ~3GB
- **System RAM**: 8GB recommended
- **Storage**: ~500MB
- **Supported Hardware**: 
  - NVIDIA GPU (CUDA)
  - Apple Silicon (M1/M2/M3/M4)
  - CPU (slow, not recommended)

### Full Fine-tuning
- **GPU Memory**: ~8GB
- **System RAM**: 16GB recommended
- **Storage**: ~1GB
- **Supported Hardware**: 
  - NVIDIA GPU (8GB+)
  - Apple Silicon (16GB+ unified memory)

## 📖 Additional Documentation

- [Ablation Study Guide](../docs/ABLATION_STUDY_GUIDE.md)
- [Model Architecture](../docs/Model_Layer.md)
- [Dataset Description](../docs/Dataset.md)
- [Paper Structure](../docs/CNN_Optimization_Paper_Structure.md)
- [Documentation Index](../docs/README.md)

## 🎯 Experimental Validation

These scripts were executed and validated on November 8, 2025:

- ✅ **BN-Only**: 72.72% validation accuracy (1.2M params)
- ✅ **Full FT**: 73.19% validation accuracy (24.7M params)
- ✅ **Comparison Figures**: Publication-ready high-quality graphs
- ✅ **Reproducibility**: Random seed 42 fixed

## 📧 Contact

For questions or issues, please use GitHub Issues.

---

**Author**: Edwin R. Cho  
**Date**: 2025.11.08  
**License**: MIT
