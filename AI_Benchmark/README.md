# AI Benchmark Results

> **English** | [한국어](README.ko.md)

This directory contains visualization results and performance metrics for the BN-Only fine-tuning model.

## 📂 Directory Structure

```
AI_Benchmark/
├── metrics/                      # Performance metrics and visualizations
│   ├── f1_score_distribution.png         ✅ Paper: Class-wise F1 scores
│   ├── normalized_confusion_matrix.png   ✅ Paper: Top-25 confusion matrix
│   ├── result_2.png                      ✅ Prediction example (Jindo)
│   ├── Custom_CNN_Matrix.png             ⚠️  Legacy
│   ├── confusion_matrix.png              ⚠️  Legacy
│   ├── result_1.jpeg                     ⚠️  Legacy
│   ├── result_3.png                      ⚠️  Legacy
│   └── result_terminal.png               ⚠️  Legacy
│
└── model_visualizations/         # Model architecture diagrams
    ├── custom_architecture_diagram.png   ✅ Paper: BN-Only architecture
    └── parameter_distribution.png        ✅ Paper: Parameter comparison

✅ = Recommended for paper
⚠️  = Legacy files (can be removed)
```

---

## 📊 Metrics

### Performance Visualizations

#### `f1_score_distribution.png` ✅ **Recommended**
- **Description**: Distribution of F1 scores across all 122 dog breeds
- **Key Insight**: Most breeds achieve F1 > 0.92, showing balanced performance
- **Paper Usage**: Section 3 (Results) - Class-wise performance analysis
- **Size**: 90 KB

#### `normalized_confusion_matrix.png` ✅ **Recommended**
- **Description**: Normalized confusion matrix for top-25 classes
- **Key Insight**: Strong diagonal, minimal off-diagonal confusion
- **Paper Usage**: Section 3 (Results) - Detailed performance analysis
- **Size**: 856 KB

#### `result_2.png` ✅ **Optional**
- **Description**: Prediction example for Korean Jindo dog
- **Key Insight**: Qualitative demonstration of model capability
- **Paper Usage**: Appendix or Discussion section
- **Size**: 197 KB

---

## 🏗️ Model Visualizations

### Architecture Diagrams

#### `custom_architecture_diagram.png` ✅ **Highly Recommended**
- **Description**: BN-Only fine-tuning architecture diagram
- **Content**:
  - ResNet50 base model structure
  - Frozen Conv layers (❄️)
  - Trainable BN layers (🔥)
  - Custom classification head
- **Paper Usage**: Section 2 (Methodology) - Main architecture figure
- **Size**: 430 KB

#### `parameter_distribution.png` ✅ **Recommended**
- **Description**: Parameter distribution comparison
- **Content**:
  - BN-Only: 1.2M (4.7%)
  - Full Fine-tuning: 24.7M (99.8%)
  - Visual comparison of trainable vs frozen parameters
- **Paper Usage**: Section 3 (Results) - Parameter efficiency
- **Size**: 360 KB

---

## 📝 Usage in Paper

### Main Figures

**Figure 1**: BN-Only Architecture
- File: `model_visualizations/custom_architecture_diagram.png`
- Section: 2. Methodology
- Caption: "Proposed BN-Only fine-tuning architecture. Only BatchNormalization layers (red) are trainable while Convolutional layers (blue) remain frozen."

**Figure 2**: Parameter Efficiency
- File: `model_visualizations/parameter_distribution.png`
- Section: 3. Experimental Results
- Caption: "Parameter distribution comparison showing 95.3% reduction in trainable parameters."

### Supporting Figures

**Figure 3**: Class-wise Performance
- File: `metrics/f1_score_distribution.png`
- Section: 3. Experimental Results
- Caption: "F1 score distribution across 122 dog breeds, demonstrating consistent performance."

**Figure 4**: Confusion Matrix
- File: `metrics/normalized_confusion_matrix.png`
- Section: 3. Experimental Results
- Caption: "Normalized confusion matrix for top-25 most frequent classes."

---

## 🗑️ Legacy Files

The following files are from previous experiments and can be safely removed:

- `Custom_CNN_Matrix.png` - Old custom CNN confusion matrix
- `confusion_matrix.png` - Duplicate/older version
- `result_1.jpeg` - Old prediction example
- `result_3.png` - Old prediction example
- `result_terminal.png` - Terminal output screenshot

**Cleanup Command**:
```bash
cd AI_Benchmark/metrics
rm Custom_CNN_Matrix.png confusion_matrix.png result_1.jpeg result_3.png result_terminal.png
```

---

## 📊 Performance Summary

**Model**: BN-Only Fine-tuning (ResNet50)
- **Validation Accuracy**: 72.72%
- **Number of Classes**: 122 (Stanford Dogs + 2 Korean breeds)
- **Trainable Parameters**: 1.2M (4.7%)
- **Total Parameters**: 24.7M

**Key Metrics**:
- **Average F1 Score**: ~0.94
- **Top-1 Accuracy**: 72.72%
- **Parameter Reduction**: 95.3% vs Full Fine-tuning
- **Overfitting Prevention**: Train-Val gap -3.8%

---

## 🎨 Figure Quality

All figures are high-resolution (300 DPI) and publication-ready:
- PNG format for lossless quality
- Clear labels and legends
- Consistent color scheme
- Professional appearance

---

## 📖 Related Documentation

- [Ablation Study Scripts](../scripts/README.md)
- [Model Architecture](../docs/Model_Layer.md)
- [Dataset Description](../docs/Dataset.md)
- [Paper Structure](../docs/CNN_Optimization_Paper_Structure.md)

---

**Generated**: 2025.11.08  
**Author**: Edwin R. Cho  
**Model**: BN-Only Fine-tuning (ResNet50)
