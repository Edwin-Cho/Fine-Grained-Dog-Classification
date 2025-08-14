# Dog Breed Classifier V3.5 - Modularized Version

[한국어 버전](README.ko.md) | **English Version**

A comprehensive dog breed classification system with advanced features including mixed breed detection, GradCAM visualization, and modular architecture.

## 🚀 Features

### Core Functionality
- **ResNet50-based Transfer Learning**: Pre-trained model fine-tuned for dog breed classification
- **Mixed Breed Detection**: Advanced algorithms to detect and analyze mixed breeds
- **Confidence Analysis**: Automatic confidence level assessment with recommendations
- **Batch Processing**: Support for multiple image predictions

### Advanced Analysis
- **GradCAM Visualization**: Model interpretability through gradient-weighted class activation mapping
- **Confusion Matrix**: Comprehensive performance evaluation
- **Misclassified Sample Analysis**: Detailed analysis of prediction errors
- **Model Complexity Analysis**: Layer-wise parameter and computation analysis

### System Features
- **Modular Architecture**: Clean separation of concerns across multiple modules
- **Comprehensive Logging**: Structured logging with file and console output
- **Error Handling**: Robust error handling with detailed error messages
- **Type Safety**: Complete type hints for better code reliability

## 📁 Project Structure

```
dog_breed_classifier_v3_5/
├── 📁 config/                    # Configuration management
│   ├── __init__.py
│   ├── settings.py               # Config class and constants
│   └── logging_config.py         # Logging setup
│
├── 📁 core/                      # Core functionality
│   ├── __init__.py
│   ├── model.py                  # Model loading and management
│   ├── prediction.py             # Prediction logic
│   └── data_processing.py        # Data preprocessing
│
├── 📁 analysis/                  # Evaluation and visualization
│   ├── __init__.py
│   ├── evaluation.py             # Model evaluation
│   └── visualization.py          # Advanced visualizations
│
├── 📁 utils/                     # Utility functions
│   ├── __init__.py
│   ├── system_utils.py           # System setup (GPU, fonts)
│   ├── file_utils.py             # File validation
│   └── plot_utils.py             # Safe plotting utilities
│
├── 📁 cli/                       # Command-line interface
│   ├── __init__.py
│   └── interface.py              # CLI implementation
│
├── main.py                       # Main entry point
├── requirements.txt              # Dependencies
├── README.md                     # This file (English)
└── README.ko.md                  # Korean version
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies
```bash
cd dog_breed_classifier_v3_5
pip install -r requirements.txt
```

### Optional: GPU Support
For GPU acceleration, install TensorFlow with GPU support:
```bash
pip install tensorflow-gpu>=2.8.0
```

## 🎯 Usage

### Command Line Interface

#### Interactive Mode (Recommended)
```bash
python main.py
```
The system will prompt you to enter an image path.

#### Direct Image Prediction
```bash
python main.py /path/to/your/dog_image.jpg
```

#### Training Mode
```bash
python main.py --train
```

#### Advanced Options
```bash
python main.py --help                    # Show all options
python main.py --verbose image.jpg       # Verbose logging
python main.py --log-level DEBUG         # Set log level
```

### Python API Usage

```python
from dog_breed_classifier_v3_5 import predict_breed, Config

# Simple prediction
success = predict_breed("/path/to/image.jpg")

# Advanced usage
from dog_breed_classifier_v3_5.core import load_model_and_classes, perform_prediction
from dog_breed_classifier_v3_5.analysis import visualize_gradcam

# Load model
model, class_names = load_model_and_classes()

# Perform prediction
results = perform_prediction(model, "/path/to/image.jpg", class_names)

# Visualize with GradCAM
visualize_gradcam(model, "/path/to/image.jpg", class_idx=0)
```

## 📊 Model Information

### Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Custom Layers**: Global Average Pooling + Dense layers with Dropout
- **Input Size**: 224×224×3 RGB images
- **Output**: Softmax probabilities for dog breeds

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Rotation, shifting, zoom, horizontal flip
- **Early Stopping**: Patience of 10 epochs
- **Learning Rate Reduction**: Factor 0.1 with patience 3

### Performance Features
- **Mixed Breed Detection**: Multiple algorithms including entropy analysis
- **Confidence Thresholds**: High (70%), Medium (40%), Low (<40%)
- **Automatic Recommendations**: Suggests re-photographing for low confidence

## 🔧 Configuration

### Settings Customization
Edit `config/settings.py` to customize:

```python
class Config:
    # Image processing
    IMAGE_SIZE = (224, 224)
    
    # Confidence thresholds
    HIGH_CONFIDENCE = 0.7
    MEDIUM_CONFIDENCE = 0.4
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    # File paths
    DATASET_PATH = "/path/to/your/dataset"
    MODEL_SAVE_PATH = "/path/to/models/"
```

### Logging Configuration
Customize logging in `config/logging_config.py`:
- Log levels: DEBUG, INFO, WARNING, ERROR
- Output: Console and file logging
- Format: Timestamp, level, function name, message

## 📈 Advanced Features

### Mixed Breed Analysis
The system uses multiple detection methods:
1. **Threshold-based**: Analyzes probability differences
2. **Entropy-based**: Measures prediction uncertainty
3. **Multi-breed detection**: Identifies complex mixes

### Model Interpretability
- **GradCAM**: Visualizes important image regions
- **Feature Maps**: Shows learned representations
- **Prediction Distribution**: Probability analysis

### Performance Monitoring
- **Confusion Matrix**: Classification performance
- **Per-class Metrics**: Precision, recall, F1-score
- **Misclassified Analysis**: Error pattern identification

## 🐛 Troubleshooting

### Common Issues

#### GPU Memory Issues
```python
# Automatic GPU memory growth is enabled by default
# If issues persist, try reducing batch size in config/settings.py
Config.BATCH_SIZE = 16
```

#### Font Issues (Korean text)
```python
# The system automatically detects and sets Korean fonts
# For manual setup, edit utils/system_utils.py
```

#### Model Loading Errors
```bash
# Ensure model files exist:
ls /path/to/models/dog_breed_classifier_custom_stanford_v2.h5
ls /path/to/models/class_names.npy

# If missing, run training mode:
python main.py --train
```

### Debug Mode
```bash
python main.py --log-level DEBUG --verbose image.jpg
```

## 📝 Development

### Code Quality
- **Type Hints**: Complete type annotations
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Robust exception management
- **Logging**: Structured logging throughout

### Testing
```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests (when implemented)
pytest tests/
```

### Code Formatting
```bash
# Install formatting tools
pip install black flake8

# Format code
black dog_breed_classifier_v3_5/

# Check style
flake8 dog_breed_classifier_v3_5/
```

## 🔄 Migration from V2_3_5

This modular version maintains API compatibility with the original single-file V2_3_5:

```python
# Old way (V2_3_5 single file)
from V2_3_5_AI_Practical_Refactor import predict_breed

# New way (V3_5 modular)
from dog_breed_classifier_v3_5 import predict_breed
```

## 📄 License

This project is part of the AI Performance Metrics research initiative.

## 🤝 Contributing

1. Follow the modular architecture principles
2. Add comprehensive type hints and docstrings
3. Include appropriate error handling and logging
4. Update tests and documentation

---

**Dog Breed Classifier V3.5** - Advanced, Modular, and Maintainable 🐕
