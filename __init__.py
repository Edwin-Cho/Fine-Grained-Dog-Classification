"""
Dog Breed Classifier V3.5 - Modularized Version

A comprehensive dog breed classification system with advanced features including:
- ResNet50-based transfer learning
- Mixed breed detection and analysis
- GradCAM visualization for model interpretability
- Comprehensive evaluation and reporting
- Modular architecture for maintainability

This package is the modularized version of the original V2_3_5 single-file implementation,
providing better code organization, reusability, and maintainability.

Modules:
    config: Configuration management and logging setup
    core: Core functionality (model, data processing, prediction)
    analysis: Model evaluation and visualization
    utils: Utility functions (file handling, plotting, system setup)
    cli: Command-line interface

Usage:
    python main.py                    # Interactive mode
    python main.py image.jpg          # Direct prediction
    python main.py --train            # Training mode
"""

__version__ = "3.5.0"
__author__ = "Dog Breed Classifier Team"
__description__ = "Advanced Dog Breed Classification System with Mixed Breed Detection"

# Import main components for easy access
from .config import Config, setup_logging, get_logger
from .core import predict_breed, load_model_and_classes, preprocess_image
from .analysis import plot_confusion_matrix, visualize_gradcam
from .utils import validate_file_exists, safe_plot_display, initialize_system
from .cli import run_cli

# Package metadata
__all__ = [
    # Core functionality
    'predict_breed',
    'load_model_and_classes', 
    'preprocess_image',
    
    # Configuration
    'Config',
    'setup_logging',
    'get_logger',
    
    # Analysis
    'plot_confusion_matrix',
    'visualize_gradcam',
    
    # Utilities
    'validate_file_exists',
    'safe_plot_display',
    'initialize_system',
    
    # CLI
    'run_cli',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]
