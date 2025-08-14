"""
Configuration settings for Dog Breed Classifier V3.5

This module contains all configuration constants and settings used throughout
the application, centralized from the original V2_3_5 single file.
"""

import os
from typing import List


class Config:
    """
    [REFACTOR] Configuration management class - centralized hardcoded constants
    - Before: Settings scattered as global variables
    - Improved: Class-based centralized configuration management
    """
    
    # Image and model settings
    IMAGE_SIZE = (224, 224)                    # Model input image size
    
    # Prediction-related thresholds
    HIGH_CONFIDENCE = 0.7                      # High confidence threshold (70%)
    MEDIUM_CONFIDENCE = 0.4                    # Medium confidence threshold (40%)
    MIX_BREED_THRESHOLD = 0.25                 # Mixed breed detection threshold (probability difference between top 2 breeds)
    LOW_CONFIDENCE_THRESHOLD = 0.4             # Low confidence threshold (re-photographing recommended)
    
    # Visualization settings
    TOP_K_PREDICTIONS = 5                      # Display top K prediction results
    FIGURE_SIZE = (12, 8)                      # Default graph size
    DPI = 300                                  # Image save resolution
    
    # File path settings
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.getenv('DATASET_PATH', './dataset/stanford_dogs_dataset')
    MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', './models/')
    
    # Model filenames
    CUSTOM_MODEL_FILENAME = "dog_breed_classifier_custom_stanford_v2.h5"
    CLASS_NAMES_FILENAME = "class_names.npy"
    REFERENCE_EMBEDDINGS_FILENAME = "reference_embeddings.npy"
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    PATIENCE = 10                              # EarlyStopping patience
    
    # Data augmentation settings
    ROTATION_RANGE = 20
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    HORIZONTAL_FLIP = True
    ZOOM_RANGE = 0.2
    SHEAR_RANGE = 0.2
    FILL_MODE = 'nearest'
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "dog_breed_classifier.log"
    
    # Supported image extensions
    SUPPORTED_IMAGE_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
    ]
    
    @classmethod
    def get_custom_model_path(cls) -> str:
        """Return custom model file path"""
        return os.path.join(cls.MODEL_SAVE_PATH, cls.CUSTOM_MODEL_FILENAME)
    
    @classmethod
    def get_class_names_path(cls) -> str:
        """Return class names file path"""
        return os.path.join(cls.MODEL_SAVE_PATH, cls.CLASS_NAMES_FILENAME)
    
    @classmethod
    def get_reference_embeddings_path(cls) -> str:
        """Return reference embeddings file path"""
        return os.path.join(cls.MODEL_SAVE_PATH, cls.REFERENCE_EMBEDDINGS_FILENAME)


# Global constants (maintained for backward compatibility)
DATASET_PATH = Config.DATASET_PATH
CUSTOM_MODEL_PATH = Config.get_custom_model_path()
CLASS_NAMES_PATH = Config.get_class_names_path()
REFERENCE_EMBEDDINGS_PATH = Config.get_reference_embeddings_path()
