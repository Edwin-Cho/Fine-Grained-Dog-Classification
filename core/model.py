"""
Model management for Dog Breed Classifier V3.5

This module provides model loading and management functionality,
extracted from the original V2_3_5 single file.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from typing import Tuple, Optional

from ..config import get_logger, Config
from ..utils import validate_file_exists

logger = get_logger(__name__)


def load_pretrained_model() -> tf.keras.Model:
    """
    Load ResNet50 model pre-trained on ImageNet.
    
    Returns:
        tf.keras.Model: Pre-trained ResNet50 model
    """
    try:
        model = ResNet50(weights='imagenet')
        logger.info("Pre-trained ResNet50 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load pre-trained model: {e}")
        raise


def load_model_and_classes() -> Tuple[Optional[tf.keras.Model], Optional[np.ndarray]]:
    """
    [REFACTOR] Safely load custom model and class names
    - Before: Individual file existence validation
    - Improved: Utility function usage and systematic error handling
    
    Returns:
        Tuple[Optional[tf.keras.Model], Optional[np.ndarray]]: 
            (model, class names array) or (None, None) if failed
            
    Raises:
        FileNotFoundError: When model or class names file is missing
        Exception: When error occurs during model loading
    """
    try:
        custom_model_path = Config.get_custom_model_path()
        class_names_path = Config.get_class_names_path()
        
        # File existence validation
        if not validate_file_exists(custom_model_path, "model file"):
            return None, None
            
        if not validate_file_exists(class_names_path, "class names file"):
            return None, None
        
        # Model loading
        logger.info(f"Loading model: {custom_model_path}")
        model = load_model(custom_model_path)
        
        # Class names loading
        class_names = np.load(class_names_path, allow_pickle=True)
        
        logger.info(f"Model loading completed: {len(class_names)} classes")
        return model, class_names
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        return None, None


def create_custom_model(num_classes: int, input_shape: Tuple[int, int, int] = None) -> tf.keras.Model:
    """
    Create custom dog breed classification model.
    
    Args:
        num_classes (int): Number of classes to classify
        input_shape (Tuple[int, int, int], optional): Input image shape
        
    Returns:
        tf.keras.Model: Created model
    """
    if input_shape is None:
        input_shape = (*Config.IMAGE_SIZE, 3)
    
    try:
        # Base ResNet50 model (for transfer learning)
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Set some layers of base model to be trainable
        base_model.trainable = True
        fine_tune_at = 100
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=Config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Custom model creation completed: {num_classes} classes")
        return model
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        raise


def save_model_and_classes(model: tf.keras.Model, class_names: np.ndarray, 
                          model_path: str = None, class_names_path: str = None) -> bool:
    """
    Save model and class names.
    
    Args:
        model (tf.keras.Model): Model to save
        class_names (np.ndarray): Class names array
        model_path (str, optional): Model save path
        class_names_path (str, optional): Class names save path
        
    Returns:
        bool: True if save successful, False if failed
    """
    if model_path is None:
        model_path = Config.get_custom_model_path()
    if class_names_path is None:
        class_names_path = Config.get_class_names_path()
    
    try:
        # Create save directory
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model.save(model_path)
        logger.info(f"Model saved successfully: {model_path}")
        
        # Save class names
        np.save(class_names_path, class_names)
        logger.info(f"Class names saved successfully: {class_names_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model save failed: {e}")
        return False


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Return model summary information as string.
    
    Args:
        model (tf.keras.Model): Model to summarize
        
    Returns:
        str: Model summary information
    """
    try:
        import io
        import sys
        
        # Capture stdout to convert summary to string
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        model.summary()
        
        sys.stdout = old_stdout
        summary_str = buffer.getvalue()
        
        return summary_str
        
    except Exception as e:
        logger.error(f"Model summary generation failed: {e}")
        return f"Model summary generation failed: {e}"
