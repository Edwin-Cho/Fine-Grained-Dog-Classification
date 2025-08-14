"""
Data processing for Dog Breed Classifier V3.5

This module provides data preprocessing and data generator functionality,
extracted from the original V2_3_5 single file.
"""

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from typing import Tuple, Optional

from ..config import get_logger, Config
from ..utils import validate_directory_exists

logger = get_logger(__name__)


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    [REFACTOR] Converts image to model input format.
    - Before: Used global variable IMAGE_SIZE
    - Improved: Uses Config class-based settings
    
    Args:
        image_path (str): Image file path
        
    Returns:
        Optional[np.ndarray]: Preprocessed image array or None on error
    """
    try:
        # Load as PIL image
        img = load_img(image_path, target_size=Config.IMAGE_SIZE)
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Add batch dimension (1, height, width, channels)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply ResNet50 preprocessing
        img_array = preprocess_input(img_array)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error occurred during image processing: {e}")
        return None


def create_data_generators(dataset_path: str = None) -> Tuple[ImageDataGenerator, ImageDataGenerator, int]:
    """
    [REFACTOR] Function to create data generators for training and validation
    - Before: Used global variable DATASET_PATH
    - Improved: Uses Config class-based settings, improved type hinting
    
    Args:
        dataset_path (str, optional): Dataset path. Uses Config.DATASET_PATH if None
        
    Returns:
        Tuple[ImageDataGenerator, ImageDataGenerator, int]: 
            (training generator, validation generator, number of classes)
            
    Raises:
        FileNotFoundError: When specified dataset path does not exist
        ValueError: When train/ or validation/ folder is missing
    """
    if dataset_path is None:
        dataset_path = Config.DATASET_PATH
    
    # Validate dataset path
    if not validate_directory_exists(dataset_path, "dataset directory"):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    train_path = f"{dataset_path}/train"
    validation_path = f"{dataset_path}/validation"
    
    if not validate_directory_exists(train_path, "training data directory"):
        raise ValueError(f"Training data folder not found: {train_path}")
        
    if not validate_directory_exists(validation_path, "validation data directory"):
        raise ValueError(f"Validation data folder not found: {validation_path}")
    
    try:
        # Training data augmentation settings
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=Config.ROTATION_RANGE,
            width_shift_range=Config.WIDTH_SHIFT_RANGE,
            height_shift_range=Config.HEIGHT_SHIFT_RANGE,
            shear_range=Config.SHEAR_RANGE,
            zoom_range=Config.ZOOM_RANGE,
            horizontal_flip=Config.HORIZONTAL_FLIP,
            fill_mode=Config.FILL_MODE
        )
        
        # Validation data (no augmentation)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical'
        )
        
        # Validation data generator
        validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=Config.IMAGE_SIZE,
            batch_size=Config.BATCH_SIZE,
            class_mode='categorical'
        )
        
        num_classes = len(train_generator.class_indices)
        
        logger.info(f"Data generator creation completed")
        logger.info(f"- Training samples: {train_generator.samples}")
        logger.info(f"- Validation samples: {validation_generator.samples}")
        logger.info(f"- Number of classes: {num_classes}")
        logger.info(f"- Batch size: {Config.BATCH_SIZE}")
        
        return train_generator, validation_generator, num_classes
        
    except Exception as e:
        logger.error(f"Data generator creation failed: {e}")
        raise


def get_class_names_from_generator(generator: ImageDataGenerator) -> np.ndarray:
    """
    Extracts class names from data generator.
    
    Args:
        generator (ImageDataGenerator): Data generator
        
    Returns:
        np.ndarray: Class names array
    """
    try:
        # Convert class indices to class names
        class_indices = generator.class_indices
        class_names = list(class_indices.keys())
        class_names_array = np.array(class_names)
        
        logger.info(f"Class name extraction completed: {len(class_names)} classes")
        return class_names_array
        
    except Exception as e:
        logger.error(f"Class name extraction failed: {e}")
        return np.array([])


def preprocess_batch_images(image_paths: list) -> Optional[np.ndarray]:
    """
    Preprocesses multiple images as a batch.
    
    Args:
        image_paths (list): List of image file paths
        
    Returns:
        Optional[np.ndarray]: Preprocessed image batch or None
    """
    try:
        processed_images = []
        
        for image_path in image_paths:
            img_array = preprocess_image(image_path)
            if img_array is not None:
                # Remove batch dimension (added by preprocess_image)
                img_array = np.squeeze(img_array, axis=0)
                processed_images.append(img_array)
            else:
                logger.warning(f"Image processing failed, skipping: {image_path}")
        
        if not processed_images:
            logger.error("No processable images available.")
            return None
        
        # Combine into batch
        batch_array = np.array(processed_images)
        logger.info(f"Batch image preprocessing completed: {batch_array.shape}")
        
        return batch_array
        
    except Exception as e:
        logger.error(f"Batch image preprocessing failed: {e}")
        return None


def calculate_dataset_statistics(generator: ImageDataGenerator) -> dict:
    """
    Calculates dataset statistics.
    
    Args:
        generator (ImageDataGenerator): Data generator
        
    Returns:
        dict: Dataset statistics information
    """
    try:
        stats = {
            'total_samples': generator.samples,
            'num_classes': len(generator.class_indices),
            'batch_size': generator.batch_size,
            'image_size': generator.target_size,
            'class_distribution': {}
        }
        
        # Calculate samples per class
        import os
        for class_name, class_idx in generator.class_indices.items():
            class_path = os.path.join(generator.directory, class_name)
            if os.path.exists(class_path):
                class_samples = len([f for f in os.listdir(class_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                stats['class_distribution'][class_name] = class_samples
        
        logger.info("Dataset statistics calculation completed")
        return stats
        
    except Exception as e:
        logger.error(f"Dataset statistics calculation failed: {e}")
        return {}
