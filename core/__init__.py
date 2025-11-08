"""
Core package for Dog Breed Classifier V3.5

This package provides core functionality including model management,
data processing, and prediction capabilities.
"""

from .model import (
    load_pretrained_model,
    load_model_and_classes,
    create_custom_model,
    create_custom_model_bn_only,
    save_model_and_classes,
    get_model_summary
)

from .data_processing import (
    preprocess_image,
    create_data_generators,
    get_class_names_from_generator,
    preprocess_batch_images,
    calculate_dataset_statistics
)

from .prediction import (
    process_prediction_results,
    perform_prediction,
    calculate_entropy,
    is_mixed_breed_by_threshold,
    detect_multi_breed_mix,
    analyze_mixed_breed,
    process_confidence_level,
    display_prediction_results,
    predict_breed
)

__all__ = [
    # Model management
    'load_pretrained_model',
    'load_model_and_classes',
    'create_custom_model',
    'create_custom_model_bn_only',
    'save_model_and_classes',
    'get_model_summary',
    
    # Data processing
    'preprocess_image',
    'create_data_generators',
    'get_class_names_from_generator',
    'preprocess_batch_images',
    'calculate_dataset_statistics',
    
    # Prediction
    'process_prediction_results',
    'perform_prediction',
    'calculate_entropy',
    'is_mixed_breed_by_threshold',
    'detect_multi_breed_mix',
    'analyze_mixed_breed',
    'process_confidence_level',
    'display_prediction_results',
    'predict_breed'
]
