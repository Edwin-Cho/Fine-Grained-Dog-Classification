"""
Configuration package for Dog Breed Classifier V3.5

This package provides centralized configuration management and logging setup.
"""

from .settings import Config, DATASET_PATH, CUSTOM_MODEL_PATH, CLASS_NAMES_PATH, REFERENCE_EMBEDDINGS_PATH
from .logging_config import setup_logging, get_logger

__all__ = [
    'Config',
    'DATASET_PATH',
    'CUSTOM_MODEL_PATH', 
    'CLASS_NAMES_PATH',
    'REFERENCE_EMBEDDINGS_PATH',
    'setup_logging',
    'get_logger'
]
