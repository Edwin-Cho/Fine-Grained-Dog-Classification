"""
CLI package for Dog Breed Classifier V3.5

This package provides command-line interface functionality.
"""

from .interface import (
    parse_arguments,
    get_image_path_interactive,
    validate_model_files,
    run_training_mode,
    run_prediction_mode,
    main,
    run_cli
)

__all__ = [
    'parse_arguments',
    'get_image_path_interactive',
    'validate_model_files',
    'run_training_mode',
    'run_prediction_mode',
    'main',
    'run_cli'
]
