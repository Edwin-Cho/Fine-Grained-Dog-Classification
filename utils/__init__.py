"""
Utilities package for Dog Breed Classifier V3.5

This package provides various utility functions including system setup,
file validation, and plot management.
"""

from .system_utils import setup_gpu_memory, setup_matplotlib_font, suppress_warnings, initialize_system
from .file_utils import validate_file_exists, validate_directory_exists, is_supported_image_file, ensure_directory_exists, get_file_size_mb
from .plot_utils import safe_plot_display, create_figure_with_size, setup_plot_style, close_all_plots, get_color_palette

__all__ = [
    # System utilities
    'setup_gpu_memory',
    'setup_matplotlib_font', 
    'suppress_warnings',
    'initialize_system',
    
    # File utilities
    'validate_file_exists',
    'validate_directory_exists',
    'is_supported_image_file',
    'ensure_directory_exists',
    'get_file_size_mb',
    
    # Plot utilities
    'safe_plot_display',
    'create_figure_with_size',
    'setup_plot_style',
    'close_all_plots',
    'get_color_palette'
]
