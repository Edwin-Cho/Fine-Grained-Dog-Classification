"""
System utilities for Dog Breed Classifier V3.5

This module provides system-level utilities including GPU setup and font configuration,
extracted from the original V2_3_5 single file.
"""

import os
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional

from ..config import get_logger

logger = get_logger(__name__)


def setup_gpu_memory() -> None:
    """
    [REFACTOR] GPU memory dynamic growth setup function
    - Before: GPU setup in global scope
    - Improved: Separated into standalone function for better reusability
    
    Features:
    1. GPU device detection and memory growth setup
    2. Automatic fallback to CPU mode when no GPU available
    3. Output setup results to console
    
    Note:
        - Automatically switches to CPU mode when no GPU is available
        - Displays error messages when runtime errors occur
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory dynamic growth setup completed: {len(gpus)} GPU(s) detected")
            except RuntimeError as e:
                logger.error(f"GPU setup error: {e}")
        else:
            logger.info("No GPU found. Running in CPU mode.")
    except Exception as e:
        logger.error(f"Exception occurred during GPU memory setup: {e}")


def setup_matplotlib_font() -> Optional[bool]:
    """
    [REFACTOR] Matplotlib Korean font automatic setup function
    - Before: Font setup in global scope
    - Improved: Separated into standalone function for better reusability
    
    Features:
    1. Automatic Korean font detection in macOS environment
    2. Priority-based available font setup
    3. Fallback to default font when font setup fails
    4. Minus sign corruption prevention setup
    
    Returns:
        bool: Font setup success status
        - Automatically falls back to default font when font setup fails
    """
    try:
        # macOS system font paths
        font_paths = [
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleGothic.ttf'
        ]
        
        font_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_prop = fm.FontProperties(fname=font_path)
                    plt.rcParams['font.family'] = font_prop.get_name()
                    logger.info(f"Korean font setup completed: {font_prop.get_name()}")
                    font_found = True
                    break
                except Exception as font_error:
                    logger.warning(f"Font loading failed ({font_path}): {font_error}")
                    continue
        
        if not font_found:
            # Search for available Korean fonts in system
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            korean_fonts = ['AppleSDGothicNeo-Regular', 'AppleGothic', 'Malgun Gothic']
            
            for korean_font in korean_fonts:
                if korean_font in available_fonts:
                    plt.rcParams['font.family'] = korean_font
                    logger.info(f"Using system font: {korean_font}")
                    font_found = True
                    break
            
            if not font_found:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                logger.warning("Korean font not found, using default font (DejaVu Sans).")
        
        # Prevent minus sign corruption
        plt.rcParams['axes.unicode_minus'] = False
        
        return font_found
    
    except Exception as e:
        logger.error(f"Error occurred during font setup: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        logger.info("Fallback to default font settings.")
        return False


def suppress_warnings() -> None:
    """
    Filters unnecessary warning messages.
    """
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    logger.debug("Warning message filtering setup completed")


def initialize_system() -> None:
    """
    Performs system initialization.
    
    System initialization function:
    - GPU memory setup
    - Matplotlib font setup
    - Warning message filtering
    """
    logger.info("Starting system initialization...")
    
    suppress_warnings()
    setup_gpu_memory()
    setup_matplotlib_font()
    
    logger.info("System initialization completed")
