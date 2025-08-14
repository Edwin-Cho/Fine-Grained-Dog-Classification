"""
File utilities for Dog Breed Classifier V3.5

This module provides file validation and management utilities,
extracted from the original V2_3_5 single file.
"""

import os
from typing import Optional

from ..config import get_logger, Config

logger = get_logger(__name__)


def validate_file_exists(file_path: str, file_description: str = "file") -> bool:
    """
    [REFACTOR] Utility function to validate file existence
    - Before: Individual file existence validation in each function
    - Improved: Centralized file validation logic to eliminate code duplication
    
    Args:
        file_path (str): File path to validate
        file_description (str): File description (for log messages)
        
    Returns:
        bool: True if file exists, False otherwise
        
    Example:
        >>> validate_file_exists("/path/to/image.jpg", "image file")
        True
    """
    if not file_path:
        logger.error(f"{file_description} path was not provided.")
        return False
        
    if not os.path.exists(file_path):
        logger.error(f"{file_description} not found: {file_path}")
        return False
        
    if not os.path.isfile(file_path):
        logger.error(f"Specified path is not a file: {file_path}")
        return False
        
    logger.debug(f"{file_description} validation completed: {file_path}")
    return True


def validate_directory_exists(dir_path: str, dir_description: str = "directory") -> bool:
    """
    Validates directory existence.
    
    Args:
        dir_path (str): Directory path to validate
        dir_description (str): Directory description (for log messages)
        
    Returns:
        bool: True if directory exists, False otherwise
    """
    if not dir_path:
        logger.error(f"{dir_description} path was not provided.")
        return False
        
    if not os.path.exists(dir_path):
        logger.error(f"{dir_description} not found: {dir_path}")
        return False
        
    if not os.path.isdir(dir_path):
        logger.error(f"Specified path is not a directory: {dir_path}")
        return False
        
    logger.debug(f"{dir_description} validation completed: {dir_path}")
    return True


def is_supported_image_file(file_path: str) -> bool:
    """
    Checks if the file is a supported image format.
    
    Args:
        file_path (str): File path to check
        
    Returns:
        bool: True if supported image file, False otherwise
    """
    if not file_path:
        return False
        
    file_ext = os.path.splitext(file_path)[1].lower()
    is_supported = file_ext in Config.SUPPORTED_IMAGE_EXTENSIONS
    
    if not is_supported:
        logger.warning(f"Unsupported image format: {file_ext}")
        logger.info(f"Supported formats: {', '.join(Config.SUPPORTED_IMAGE_EXTENSIONS)}")
    
    return is_supported


def ensure_directory_exists(dir_path: str) -> bool:
    """
    Creates directory if it doesn't exist.
    
    Args:
        dir_path (str): Directory path to create
        
    Returns:
        bool: True if successful, False if failed
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory creation/verification completed: {dir_path}")
        return True
    except Exception as e:
        logger.error(f"Directory creation failed ({dir_path}): {e}")
        return False


def get_file_size_mb(file_path: str) -> Optional[float]:
    """
    Returns file size in MB.
    
    Args:
        file_path (str): File path
        
    Returns:
        Optional[float]: File size in MB or None if error occurs
    """
    try:
        if not os.path.exists(file_path):
            return None
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except Exception as e:
        logger.error(f"File size check failed ({file_path}): {e}")
        return None
