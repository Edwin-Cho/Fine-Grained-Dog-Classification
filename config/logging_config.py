"""
Logging configuration for Dog Breed Classifier V3.5

This module provides centralized logging setup functionality,
extracted from the original V2_3_5 single file.
"""

import logging
import os
from typing import Optional

from .settings import Config


def setup_logging(log_level: str = None, log_file: str = None) -> logging.Logger:
    """
    [REFACTOR] Logging system setup function
    - Before: Simple output using print statements
    - Improved: Systematic logging system introduction for enhanced debugging and monitoring
    
    Key features:
    1. Simultaneous logging to console and file
    2. Color-coded log levels (console)
    3. Detailed log format (file)
    4. Automatic log file creation and management
    
    Args:
        log_level (str, optional): Log level (DEBUG, INFO, WARNING, ERROR)
        log_file (str, optional): Log file path
    
    Output format:
    - Console: [time] [level] message
    - File: [time] [level] [function] message
    
    Returns:
        logging.Logger: Configured logger instance
    """
    if log_level is None:
        log_level = Config.LOG_LEVEL
    if log_file is None:
        log_file = Config.LOG_FILE
    
    # Remove existing handlers (prevent duplication)
    logger = logging.getLogger('dog_breed_classifier')
    logger.handlers.clear()
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    try:
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Setup file handler
        try:
            # Create log directory
            log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
            os.makedirs(log_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging system initialization completed - Level: {log_level}, File: {log_file}")
            
        except Exception as file_error:
            logger.warning(f"File logging setup failed: {file_error}")
            logger.info("Using console logging only.")
            
    except Exception as e:
        # Use default logger on logging setup failure
        print(f"Logging setup error: {e}")
        print("Using default logging configuration.")
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger('dog_breed_classifier')
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Returns a logger instance.
    
    Args:
        name (str, optional): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    if name is None:
        name = 'dog_breed_classifier'
    return logging.getLogger(name)
