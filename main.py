#!/usr/bin/env python3
"""
Main entry point for Dog Breed Classifier V3.5

This is the main entry point for the modularized dog breed classifier,
providing a clean interface to the CLI functionality.
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import setup_logging
from utils import initialize_system
from cli import run_cli


def main():
    """
    Main entry point for the Dog Breed Classifier V3.5
    
    This function initializes the system and runs the CLI interface.
    """
    try:
        # Initialize logging system
        logger = setup_logging()
        logger.info("=== Dog Breed Classifier V3.5 Started ===")
        
        # Initialize system (GPU, fonts, warnings)
        initialize_system()
        
        # Run CLI interface
        run_cli()
        
        logger.info("=== Dog Breed Classifier V3.5 Finished ===")
        
    except Exception as e:
        print(f"Error occurred during system initialization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
