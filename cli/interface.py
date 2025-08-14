"""
CLI interface for Dog Breed Classifier V3.5

This module provides command-line interface functionality,
extracted from the original V2_3_5 single file.
"""

import os
import argparse
from typing import Optional

from ..config import get_logger, DATASET_PATH, CUSTOM_MODEL_PATH, CLASS_NAMES_PATH
from ..utils import validate_file_exists
from ..core import predict_breed

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    [REFACTOR] Command-line argument parsing
    - Before: Included within main function
    - Improved: Separated into standalone function for CLI logic independence
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Dog Breed Classification System (Auto Mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s                           # Interactive mode for image path input
  %(prog)s /path/to/image.jpg        # Direct image path specification
  %(prog)s --train                   # Model training mode
        """
    )
    
    parser.add_argument(
        'image_path', 
        type=str, 
        nargs='?', 
        help='Image path for prediction (optional, interactive prompt if not provided)'
    )
    
    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Enable training mode (uses default dataset)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose log output'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set log level (default: INFO)'
    )
    
    return parser.parse_args()


def get_image_path_interactive() -> Optional[str]:
    """
    Interactively receives image path input.
    
    Returns:
        Optional[str]: Input image path or None
    """
    try:
        logger.info("Please enter the image path directly (copy and paste path is possible):")
        image_path = input("> ").strip()
        
        if not image_path:
            logger.warning("No image path was entered.")
            return None
        
        # Remove quotes from input (in case user copied with quotes)
        if (image_path.startswith('"') and image_path.endswith('"')) or \
           (image_path.startswith("'") and image_path.endswith("'")):
            image_path = image_path[1:-1]
        
        return image_path
        
    except (KeyboardInterrupt, EOFError):
        logger.info("Input was cancelled.")
        return None
    except Exception as e:
        logger.error(f"Error during image path input: {e}")
        return None


def validate_model_files() -> bool:
    """
    Checks if required model files exist.
    
    Returns:
        bool: True if all files exist, False otherwise
    """
    has_custom_model = CUSTOM_MODEL_PATH is not None and os.path.exists(CUSTOM_MODEL_PATH)
    has_class_names = os.path.exists(CLASS_NAMES_PATH)
    
    if not has_custom_model:
        logger.error(f"Custom model file not found: {CUSTOM_MODEL_PATH}")
        
    if not has_class_names:
        logger.error(f"Class names file not found: {CLASS_NAMES_PATH}")
    
    if not (has_custom_model and has_class_names):
        logger.error("Custom model files are missing. Cannot run the program.")
        logger.info("Use --train option to train the model.")
        return False
    
    return True


def run_training_mode() -> bool:
    """
    Runs training mode.
    
    Returns:
        bool: True if training succeeds, False if it fails
    """
    try:
        from ..core.model import create_custom_model, save_model_and_classes
        from ..core.data_processing import create_data_generators, get_class_names_from_generator
        from ..analysis import visualize_training_history
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
        from ..config import Config
        
        logger.info(f"Starting model training from default dataset path '{DATASET_PATH}'.")
        
        # Create data generators
        train_gen, val_gen, num_classes = create_data_generators(DATASET_PATH)
        
        # Extract class names
        class_names = get_class_names_from_generator(train_gen)
        
        # Create model
        model = create_custom_model(num_classes)
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(CUSTOM_MODEL_PATH, monitor='val_accuracy', save_best_only=True),
            EarlyStopping(monitor='val_loss', patience=Config.PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
        ]
        
        # Train model
        logger.info("Starting model training")
        history = model.fit(
            train_gen,
            epochs=Config.EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=2
        )
        
        # Save class names
        save_model_and_classes(model, class_names)
        
        # Visualize training results
        visualize_training_history(history)
        
        logger.info(f"Final model saved: {CUSTOM_MODEL_PATH}")
        logger.info(f"Class names saved: {CLASS_NAMES_PATH}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        return False


def run_prediction_mode(image_path: str) -> bool:
    """
    Runs prediction mode.
    
    Args:
        image_path (str): Image path for prediction
        
    Returns:
        bool: True if prediction succeeds, False if it fails
    """
    try:
        # Check image file existence
        if not validate_file_exists(image_path, "image file"):
            return False
        
        # Check model files
        if not validate_model_files():
            return False
        
        # Execute prediction
        logger.info(f"Starting automatic prediction for image '{os.path.basename(image_path)}'.")
        return predict_breed(image_path)
        
    except Exception as e:
        logger.error(f"Error during prediction mode execution: {e}")
        return False


def main() -> None:
    """
    [REFACTOR] Main CLI function
    - Before: All CLI logic included in one long function
    - Improved: Decomposed into smaller functions for better readability and maintainability
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set log level (if needed)
        if args.verbose:
            import logging
            logging.getLogger('dog_breed_classifier').setLevel(logging.DEBUG)
        
        # Training mode
        if args.train:
            success = run_training_mode()
            if not success:
                logger.error("Model training failed.")
                return
            return
        
        # Prediction mode
        image_path = args.image_path
        
        # Request interactive input if image path is not provided
        if not image_path:
            image_path = get_image_path_interactive()
            if not image_path:
                return
        
        # Execute prediction
        success = run_prediction_mode(image_path)
        if not success:
            logger.error("Dog breed prediction failed.")
        
    except KeyboardInterrupt:
        logger.info("Program was interrupted by user.")
    except Exception as e:
        logger.error(f"Error occurred during program execution: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def run_cli() -> None:
    """
    CLI entry point function
    """
    main()


if __name__ == "__main__":
    run_cli()
