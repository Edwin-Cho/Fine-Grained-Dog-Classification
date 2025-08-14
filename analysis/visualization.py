"""
Visualization functionality for Dog Breed Classifier V3.5

This module provides advanced visualization capabilities including GradCAM,
extracted from the original V2_3_5 single file.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Optional, Tuple

from ..config import get_logger, Config
from ..utils import safe_plot_display, create_figure_with_size

logger = get_logger(__name__)


def generate_gradcam_heatmap(model: tf.keras.Model, img_array: np.ndarray, 
                           layer_name: str, class_idx: int) -> Optional[np.ndarray]:
    """
    [REFACTOR] GradCAM heatmap generation
    - Before: Nested function within plot_confusion_matrix function
    - Improved: Separated into standalone function for model interpretation feature independence
    
    Args:
        model (tf.keras.Model): Trained model
        img_array (np.ndarray): Input image array
        layer_name (str): Layer name to analyze
        class_idx (int): Class index
    
    Returns:
        Optional[np.ndarray]: GradCAM heatmap or None (on failure)
    """
    try:
        # Create model for GradCAM
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Extract gradients and feature maps
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Calculate weights
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
        
        # Generate heatmap
        cam = cv2.resize(cam.numpy(), (img_array.shape[2], img_array.shape[1]))
        cam = np.maximum(cam, 0)
        
        # Normalize
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        logger.debug(f"GradCAM heatmap generation completed: {layer_name}, class {class_idx}")
        return heatmap
        
    except Exception as e:
        logger.error(f"GradCAM generation error: {e}")
        return None


def visualize_gradcam(model: tf.keras.Model, image_path: str, class_idx: int,
                     layer_name: str = None, save_path: str = None) -> bool:
    """
    Performs GradCAM visualization.
    
    Args:
        model (tf.keras.Model): Trained model
        image_path (str): Image file path
        class_idx (int): Class index to analyze
        layer_name (str, optional): Layer name to analyze
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        from ..core.data_processing import preprocess_image
        
        # Image preprocessing
        img_array = preprocess_image(image_path)
        if img_array is None:
            logger.error("Image preprocessing failed")
            return False
        
        # Set default layer name
        if layer_name is None:
            # Last convolutional layer of ResNet50
            layer_name = 'conv5_block3_out'
        
        # Generate GradCAM heatmap
        heatmap = generate_gradcam_heatmap(model, img_array, layer_name, class_idx)
        if heatmap is None:
            return False
        
        # Load original image
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, Config.IMAGE_SIZE)
        
        # Visualization
        fig = create_figure_with_size((15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('off')
        
        # GradCAM heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title(f'GradCAM Heatmap\n(Layer: {layer_name})')
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        plt.imshow(overlay)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = 'gradcam_visualization.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"GradCAM visualization saved successfully: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"GradCAM visualization failed: {e}")
        return False


def plot_prediction_distribution(predictions: np.ndarray, class_names: list,
                                top_k: int = None, save_path: str = None) -> bool:
    """
    Visualizes prediction probability distribution.
    
    Args:
        predictions (np.ndarray): Prediction probability array
        class_names (list): List of class names
        top_k (int, optional): Display only top K
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        if top_k is None:
            top_k = Config.TOP_K_PREDICTIONS
        
        # Extract top K
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        top_probs = predictions[0][top_indices]
        top_classes = [class_names[i] for i in top_indices]
        
        # Visualization
        fig = create_figure_with_size((12, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_classes)))
        bars = plt.bar(range(len(top_classes)), top_probs, color=colors)
        
        plt.title(f'Top {top_k} Prediction Probabilities', fontsize=14)
        plt.xlabel('Dog Breeds')
        plt.ylabel('Probability')
        plt.xticks(range(len(top_classes)), top_classes, rotation=45, ha='right')
        
        # Display probability values
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = 'prediction_distribution.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"Prediction distribution visualization saved successfully: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Prediction distribution visualization failed: {e}")
        return False


def visualize_feature_maps(model: tf.keras.Model, img_array: np.ndarray,
                          layer_name: str, save_path: str = None) -> bool:
    """
    Visualizes feature maps.
    
    Args:
        model (tf.keras.Model): Trained model
        img_array (np.ndarray): Input image array
        layer_name (str): Layer name to visualize
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        # Create model for feature map extraction
        feature_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        
        # Extract feature maps
        feature_maps = feature_model.predict(img_array, verbose=0)
        
        # Determine number of feature maps to visualize (maximum 16)
        n_features = min(16, feature_maps.shape[-1])
        
        # Visualization
        fig = create_figure_with_size((16, 16))
        
        for i in range(n_features):
            plt.subplot(4, 4, i + 1)
            plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
            plt.title(f'Feature Map {i+1}')
            plt.axis('off')
        
        plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=16)
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = f'feature_maps_{layer_name}.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"Feature map visualization saved successfully: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Feature map visualization failed: {e}")
        return False


def create_model_architecture_plot(model: tf.keras.Model, save_path: str = None) -> bool:
    """
    Visualizes model architecture.
    
    Args:
        model (tf.keras.Model): Model to visualize
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        # Use Keras plot_model
        from tensorflow.keras.utils import plot_model
        
        if save_path is None:
            save_path = 'model_architecture.png'
        
        plot_model(
            model,
            to_file=save_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96
        )
        
        logger.info(f"Model architecture visualization saved successfully: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Model architecture visualization failed: {e}")
        return False
