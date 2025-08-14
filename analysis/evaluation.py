"""
Evaluation functionality for Dog Breed Classifier V3.5

This module provides model evaluation and performance analysis functionality,
extracted from the original V2_3_5 single file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Optional, Tuple
from tensorflow.keras import backend as K

from ..config import get_logger, Config
from ..utils import safe_plot_display, create_figure_with_size

logger = get_logger(__name__)


def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: List[str], save_path: str = None) -> bool:
    """
    [REFACTOR] Confusion Matrix generation and visualization
    - Before: Included within plot_confusion_matrix function
    - Improved: Separated into standalone function for evaluation feature independence
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): List of class names
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        # Calculate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Visualization
        fig = create_figure_with_size((12, 10))
        
        # Original Confusion Matrix
        plt.subplot(2, 1, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Count)', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Normalized Confusion Matrix
        plt.subplot(2, 1, 2)
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Normalized)', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = 'confusion_matrix.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"Confusion Matrix saved successfully: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Confusion Matrix generation failed: {e}")
        return False


def create_class_performance_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str]) -> str:
    """
    [REFACTOR] Class-wise performance report generation
    - Before: Included within plot_confusion_matrix function
    - Improved: Separated into standalone function for performance analysis feature independence
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): List of class names
        
    Returns:
        str: Performance report string
    """
    try:
        # Generate Classification Report
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            output_dict=False,
            zero_division=0
        )
        
        logger.info("=== Class-wise Performance Report ===")
        logger.info(f"\n{report}")
        
        # Dictionary format report for detailed analysis
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Identify low-performance classes
        low_performance_classes = []
        for class_name in class_names:
            if class_name in report_dict:
                f1_score = report_dict[class_name]['f1-score']
                if f1_score < 0.7:  # F1 score below 70%
                    low_performance_classes.append((class_name, f1_score))
        
        if low_performance_classes:
            logger.warning(f"Low-performance classes (F1 < 0.7): {low_performance_classes}")
            for class_name, f1_score in low_performance_classes:
                logger.warning(f"  - {class_name}: F1-Score {f1_score:.3f}")
        
        return report
        
    except Exception as e:
        logger.error(f"Performance report generation failed: {e}")
        return f"Performance report generation failed: {e}"


def visualize_misclassified_samples(model: tf.keras.Model, test_gen, 
                                  class_names: List[str], num_samples: int = 5,
                                  save_path: str = None) -> bool:
    """
    [REFACTOR] Misclassified samples visualization
    - Before: Included within plot_confusion_matrix function
    - Improved: Separated into standalone function for misclassification analysis feature independence
    
    Args:
        model (tf.keras.Model): Trained model
        test_gen: Test data generator
        class_names (List[str]): List of class names
        num_samples (int, optional): Number of samples to display
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        images, true_labels = [], []
        
        # Collect test images and labels
        for i, (img_batch, label_batch) in enumerate(test_gen):
            images.extend(img_batch)
            true_labels.extend(np.argmax(label_batch, axis=1))
            if len(images) >= 100:  # Sufficient samples collected
                break
        
        if not images:
            logger.warning("No test data available.")
            return False
        
        # Perform predictions
        preds = model.predict(np.array(images), verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        
        # Find misclassified samples
        misclassified_idx = np.where(pred_labels != true_labels)[0]
        
        if len(misclassified_idx) == 0:
            logger.info("No misclassified samples found.")
            plt.tight_layout()
        
        # Visualization
        fig = create_figure_with_size((15, num_samples*2))
        
        for i, idx in enumerate(misclassified_idx[:num_samples]):
            # Display image
            plt.subplot(num_samples, 2, i*2+1)
            plt.imshow(images[idx]/255.)  # Denormalize
            plt.title(f"Actual: {class_names[true_labels[idx]]}")
            plt.axis('off')
            
            # Display prediction distribution
            plt.subplot(num_samples, 2, i*2+2)
            plt.bar(range(len(class_names)), preds[idx])
            plt.xticks(range(len(class_names)), class_names, rotation=90)
            plt.title(f"Predicted: {class_names[pred_labels[idx]]}")
        
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = 'misclassified_samples.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"Misclassified samples visualization saved successfully: {save_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Misclassified samples visualization failed: {e}")
        return False


def analyze_model_complexity(model: tf.keras.Model) -> dict:
    """
    [REFACTOR] Model complexity analysis
    - Before: Nested function within plot_confusion_matrix function
    - Improved: Separated into standalone function for model analysis feature independence
    
    Args:
        model (tf.keras.Model): Model to analyze
        
    Returns:
        dict: Model complexity analysis results
    """
    try:
        # Layer-wise parameter and computation analysis
        total_params = model.count_params()
        trainable_params = sum([K.count_params(w) for w in model.trainable_weights])
        
        analysis_result = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_ratio': trainable_params/total_params*100 if total_params > 0 else 0,
            'layer_details': []
        }
        
        logger.info("=== Model Complexity Analysis ===")
        logger.info(f"Model complexity analysis completed - Total parameters: {analysis_result['total_params']:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Fixed parameters: {total_params - trainable_params:,}")
        logger.info(f"Trainable ratio: {analysis_result['trainable_ratio']:.1f}%")
        
        logger.info("\n=== Layer-wise Detailed Information ===")
        for i, layer in enumerate(model.layers):
            layer_params = layer.count_params()
            if layer_params > 0:  # Output only layers with parameters
                layer_info = {
                    'index': i,
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'params': layer_params,
                    'output_shape': str(layer.output_shape)
                }
                analysis_result['layer_details'].append(layer_info)
                
                logger.info(f"Layer {i:2d}: {layer.name:<20} | "
                           f"Type: {layer.__class__.__name__:<15} | "
                           f"Params: {layer_params:>8,} | "
                           f"Shape: {layer.output_shape}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Model complexity analysis failed: {e}")
        return {}


def visualize_training_history(history: tf.keras.callbacks.History, 
                             save_path: str = None) -> bool:
    """
    [REFACTOR] Training curve visualization
    - Before: visualize_training function
    - Improved: Added type hinting and safe plot handling
    
    Args:
        history (tf.keras.callbacks.History): Training history
        save_path (str, optional): Save path
        
    Returns:
        bool: True on success, False on failure
    """
    try:
        fig = create_figure_with_size((14, 5))
        
        # Accuracy curve
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Trend', fontsize=14)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Loss curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation') 
        plt.title('Loss Trend', fontsize=14)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        
        # Save and display
        if save_path is None:
            save_path = 'training_performance.png'
        
        success = safe_plot_display(fig, save_path, show=True)
        
        if success:
            logger.info(f"Training results visualization saved to '{save_path}'.")
        
        return success
        
    except Exception as e:
        logger.error(f"Training curve visualization failed: {e}")
        return False


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str]) -> None:
    """
    [REFACTOR] Integrated performance analysis function
    - Before: All analysis features included in one long function
    - Improved: Combines small functions to provide clean interface
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str]): List of class names
    """
    logger.info("=== Model Performance Analysis Started ===")
    
    # 1. Generate Confusion Matrix
    create_confusion_matrix(y_true, y_pred, class_names)
    
    # 2. Class-wise performance report
    create_class_performance_report(y_true, y_pred, class_names)
    
    logger.info("=== Model Performance Analysis Completed ===")


class DetailedTrainingMonitor(tf.keras.callbacks.Callback):
    """
    [REFACTOR] Detailed training monitoring callback
    - Before: Nested class within plot_confusion_matrix function
    - Improved: Separated into standalone class for training monitoring feature independence
    """
    
    def __init__(self):
        super(DetailedTrainingMonitor, self).__init__()
        self.batch_losses = []
        self.batch_accuracies = []
        
    def on_batch_end(self, batch, logs=None):
        """Record loss and accuracy at batch end"""
        if logs:
            self.batch_losses.append(logs.get('loss', 0))
            self.batch_accuracies.append(logs.get('accuracy', 0))
            
    def on_epoch_end(self, epoch, logs=None):
        """Visualize batch-wise performance at epoch end"""
        if not self.batch_losses:
            return
            
        try:
            fig = create_figure_with_size((15, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.batch_losses)
            plt.title(f'Epoch {epoch+1} Batch-wise Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            
            plt.subplot(1, 2, 2)
            plt.plot(self.batch_accuracies)
            plt.title(f'Epoch {epoch+1} Batch-wise Accuracy')
            plt.xlabel('Batch')
            plt.ylabel('Accuracy')
            
            plt.tight_layout()
            
            save_path = f'epoch_{epoch+1}_details.png'
            safe_plot_display(fig, save_path, show=False)
            
            logger.info(f"Epoch {epoch+1} detailed analysis saved: {save_path}")
            
        except Exception as e:
            logger.error(f"Epoch detailed analysis failed: {e}")
