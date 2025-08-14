"""
Analysis package for Dog Breed Classifier V3.5

This package provides model evaluation and visualization capabilities.
"""

from .evaluation import (
    create_confusion_matrix,
    create_class_performance_report,
    visualize_misclassified_samples,
    analyze_model_complexity,
    visualize_training_history,
    plot_confusion_matrix,
    DetailedTrainingMonitor
)

from .visualization import (
    generate_gradcam_heatmap,
    visualize_gradcam,
    plot_prediction_distribution,
    visualize_feature_maps,
    create_model_architecture_plot
)

__all__ = [
    # Evaluation
    'create_confusion_matrix',
    'create_class_performance_report',
    'visualize_misclassified_samples',
    'analyze_model_complexity',
    'visualize_training_history',
    'plot_confusion_matrix',
    'DetailedTrainingMonitor',
    
    # Visualization
    'generate_gradcam_heatmap',
    'visualize_gradcam',
    'plot_prediction_distribution',
    'visualize_feature_maps',
    'create_model_architecture_plot'
]
