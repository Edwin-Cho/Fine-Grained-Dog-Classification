"""
Prediction functionality for Dog Breed Classifier V3.5

This module provides breed prediction and analysis functionality,
extracted from the original V2_3_5 single file.
"""

import time
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Optional

from ..config import get_logger, Config
from .data_processing import preprocess_image
from .model import load_model_and_classes

logger = get_logger(__name__)


def process_prediction_results(predictions: np.ndarray, class_names: np.ndarray, 
                             top_k: int = None) -> List[Tuple[str, float]]:
    """
    [REFACTOR] Process model prediction results to top K breeds
    - Before: Using hardcoded default values
    - Improved: Using Config class-based settings, added type hinting
    
    Args:
        predictions (np.ndarray): Model prediction result array
        class_names (np.ndarray): Class names array
        top_k (int, optional): Return top K results (default: Config.TOP_K_PREDICTIONS)
        
    Returns:
        List[Tuple[str, float]]: List in format [(breed_name, probability), ...]
        
    Example:
        >>> predictions = np.array([[0.7, 0.2, 0.1]])
        >>> class_names = np.array(['Golden Retriever', 'Labrador', 'Beagle'])
        >>> process_prediction_results(predictions, class_names, 2)
        [('Golden Retriever', 0.7), ('Labrador', 0.2)]
    """
    if top_k is None:
        top_k = Config.TOP_K_PREDICTIONS
        
    if predictions is None or len(predictions) == 0:
        logger.warning("Prediction results are empty.")
        return []
    
    try:
        # Extract top K indices
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        # Create (breed_name, probability) tuple list
        results = [(class_names[i], float(predictions[0][i])) for i in top_indices]
        
        logger.debug(f"Top {top_k} prediction results processing completed")
        return results
        
    except Exception as e:
        logger.error(f"Prediction results processing error: {e}")
        return []


def perform_prediction(model: tf.keras.Model, image_path: str, 
                      class_names: np.ndarray) -> Optional[List[Tuple[str, float]]]:
    """
    [REFACTOR] Function that performs actual prediction
    - Before: Included inside predict_breed function
    - Improved: Separated into standalone function for better testability
    
    Args:
        model (tf.keras.Model): Model to use for prediction
        image_path (str): Image file path
        class_names (np.ndarray): Class names array
    
    Returns:
        Optional[List[Tuple[str, float]]]: Prediction results list [(breed, confidence), ...] or None
    """
    try:
        # Image preprocessing
        img_array = preprocess_image(image_path)
        if img_array is None:
            logger.error("Image preprocessing failed")
            return None
        
        # Perform prediction
        logger.info("Performing prediction...")
        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        end_time = time.time()
        
        # Extract top K prediction results
        results = process_prediction_results(predictions, class_names, Config.TOP_K_PREDICTIONS)
        
        # Log prediction results
        logger.info(f"Prediction completed (elapsed time: {end_time - start_time:.2f}s)")
        for i, (breed, probability) in enumerate(results, 1):
            logger.info(f"  {i}. {breed}: {probability:.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction execution error: {e}")
        return None


def calculate_entropy(probabilities: List[float]) -> float:
    """
    Calculates the entropy of a probability distribution.
    
    Args:
        probabilities (List[float]): List of probabilities
        
    Returns:
        float: Entropy value
    """
    try:
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    except Exception as e:
        logger.error(f"Entropy calculation error: {e}")
        return 0.0


def is_mixed_breed_by_threshold(results: List[Tuple[str, float]]) -> Tuple[bool, str]:
    """
    Mixed breed determination based on threshold
    
    Args:
        results (List[Tuple[str, float]]): List of prediction results
        
    Returns:
        Tuple[bool, str]: (Mixed breed status, Reasoning)
    """
    if len(results) < 2:
        return False, "Insufficient prediction results"
    
    top1_conf = results[0][1]
    top2_conf = results[1][1]
    
    # Determine as mixed breed if probability difference between top 2 breeds is less than threshold
    confidence_diff = top1_conf - top2_conf
    
    if confidence_diff < Config.MIX_BREED_THRESHOLD:
        return True, f"Probability difference {confidence_diff:.3f} < threshold {Config.MIX_BREED_THRESHOLD}"
    
    return False, f"Probability difference {confidence_diff:.3f} >= threshold {Config.MIX_BREED_THRESHOLD}"


def detect_multi_breed_mix(results: List[Tuple[str, float]]) -> Tuple[str, List[str]]:
    """
    Multi-breed mix detection
    
    Args:
        results (List[Tuple[str, float]]): List of prediction results
        
    Returns:
        Tuple[str, List[str]]: (Mix type, List of major breeds)
    """
    if len(results) < 2:
        return "single_breed", []
    
    # Extract breeds with significant probabilities (5% or higher)
    significant_breeds = [(breed, conf) for breed, conf in results if conf > 0.05]
    
    if len(significant_breeds) == 1:
        return "single_breed", [significant_breeds[0][0]]
    elif len(significant_breeds) == 2:
        return "simple_mix", [breed for breed, _ in significant_breeds]
    else:
        return "complex_mix", [breed for breed, _ in significant_breeds]


def analyze_mixed_breed(results: List[Tuple[str, float]]) -> Tuple[bool, str, str]:
    """
    [REFACTOR] Mixed breed analysis function
    - Before: Included within predict_breed function
    - Improved: Separated into standalone function for logic clarity
    
    Args:
        results (List[Tuple[str, float]]): List of prediction results
    
    Returns:
        Tuple[bool, str, str]: (Mixed breed status, Mixed breed info, Detection method)
    """
    if len(results) < 2:
        return False, "", ""
    
    # Extract probability values
    top1_conf = results[0][1]
    top2_conf = results[1][1]
    
    # Extract breed names (only part after hyphen)
    breed1_name = results[0][0].split('-')[-1] if '-' in results[0][0] else results[0][0].lower()
    
    # 1. Threshold-based mixed breed determination
    threshold_result, threshold_info = is_mixed_breed_by_threshold(results)
    
    # 2. Entropy-based uncertainty analysis
    probs = [result[1] for result in results[:5]]  # Consider only top 5
    entropy_value = calculate_entropy(probs)
    entropy_threshold = 0.7
    
    # 3. Multi-breed mix detection
    mix_type, major_breeds = detect_multi_breed_mix(results)
    
    # Comprehensive assessment
    is_mixed = (threshold_result or 
                entropy_value > entropy_threshold or 
                mix_type in ["simple_mix", "complex_mix"] or
                (breed1_name == 'keeshond' and top2_conf > 0.02))
    
    if not is_mixed:
        return False, "", ""
    
    # Generate mixed breed information
    total_conf = top1_conf + top2_conf
    if total_conf > 0:
        ratio1 = int(round((top1_conf / total_conf) * 10))
        ratio2 = 10 - ratio1
        mixed_info = f"Mixed breed (Primary breeds: {', '.join(major_breeds[:2])})"
        if len(major_breeds) > 2:
            mixed_info += f" + {len(major_breeds)-2} additional breeds"
    
    # Case of 3 or more breeds
    if len(results) >= 3 and results[2][1] > 0.05:
        breed3_name = results[2][0].split('-')[-1] if '-' in results[2][0] else results[2][0]
        if total_conf > 0:
            mixed_info = f"Mixed breed (Primary breeds: {', '.join(major_breeds[:2])} + {breed3_name})"
        else:
            mixed_info = f"Mixed breed ({breed1_name} + {breed2_name} + {breed3_name})"
    
    # Determine detection method
    if threshold_result:
        detection_method = "Threshold-based"
    elif entropy_value > entropy_threshold:
        detection_method = "Entropy-based"
    elif mix_type != "single_breed":
        detection_method = "Multi-breed analysis"
    else:
        detection_method = "Special case"
    
    logger.info(f"Mixed breed detected: {mixed_info} (Method: {detection_method})")
    return True, mixed_info, detection_method


def process_confidence_level(confidence: float) -> str:
    """
    [REFACTOR] Confidence level message processing function
    - Before: Included within predict_breed function
    - Improved: Separated into standalone function for enhanced reusability
    
    Args:
        confidence (float): Highest confidence value
        
    Returns:
        str: Message based on confidence level
    """
    if confidence > Config.HIGH_CONFIDENCE:
        message = f"Accurate prediction with high confidence ({confidence*100:.2f}%)"
        logger.info(f"[Auto Assessment] {message}")
        return message
    elif confidence > Config.MEDIUM_CONFIDENCE:
        message = f"Using prediction results with medium confidence ({confidence*100:.2f}%)"
        logger.info(f"[Auto Assessment] {message}")
        return message
    else:
        message = f"Low confidence ({confidence*100:.2f}%) - Retaking photo recommended"
        logger.warning(f"Highest confidence ({confidence*100:.2f}%) is below {Config.MEDIUM_CONFIDENCE*100}%.")
        logger.info("Please upload another photo that better shows the breed characteristics.")
        logger.info("Tip: Photos taken from the front or side that clearly show facial and body features work best.")
        return message


def display_prediction_results(results: List[Tuple[str, float]], mixed_info: str = None) -> None:
    """
    [REFACTOR] Prediction results display function
    - Before: Included within predict_breed function
    - Improved: Separated into standalone function for output logic independence
    
    Args:
        results (List[Tuple[str, float]]): List of prediction results
        mixed_info (str, optional): Mixed breed information
    """
    if not results:
        logger.warning("No prediction results to display.")
        return
    
    logger.info("\n=== Final Prediction Results ===")
    
    # Display mixed breed information first if available
    if mixed_info:
        logger.info(f"🐕 {mixed_info}")
    else:
        # Display single breed results
        top_breed, top_confidence = results[0]
        confidence_message = process_confidence_level(top_confidence)
        logger.info(f"🐕 Predicted breed: {top_breed}")
        logger.info(f"📊 {confidence_message}")
    
    # Display top prediction results
    logger.info("\n📈 Top prediction results:")
    for i, (breed, probability) in enumerate(results, 1):
        logger.info(f"  {i}. {breed}: {probability:.2%}")


def predict_breed(image_path: str) -> bool:
    """
    [REFACTOR] Main breed prediction function
    - Before: All logic included in one long function
    - Improved: Decomposed into smaller functions for enhanced readability and maintainability
    
    Args:
        image_path (str): Image file path for prediction
        
    Returns:
        bool: True on successful prediction, False on failure
    """
    try:
        # 1. Load model and classes
        model, class_names = load_model_and_classes()
        if model is None or class_names is None:
            logger.error("Model or class names loading failed")
            return False
        
        # 2. Perform prediction
        results = perform_prediction(model, image_path, class_names)
        if not results:
            logger.error("Prediction execution failed")
            return False
        
        # 3. Mixed breed analysis
        is_mixed, mixed_info, detection_method = analyze_mixed_breed(results)
        
        # 4. Display results
        display_prediction_results(results, mixed_info if is_mixed else None)
        
        logger.info("Breed prediction completed")
        return True
        
    except Exception as e:
        logger.error(f"Error occurred during breed prediction: {e}")
        return False
