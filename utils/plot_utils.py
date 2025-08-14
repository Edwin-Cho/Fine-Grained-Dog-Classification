"""
Plot utilities for Dog Breed Classifier V3.5

This module provides safe plotting and visualization utilities,
extracted from the original V2_3_5 single file.
"""

import os
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..config import get_logger, Config

logger = get_logger(__name__)


def safe_plot_display(fig: plt.Figure, save_path: str = None, show: bool = True, 
                     dpi: int = None, bbox_inches: str = 'tight') -> bool:
    """
    [REFACTOR] Safe plot saving and display utility function
    - Before: Individual save/display handling in each visualization function
    - Improved: Centralized plot processing logic to eliminate code duplication and strengthen error handling
    
    Args:
        fig (plt.Figure): matplotlib Figure object to save/display
        save_path (str, optional): File path to save
        show (bool): Whether to display on screen (default: True)
        dpi (int, optional): Resolution for saving (default: Config.DPI)
        bbox_inches (str): Margin setting for saving (default: 'tight')
        
    Returns:
        bool: True if successful, False if failed
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>> safe_plot_display(fig, "output.png", show=False)
        True
    """
    success = True
    
    if dpi is None:
        dpi = Config.DPI
    
    try:
        # Save file
        if save_path:
            try:
                # Create save directory
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
                           facecolor='white', edgecolor='none')
                logger.info(f"Plot saved successfully: {save_path}")
                
            except Exception as save_error:
                logger.error(f"Plot save failed ({save_path}): {save_error}")
                success = False
        
        # Display on screen
        if show:
            try:
                plt.show()
                logger.debug("Plot display completed")
            except Exception as show_error:
                logger.warning(f"Plot display failed: {show_error}")
                success = False
        
        # Memory cleanup
        try:
            plt.close(fig)
            logger.debug("Plot memory cleanup completed")
        except Exception as close_error:
            logger.warning(f"Plot memory cleanup failed: {close_error}")
            
    except Exception as e:
        logger.error(f"Exception occurred during plot processing: {e}")
        success = False
        
        # Attempt memory cleanup even on error
        try:
            plt.close(fig)
        except:
            pass
    
    return success


def create_figure_with_size(figsize: Tuple[int, int] = None) -> plt.Figure:
    """
    Creates a Figure with specified size.
    
    Args:
        figsize (Tuple[int, int], optional): Figure size (default: Config.FIGURE_SIZE)
        
    Returns:
        plt.Figure: Created Figure object
    """
    if figsize is None:
        figsize = Config.FIGURE_SIZE
    
    try:
        fig = plt.figure(figsize=figsize)
        logger.debug(f"Figure creation completed: {figsize}")
        return fig
    except Exception as e:
        logger.error(f"Figure creation failed: {e}")
        # Retry with default size
        return plt.figure(figsize=(10, 6))


def setup_plot_style() -> None:
    """
    Sets up plot style.
    """
    try:
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.edgecolor'] = 'none'
        logger.debug("Plot style setup completed")
    except Exception as e:
        logger.warning(f"Plot style setup failed: {e}")


def close_all_plots() -> None:
    """
    Closes all open plots and cleans up memory.
    """
    try:
        plt.close('all')
        logger.debug("All plots cleanup completed")
    except Exception as e:
        logger.warning(f"Plot cleanup failed: {e}")


def get_color_palette(n_colors: int) -> list:
    """
    Returns a color palette with specified number of colors.
    
    Args:
        n_colors (int): Number of colors needed
        
    Returns:
        list: List of colors
    """
    try:
        import seaborn as sns
        colors = sns.color_palette("husl", n_colors)
        logger.debug(f"Color palette creation completed: {n_colors} colors")
        return colors
    except ImportError:
        # Use matplotlib default colors when seaborn is not available
        colors = plt.cm.tab10(range(min(n_colors, 10)))
        if n_colors > 10:
            # Repeat colors for more than 10
            colors = list(colors) * ((n_colors // 10) + 1)
            colors = colors[:n_colors]
        logger.debug(f"Using default color palette: {n_colors} colors")
        return colors
    except Exception as e:
        logger.warning(f"Color palette creation failed: {e}")
        # Return default colors
        return ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:n_colors]
