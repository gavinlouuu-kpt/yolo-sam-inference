import numpy as np
from typing import Dict, List, Any
from .logger import setup_logger

logger = setup_logger(__name__)

def calculate_summary_statistics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate summary statistics for a list of metrics.
    
    Args:
        metrics_list: List of dictionaries containing metrics for each cell
        
    Returns:
        Dictionary containing mean and std for each metric
    """
    if not metrics_list:
        return {}
        
    # Extract all metrics
    areas = [m['area'] for m in metrics_list]
    circularities = [m['circularity'] for m in metrics_list]
    deformabilities = [m['deformability'] for m in metrics_list]
    perimeters = [m['perimeter'] for m in metrics_list]
    
    # Shape metrics
    area_ratios = [m['area_ratio'] for m in metrics_list]
    convex_hull_areas = [m['convex_hull_area'] for m in metrics_list]
    aspect_ratios = [m['aspect_ratio'] for m in metrics_list]
    
    # Brightness metrics
    mean_brightnesses = [m['mean_brightness'] for m in metrics_list]
    brightness_stds = [m['brightness_std'] for m in metrics_list]
    
    return {
        'basic_metrics': {
            'area': (np.mean(areas), np.std(areas)),
            'circularity': (np.mean(circularities), np.std(circularities)),
            'deformability': (np.mean(deformabilities), np.std(deformabilities)),
            'perimeter': (np.mean(perimeters), np.std(perimeters))
        },
        'shape_metrics': {
            'area_ratio': (np.mean(area_ratios), np.std(area_ratios)),
            'convex_hull_area': (np.mean(convex_hull_areas), np.std(convex_hull_areas)),
            'aspect_ratio': (np.mean(aspect_ratios), np.std(aspect_ratios))
        },
        'brightness_metrics': {
            'mean_brightness': (np.mean(mean_brightnesses), np.std(mean_brightnesses)),
            'brightness_std': (np.mean(brightness_stds), np.std(brightness_stds))
        }
    }

def report_summary_statistics(stats: Dict[str, Dict[str, tuple]]):
    """Report summary statistics to logger.
    
    Args:
        stats: Dictionary containing mean and std for each metric category
    """
    if not stats:
        return
        
    logger.info("Summary statistics:")
    
    # Basic metrics
    if 'basic_metrics' in stats:
        for name, (mean, std) in stats['basic_metrics'].items():
            units = "pixels" if name in ['area', 'perimeter'] else ""
            logger.info(f"{name.capitalize()}: {mean:.2f} ± {std:.2f} {units}".strip())
    
    # Shape metrics
    if 'shape_metrics' in stats:
        for name, (mean, std) in stats['shape_metrics'].items():
            units = "pixels" if name == 'convex_hull_area' else ""
            logger.info(f"{name.replace('_', ' ').capitalize()}: {mean:.2f} ± {std:.2f} {units}".strip())
    
    # Brightness metrics
    if 'brightness_metrics' in stats:
        for name, (mean, std) in stats['brightness_metrics'].items():
            logger.info(f"{name.replace('_', ' ').capitalize()}: {mean:.2f} ± {std:.2f}")

def report_cell_details(cell_idx: int, metrics: Dict[str, Any]):
    """Report detailed metrics for a single cell.
    
    Args:
        cell_idx: Index of the cell
        metrics: Dictionary containing cell metrics
    """
    logger.debug(f"\nCell {cell_idx + 1} details:")
    
    # Basic metrics
    logger.debug(f"Area: {metrics['area']} pixels")
    logger.debug(f"Circularity: {metrics['circularity']:.3f}")
    logger.debug(f"Deformability: {metrics['deformability']:.3f}")
    logger.debug(f"Perimeter: {metrics['perimeter']:.2f} pixels")
    
    # Shape metrics
    logger.debug(f"Area ratio: {metrics['area_ratio']:.3f}")
    logger.debug(f"Convex hull area: {metrics['convex_hull_area']} pixels")
    logger.debug(f"Aspect ratio: {metrics['aspect_ratio']:.3f}")
    
    # Size and position metrics
    logger.debug(f"Bounding box: x({metrics['min_x']}, {metrics['max_x']}), y({metrics['min_y']}, {metrics['max_y']})")
    logger.debug(f"Size: {metrics['mask_x_length']}x{metrics['mask_y_length']} pixels")
    
    # Brightness metrics
    logger.debug(f"Mean brightness: {metrics['mean_brightness']:.2f}")
    logger.debug(f"Brightness std: {metrics['brightness_std']:.2f}") 