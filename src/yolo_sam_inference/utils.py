import numpy as np
from skimage import measure
from typing import Dict, Any

def calculate_metrics(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculate various metrics for a segmented cell.
    
    Args:
        image: Original RGB image (H, W, 3)
        mask: Binary mask of the cell (H, W)
        
    Returns:
        Dictionary containing various metrics
    """
    # Ensure mask is 2D boolean array
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    
    # Ensure image and mask have matching dimensions
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
    
    # Calculate basic properties
    props = measure.regionprops(mask.astype(int))[0]
    
    # Calculate area
    area = props.area
    
    # Calculate circularity
    perimeter = props.perimeter
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Calculate convex hull area
    convex_hull_area = props.convex_area
    
    # Calculate deformability (ratio of actual area to convex hull area)
    deformability = area / convex_hull_area if convex_hull_area > 0 else 0
    
    # Calculate intensity metrics (convert RGB to grayscale)
    intensity_image = np.mean(image, axis=2)  # Shape will be (H, W)
    mask_intensity = intensity_image[mask]
    mean_intensity = np.mean(mask_intensity) if mask_intensity.size > 0 else 0
    
    return {
        "area": int(area),
        "circularity": float(circularity),
        "deformability": float(deformability),
        "convex_hull_area": int(convex_hull_area),
        "mean_intensity": float(mean_intensity),
        "perimeter": float(perimeter)
    } 