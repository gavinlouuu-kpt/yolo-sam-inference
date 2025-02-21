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
    circularity = (2 * np.sqrt(np.pi * area)) / perimeter if perimeter > 0 else 0
    
    # Calculate convex hull area
    convex_hull_area = props.convex_area

    area_ratio = convex_hull_area / area
    
    # Calculate deformability 
    deformability = 1 - circularity 
    
    # Calculate brightness metrics (convert RGB to grayscale)
    brightness_image = np.mean(image, axis=2)  # Shape will be (H, W)
    mask_brightness = brightness_image[mask]
    mean_brightness = np.mean(mask_brightness) if mask_brightness.size > 0 else 0
    brightness_std = np.std(mask_brightness) if mask_brightness.size > 0 else 0

    # Calculate aspect ratio
    min_x, min_y, max_x, max_y = props.bbox
    aspect_ratio = (max_x - min_x) / (max_y - min_y) if (max_x - min_x) > 0 and (max_y - min_y) > 0 else 0
    mask_x_length = max_x - min_x
    mask_y_length = max_y - min_y

    return {
        "deformability": float(deformability),
        "area": int(area),
        "area_ratio": float(area_ratio),
        "circularity": float(circularity),
        "convex_hull_area": int(convex_hull_area),
        "mask_x_length": int(mask_x_length),
        "mask_y_length": int(mask_y_length),
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
        "mean_brightness": float(mean_brightness),
        "brightness_std": float(brightness_std),
        "perimeter": float(perimeter),
        "aspect_ratio": float(aspect_ratio),
    } 