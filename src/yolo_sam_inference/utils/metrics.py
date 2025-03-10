import numpy as np
from skimage import measure, draw
from typing import Dict, Any
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Calculate various metrics for a segmented cell.
    
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

    # Get mask contours and calculate convex hull
    contours = measure.find_contours(mask.astype(int), 0.5)
    if len(contours) > 0:
        # Use the largest contour
        contour = contours[0]
        try:
            hull = ConvexHull(contour)
            # Get the vertices of the convex hull in order
            convex_hull_coords = contour[hull.vertices]
            # Add the first point at the end to close the polygon
            convex_hull_coords = np.vstack((convex_hull_coords, convex_hull_coords[0]))
            # Create convex hull mask
            convex_hull_mask = np.zeros_like(mask, dtype=bool)
            # Convert coordinates to the format expected by polygon2mask
            polygon_coords = np.column_stack((convex_hull_coords[:, 0], convex_hull_coords[:, 1]))
            # Use polygon2mask to fill the interior of the polygon
            convex_hull_mask = draw.polygon2mask(mask.shape, polygon_coords)
            # Calculate convex hull properties
            convex_props = measure.regionprops(convex_hull_mask.astype(int))[0]
            
            # Log successful convex hull calculation
            logger.debug(f"Convex hull calculated successfully. Area: {convex_props.area}, Perimeter: {convex_props.perimeter}")
        except Exception as e:
            # If convex hull calculation fails, use empty array and set properties to 0
            logger.warning(f"Convex hull calculation failed: {str(e)}")
            convex_hull_coords = np.array([])
            convex_props = None
    else:
        convex_hull_coords = np.array([])
        convex_props = None

    # Calculate area
    area = props.area
    
    # Calculate perimeter
    perimeter = props.perimeter

    # Calculate convex hull area and perimeter
    convex_hull_area = convex_props.area if convex_props else 0
    convex_hull_perimeter = convex_props.perimeter if convex_props else 0
    
    # Calculate area ratio
    area_ratio = convex_hull_area / area if area > 0 else 0
    
    # Calculate circularity
    circularity = (2 * np.sqrt(np.pi * convex_hull_area)) / convex_hull_perimeter if convex_hull_perimeter > 0 else 0 #DO NOT CHANGE THIS, USE AREA FROM CONVEX HULL
    
    # Calculate deformability 
    deformability = 1 - circularity 
    
    # Calculate brightness metrics (convert RGB to grayscale)
    brightness_image = np.mean(image, axis=2)  # Shape will be (H, W)
    
    # Calculate center region mask
    proportional_factor = 0.1  # Define the proportional factor
    center_radius = int(min(mask.shape) * proportional_factor)  # Define the radius of the center region
    center_region_mask = np.zeros_like(mask, dtype=bool)
    center_x, center_y = props.centroid
    rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
    center_region_mask = (rr - center_x)**2 + (cc - center_y)**2 <= center_radius**2
    
    # Calculate brightness in the center region
    center_brightness = brightness_image[center_region_mask]
    mean_brightness = np.mean(center_brightness) if center_brightness.size > 0 else 0
    brightness_std = np.std(center_brightness) if center_brightness.size > 0 else 0

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
        "convex_hull_perimeter": float(convex_hull_perimeter),
    }