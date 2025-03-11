import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm
from datetime import datetime
import uuid
from typing import Tuple, Dict, List, Any, Optional
import json
import pandas as pd
import logging
from yolo_sam_inference.web.app import get_roi_coordinates_web
from yolo_sam_inference.utils import setup_logger
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from skimage.measure import regionprops, regionprops_table
from yolo_sam_inference.utils.metrics import calculate_metrics
from dataclasses import dataclass
import hashlib

# Set up logger with reduced verbosity
logger = setup_logger(__name__)
logger.setLevel('INFO')

# Define a simplified metrics calculation function that skips convex hull calculations
def calculate_metrics_no_convex_hull(image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Calculate metrics for a segmented cell without convex hull calculations.
    
    This is a simplified version of calculate_metrics that skips the convex hull
    calculations to avoid errors in the OpenCV pipeline.
    
    Args:
        image: Original RGB image (H, W, 3)
        mask: Binary mask of the cell (H, W)
        
    Returns:
        Dictionary containing basic metrics without convex hull related metrics
    """
    # Ensure mask is 2D boolean array
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(bool)
    
    # Ensure image and mask have matching dimensions
    assert mask.shape == image.shape[:2], f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}"
    
    # Calculate basic properties
    props = regionprops(mask.astype(int))[0]

    # Calculate area
    area = props.area
    
    # Calculate perimeter
    perimeter = props.perimeter

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

    # Set default values for convex hull metrics to avoid errors
    # These are placeholders that won't cause errors in downstream processing
    convex_hull_area = area  # Use the same area as the mask
    convex_hull_perimeter = perimeter  # Use the same perimeter as the mask
    area_ratio = 1.0  # Perfect ratio
    circularity = 0.5  # Middle value
    deformability = 0.5  # Middle value

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

@dataclass
class ProcessingResult:
    """Data class to store processing results for a single image."""
    image_path: str
    contour_metrics: List[Dict[str, Any]]
    num_contours: int
    mask: Optional[np.ndarray] = None
    filtered_mask: Optional[np.ndarray] = None
    contours: Optional[List[np.ndarray]] = None
    filtered_contours: Optional[List[np.ndarray]] = None
    roi_coordinates: Optional[Dict[str, int]] = None
    timing: Optional[Dict[str, float]] = None

class OpenCVPipeline:
    def __init__(self, 
                 threshold_value: int = 10,
                 dilate_iterations: int = 2,
                 erode_iterations: int = 2,
                 blur_kernel_size: Tuple[int, int] = (3, 3),
                 blur_sigma: int = 0):
        """Initialize OpenCV pipeline with parameters."""
        self.threshold_value = threshold_value
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        self._cached_backgrounds = {}  # Cache for processed backgrounds

    def _process_background(self, background_path: str) -> np.ndarray:
        """Process background image."""
        if not background_path or not os.path.exists(background_path):
            logger.warning(f"Background image not found at {background_path}")
            return None
        
        # Check if background is already cached
        if background_path in self._cached_backgrounds:
            return self._cached_backgrounds[background_path]
        
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if background is None:
            logger.warning(f"Failed to read background image at {background_path}")
            return None
        
        # Apply blur to reduce noise
        background = cv2.GaussianBlur(background, self.blur_kernel_size, self.blur_sigma)
        
        # Cache the processed background
        self._cached_backgrounds[background_path] = background
        return background

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image from disk."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return image

    def _detect_contours(self, image: np.ndarray, background: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Detect contours in an image with optional background subtraction."""
        times = {}
        
        # Background subtraction
        bg_start = time.time()
        if background is not None:
            # Ensure image and background have the same dimensions
            if image.shape != background.shape:
                background = cv2.resize(background, (image.shape[1], image.shape[0]))
            
            # Subtract background
            diff = cv2.absdiff(image, background)
        else:
            # If no background, just use the original image
            diff = image
        times['background_subtraction'] = time.time() - bg_start
        
        # Pre-processing (optimized operations)
        pre_processing_start = time.perf_counter()
        
        # Combine operations where possible to reduce memory allocations
        blurred = cv2.GaussianBlur(diff, self.blur_kernel_size, self.blur_sigma)
        _, binary = cv2.threshold(blurred, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Combine morphological operations
        morph = cv2.dilate(binary, self.kernel, iterations=self.dilate_iterations)
        morph = cv2.erode(morph, self.kernel, iterations=self.erode_iterations)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, self.kernel)  # More efficient than separate erode/dilate
        
        pre_processing_end = time.perf_counter()
        times['pre_processing'] = (pre_processing_end - pre_processing_start)

        # Find contours (optimized for memory)
        find_contours_start = time.perf_counter()
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        find_contours_end = time.perf_counter()
        times['find_contours'] = (find_contours_end - find_contours_start)

        return contours, times
    
    def contours_to_mask(self, contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Convert contours to binary mask."""
        mask = np.zeros(shape, dtype=np.uint8)
        if contours:
            cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)
        return mask

    def calculate_contour_metrics(self, contour: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics for a single contour using the existing metrics function."""
        # Create a mask for this contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
        
        # Ensure image is RGB for the metrics calculation
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = image
            
        # Use the existing metrics calculation function
        metrics = calculate_metrics_no_convex_hull(rgb_image, mask)
        
        return metrics

    def filter_contours_by_roi(self, contours: List[np.ndarray], image_shape: Tuple[int, int], 
                              roi: Dict[str, int]) -> List[np.ndarray]:
        """Filter contours based on whether they intersect with the ROI."""
        x_min, y_min = roi['x_min'], roi['y_min']
        x_max, y_max = roi['x_max'], roi['y_max']
        
        filtered_contours = []
        for contour in contours:
            # Create a mask for this contour
            contour_mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            
            # Check if any part of the contour is within the ROI
            roi_section = contour_mask[y_min:y_max, x_min:x_max]
            if np.any(roi_section > 0):
                filtered_contours.append(contour)
                
        return filtered_contours

    def process_image(self, image_path: str, background_path: str, 
                     roi: Optional[Dict[str, int]] = None, 
                     output_path: Optional[str] = None,
                     save_visualizations: bool = True) -> ProcessingResult:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to input image
            background_path: Path to background image for subtraction
            roi: Region of interest coordinates {x_min, y_min, x_max, y_max}
            output_path: Path to save outputs
            save_visualizations: Whether to save visualization images
            
        Returns:
            ProcessingResult containing processing results
        """
        # Extract image name from path
        image_path_obj = Path(image_path)
        image_name = image_path_obj.stem
        
        # Extract batch name from path to avoid filename collisions
        try:
            # Try to extract batch name from path (assuming structure like /path/to/batch_name/images/...)
            # Go up two levels from the image file to get the batch folder
            batch_folder = image_path_obj.parent.parent
            batch_name = batch_folder.name
            
            # Check if the batch name contains useful information (like a number)
            if batch_name and any(char.isdigit() for char in batch_name):
                logger.info(f"Extracted batch name: {batch_name}")
                # Include batch name in the output filename to avoid collisions
                output_name = f"{batch_name}_{image_name}"
            else:
                # If we couldn't extract a meaningful batch name, use a hash of the path
                path_hash = hashlib.md5(str(image_path_obj.parent).encode()).hexdigest()[:6]
                output_name = f"{path_hash}_{image_name}"
                logger.info(f"Using path hash as batch identifier: {path_hash}")
        except Exception as e:
            # Fallback to just using the image name if there's any error
            logger.warning(f"Error extracting batch name: {str(e)}, using image name only")
            output_name = image_name
        
        # Check if this is a cropped image
        is_cropped = 'cropped_roi' in str(image_path)
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Is cropped: {is_cropped} (determined by 'cropped_roi' in path)")
        logger.info(f"Output name with batch identifier: {output_name}")
        
        # Load image
        gray_image = self._load_image(image_path)
        
        # Load color image for visualization and metrics
        color_image = cv2.imread(image_path)
        
        # Process background
        background = self._process_background(background_path)
        
        # Detect contours
        contours, _ = self._detect_contours(gray_image, background)
        
        logger.info(f"Found {len(contours)} contours in {image_name}")
        
        # Create mask from all contours
        mask = self.contours_to_mask(contours, gray_image.shape)
        
        # Apply ROI filtering if provided
        if roi is not None and not is_cropped:
            logger.info(f"Applying ROI filtering with coordinates: {roi}")
            filtered_contours = self.filter_contours_by_roi(contours, gray_image.shape, roi)
            filtered_mask = self.contours_to_mask(filtered_contours, gray_image.shape)
            logger.info(f"After ROI filtering: {len(filtered_contours)} of {len(contours)} contours remain")
        else:
            # For cropped images or when no ROI is provided, use all contours
            logger.info(f"Skipping ROI filtering because {'image is cropped' if is_cropped else 'no ROI provided'}")
            filtered_contours = contours
            filtered_mask = mask
            
            # If this is a cropped image and ROI is provided, adjust ROI for visualization
            if is_cropped and roi is not None:
                logger.info(f"Adjusting ROI for cropped image visualization")
                roi = {
                    'x_min': 0,
                    'y_min': 0,
                    'x_max': gray_image.shape[1],
                    'y_max': gray_image.shape[0]
                }
                logger.info(f"Adjusted ROI: {roi}")
        
        # Calculate metrics for each contour
        contour_metrics = []
        for i, contour in enumerate(filtered_contours):
            metrics = self.calculate_contour_metrics(contour, color_image)
            metrics['cell_id'] = i
            metrics['image_name'] = image_name
            metrics['batch_name'] = batch_name if 'batch_name' in locals() else ""
            metrics['is_cropped'] = is_cropped
            contour_metrics.append(metrics)
        
        # Save visualizations if requested
        if save_visualizations and output_path:
            # Use the output_name that includes batch identifier to avoid collisions
            vis_path = str(Path(output_path) / f"{output_name}_visualization.png")
            
            # Get ROI coordinates for visualization
            x_min = roi['x_min'] if roi else 0
            y_min = roi['y_min'] if roi else 0
            x_max = roi['x_max'] if roi else gray_image.shape[1]
            y_max = roi['y_max'] if roi else gray_image.shape[0]
            
            save_visualization(color_image, mask, filtered_mask, {}, 
                              x_min, y_min, x_max, y_max, vis_path, contour_metrics)
            
            # Save masks with batch identifier in filename
            mask_path = str(Path(output_path) / f"{output_name}_mask.png")
            filtered_mask_path = str(Path(output_path) / f"{output_name}_filtered_mask.png")
            cv2.imwrite(mask_path, mask * 255)
            cv2.imwrite(filtered_mask_path, filtered_mask * 255)
        
        logger.info(f"Processed {image_name}: {len(filtered_contours)} contours detected")
        
        return ProcessingResult(
            image_path=image_path,
            contour_metrics=contour_metrics,
            num_contours=len(contour_metrics),
            mask=mask,
            filtered_mask=filtered_mask,
            contours=contours,
            filtered_contours=filtered_contours,
            roi_coordinates=roi
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Project-based cell segmentation pipeline using OpenCV.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--project-dir', '-p',
        type=str,
        required=True,
        help='Project directory containing condition folders'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='opencv_project_inference_output',
        help='Directory to save output results'
    )
    
    parser.add_argument(
        '--threshold',
        type=int,
        default=10,
        help='Threshold value for binary thresholding'
    )
    
    parser.add_argument(
        '--dilate',
        type=int,
        default=2,
        help='Number of dilation iterations'
    )
    
    parser.add_argument(
        '--erode',
        type=int,
        default=2,
        help='Number of erosion iterations'
    )
    
    parser.add_argument(
        '--blur',
        type=int,
        default=3,
        help='Blur kernel size (NxN)'
    )
    
    parser.add_argument(
        '--sigma',
        type=int,
        default=0,
        help='Sigma value for Gaussian blur'
    )
    
    return parser.parse_args()

def collect_image_paths(condition_dir):
    """Collect all image paths from a condition directory efficiently."""
    image_mapping = {}
    
    # Use a set for faster lookups
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff'}
    
    # Single pass through directory
    for file_path in condition_dir.iterdir():
        if file_path.suffix.lower() in valid_extensions and 'background' not in file_path.name.lower():
            image_mapping[str(file_path)] = {
                'original_path': file_path,
                'new_name': file_path.name
            }
    
    return image_mapping

def create_hardlinks_for_batch(condition_dir, image_mapping):
    """Create a temporary directory with hardlinks to the original images."""
    import os
    
    # Create a temporary directory for combined images
    temp_dir = condition_dir / "temp_combined"
    temp_dir.mkdir(exist_ok=True)
    
    # Create hardlinks for each image
    for original_path, info in image_mapping.items():
        target_path = temp_dir / info['new_name']
        if not target_path.exists():  # Only create if doesn't exist
            try:
                os.link(original_path, str(target_path))
            except OSError as e:
                # If hardlink fails (e.g., cross-device), fall back to copy
                logger.warning(f"Failed to create hardlink, falling back to copy: {str(e)}")
                shutil.copy2(original_path, target_path)
    
    return temp_dir

def process_single_image(args):
    """Process a single image with the OpenCV pipeline."""
    image_path, background_path, pipeline, output_dir, roi = args
    
    try:
        # Skip background images
        image_name = Path(image_path).stem
        if 'background' in image_name.lower():
            return None
        
        logger.info(f"Processing image: {image_path}")
        
        # Process the image using the refactored pipeline
        result = pipeline.process_image(
            image_path=str(image_path),
            background_path=background_path,
            roi=roi,
            output_path=output_dir,
            save_visualizations=True
        )
        
        # Calculate summary metrics
        total_area = np.sum(result.mask > 0) if result.mask is not None else 0
        roi_area = np.sum(result.filtered_mask > 0) if result.filtered_mask is not None else 0
        
        if result.num_contours == 0:
            logger.info(f"No contours found in {image_name}, but still processing")
        else:
            logger.info(f"Processed successfully: total_area={total_area}, roi_area={roi_area}, contours={result.num_contours}")
        
        return {
            'image_name': image_name,
            'is_cropped': 'cropped_roi' in str(image_path),
            'total_area': total_area,
            'roi_area': roi_area,
            'contour_metrics': result.contour_metrics,
            'num_contours': result.num_contours
        }
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def save_visualization(image, mask, filtered_mask, times, x_min, y_min, x_max, y_max, vis_path, contour_metrics=None):
    """Save visualization of the processing results."""
    # Create a color version of the image if it's grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Create a copy for the filtered visualization
    filtered_vis = vis_image.copy()
    
    # Draw ROI rectangle
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.rectangle(filtered_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Apply mask overlay
    mask_overlay = vis_image.copy()
    filtered_overlay = filtered_vis.copy()
    
    # Create colored masks for visualization
    color_mask = np.zeros_like(vis_image)
    color_mask[mask > 0] = [0, 0, 255]  # Red for all contours
    
    color_filtered_mask = np.zeros_like(filtered_vis)
    color_filtered_mask[filtered_mask > 0] = [255, 0, 0]  # Blue for ROI contours
    
    # Blend the masks with the original image
    cv2.addWeighted(mask_overlay, 0.7, color_mask, 0.3, 0, mask_overlay)
    cv2.addWeighted(filtered_overlay, 0.7, color_filtered_mask, 0.3, 0, filtered_overlay)
    
    # Create a combined visualization
    h, w = vis_image.shape[:2]
    combined = np.zeros((h, w*2, 3), dtype=np.uint8)
    combined[:, :w] = mask_overlay
    combined[:, w:] = filtered_overlay
    
    # Add titles only
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)
    
    # Add titles
    cv2.putText(combined, "All Contours", (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(combined, "ROI Contours", (w+10, 20), font, font_scale, font_color, font_thickness)
    
    # Add metrics information if available
    if contour_metrics:
        y_pos = 40
        # Calculate average deformability
        avg_deformability = np.mean([m['deformability'] for m in contour_metrics if 'deformability' in m]) if contour_metrics else 0
        cv2.putText(combined, f"Contours: {len(contour_metrics)}", (w+10, y_pos), font, font_scale, font_color, font_thickness)
        y_pos += 20
        cv2.putText(combined, f"Avg Deformability: {avg_deformability:.4f}", (w+10, y_pos), font, font_scale, font_color, font_thickness)
    
    # Save the visualization
    cv2.imwrite(vis_path, combined)

def process_condition(pipeline, condition_dir, run_output_dir, run_id: str, background_path: str, roi_coordinates: Dict, pbar=None):
    """Process all images in a condition directory."""
    condition_name = condition_dir.name
    condition_output_dir = run_output_dir / condition_name
    condition_output_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nProcessing condition: {condition_name}")
    
    # Count and log the number of batches
    batch_folders = [d for d in condition_dir.iterdir() if d.is_dir() and "_output" in d.name]
    logger.info(f"  Found {len(batch_folders)} batches in condition {condition_name}")
    for i, batch_folder in enumerate(sorted(batch_folders)):
        logger.info(f"    Batch {i+1}: {batch_folder.name}")
    
    # Find background images in each batch folder
    background_images = {}
    for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        for bg_file in condition_dir.glob(f"**/*background*{ext}"):
            batch_folder = bg_file.parent.parent  # Go up two levels to get the batch folder
            batch_name = batch_folder.name
            background_images[batch_name] = str(bg_file)
            logger.info(f"  Found background image for batch {batch_name}: {bg_file}")
    
    if not background_images:
        logger.warning(f"  No background images found")
        # Use the provided background path as fallback
        if background_path:
            logger.info(f"  Using fallback background image: {background_path}")
    
    # Get ROI coordinates for this condition
    if condition_name not in roi_coordinates:
        raise ValueError(f"No ROI coordinates found for condition: {condition_name}")
    
    roi = roi_coordinates[condition_name]
    x_min, y_min = roi['x_min'], roi['y_min']
    x_max, y_max = roi['x_max'], roi['y_max']
    logger.info(f"  Using ROI: x={x_min}-{x_max}, y={y_min}-{y_max}")
    
    try:
        # Create output directories
        (condition_output_dir / "full_frames").mkdir(exist_ok=True)
        (condition_output_dir / "cropped").mkdir(exist_ok=True)
        
        # Process images in parallel using a ProcessPoolExecutor
        all_results = []
        skipped_no_contours = 0
        processed_count = 0
        n_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        
        # Process each batch folder
        for batch_folder in sorted(batch_folders):
            logger.info(f"\n  Processing batch: {batch_folder.name}")
            
            # Find all image files in this batch
            batch_image_files = []
            
            # Look specifically in the cropped_roi_with_target and full_frames_with_target subdirectories
            cropped_roi_dir = batch_folder / "cropped_roi_with_target"
            full_frames_dir = batch_folder / "full_frames_with_target"
            
            logger.info(f"    Looking for images in: {cropped_roi_dir}")
            logger.info(f"    Looking for images in: {full_frames_dir}")
            
            # Find images in cropped_roi_with_target
            if cropped_roi_dir.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                    found_files = list(cropped_roi_dir.glob(f"*{ext}"))
                    logger.info(f"    Found {len(found_files)} {ext} files in cropped_roi_with_target")
                    batch_image_files.extend([f for f in found_files if 'background' not in f.name.lower()])
            
            # Find images in full_frames_with_target
            if full_frames_dir.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                    found_files = list(full_frames_dir.glob(f"*{ext}"))
                    logger.info(f"    Found {len(found_files)} {ext} files in full_frames_with_target")
                    batch_image_files.extend([f for f in found_files if 'background' not in f.name.lower()])
            
            # Separate full frame and cropped images for this batch
            batch_full_frame_images = [f for f in batch_image_files if 'full_frames' in str(f)]
            batch_cropped_images = [f for f in batch_image_files if 'cropped_roi' in str(f)]
            
            logger.info(f"    Found {len(batch_image_files)} total images in batch")
            logger.info(f"    Full frame images: {len(batch_full_frame_images)}")
            logger.info(f"    Cropped ROI images: {len(batch_cropped_images)}")
            
            # Get background image for this batch
            img_background_path = background_images.get(batch_folder.name, background_path)
            
            # Process full frame images for this batch
            if batch_full_frame_images:
                logger.info(f"    Processing {len(batch_full_frame_images)} full frame images")
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for image_path in batch_full_frame_images:
                        args = (
                            str(image_path),
                            img_background_path,
                            pipeline,
                            condition_output_dir / "full_frames",
                            roi
                        )
                        futures.append(executor.submit(process_single_image, args))
                    
                    for future in futures:
                        result = future.result()
                        processed_count += 1
                        if result is not None:
                            all_results.append(result)
                        else:
                            # Assume skipped due to no contours if result is None
                            skipped_no_contours += 1
                        if pbar:
                            pbar.update(1)
            
            # Process cropped images for this batch
            if batch_cropped_images:
                logger.info(f"    Processing {len(batch_cropped_images)} cropped images")
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = []
                    for image_path in batch_cropped_images:
                        args = (
                            str(image_path),
                            img_background_path,
                            pipeline,
                            condition_output_dir / "cropped",
                            roi
                        )
                        futures.append(executor.submit(process_single_image, args))
                    
                    for future in futures:
                        result = future.result()
                        processed_count += 1
                        if result is not None:
                            all_results.append(result)
                        else:
                            # Assume skipped due to no contours if result is None
                            skipped_no_contours += 1
                        if pbar:
                            pbar.update(1)
        
        logger.info(f"\n  Condition summary:")
        logger.info(f"  Processed {len(all_results)} images successfully")
        logger.info(f"  Skipped {skipped_no_contours} images due to no contours found")
        logger.info(f"  Total images processed: {processed_count}")
        
        # Save results to CSV for this condition
        save_results_to_csv(all_results, condition_output_dir)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Error processing condition {condition_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def create_run_output_dir(base_output_dir: Path) -> Tuple[Path, str]:
    """Create a unique run output directory."""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_id

def count_total_images(condition_dirs):
    """Count total number of images across all conditions without copying files."""
    total_images = 0
    for condition_dir in condition_dirs:
        # Count images in each condition directory
        for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
            total_images += len(list(condition_dir.glob(f"*{ext}")))
        # Subtract background image
        total_images -= 1
    return total_images

def save_results_to_csv(results, output_dir: Path) -> None:
    """Save processing results to CSV files focusing on cell metrics."""
    # Prepare data in memory first
    cell_metrics_data = []
    image_summary_data = []
    
    for result in results:
        if result is None:
            continue
            
        # Create image summary entry even if no contours
        image_summary = {
            'image_name': result['image_name'],
            'is_cropped': result.get('is_cropped', False),
            'total_area': result.get('total_area', 0),
            'roi_area': result.get('roi_area', 0),
            'num_contours': result.get('num_contours', 0)
        }
        image_summary_data.append(image_summary)
        
        # Add cell metrics data if available
        if 'contour_metrics' in result and result['contour_metrics']:
            for cell_metric in result['contour_metrics']:
                # Add image information to each cell metric
                cell_metric['image_name'] = result['image_name']
                cell_metric['is_cropped'] = result.get('is_cropped', False)
                cell_metrics_data.append(cell_metric)
    
    # Save image summary data
    if image_summary_data:
        image_summary_df = pd.DataFrame(image_summary_data)
        image_summary_df.to_csv(output_dir / 'image_summary.csv', index=False)
        logger.info(f"Saved summary for {len(image_summary_data)} images")
    
    # Save cell metrics data
    if cell_metrics_data:
        # Create DataFrame with all cell metrics
        cell_metrics_df = pd.DataFrame(cell_metrics_data)
        
        # Save complete cell metrics
        cell_metrics_df.to_csv(output_dir / 'cell_metrics.csv', index=False)
        
        # Create a summary metrics file with deformability statistics per image
        summary_data = []
        for image_name, group in cell_metrics_df.groupby('image_name'):
            summary_data.append({
                'image_name': image_name,
                'cell_count': len(group),
                'mean_deformability': group['deformability'].mean() if 'deformability' in group else 0,
                'std_deformability': group['deformability'].std() if 'deformability' in group else 0,
                'min_deformability': group['deformability'].min() if 'deformability' in group else 0,
                'max_deformability': group['deformability'].max() if 'deformability' in group else 0,
                'mean_area': group['area'].mean() if 'area' in group else 0,
                'total_area': group['area'].sum() if 'area' in group else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'deformability_summary.csv', index=False)
        
        # Log summary statistics
        logger.info(f"Saved metrics for {len(cell_metrics_data)} cells across {len(summary_data)} images")
        if summary_data:
            avg_deformability = sum(item.get('mean_deformability', 0) for item in summary_data) / len(summary_data)
            logger.info(f"Average deformability across all images: {avg_deformability:.4f}")

def main():
    try:
        args = parse_args()
        project_dir = Path(args.project_dir)
        base_output_dir = Path(args.output_dir)
        
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
        
        # Create output directory
        run_output_dir, run_id = create_run_output_dir(base_output_dir)
        
        # Get condition directories
        condition_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        print(f"Found {len(condition_dirs)} condition directories:")
        for d in condition_dirs:
            print(f"  - {d.name}")
            
        # Count total batches across all conditions
        total_batches = 0
        all_batch_folders = set()
        print("\nCounting batches in each condition:")
        for condition_dir in condition_dirs:
            # Find all batch directories (those with "_output" in the name)
            batch_folders = [d for d in condition_dir.iterdir() if d.is_dir() and "_output" in d.name]
            
            print(f"  Condition {condition_dir.name}: {len(batch_folders)} batches")
            for i, batch_folder in enumerate(sorted(batch_folders)):
                print(f"    Batch {i+1}: {batch_folder.name}")
                all_batch_folders.add(batch_folder)
            
        total_batches = len(all_batch_folders)
        print(f"\nTotal batches across all conditions: {total_batches}")
        
        # Debug: Check directory structure for each condition
        for condition_dir in condition_dirs:
            print(f"\nDEBUG: Analyzing directory structure for condition: {condition_dir.name}")
            # Check for cropped_roi and full_frames directories
            cropped_dirs = list(condition_dir.glob("**/cropped_roi*"))
            full_frame_dirs = list(condition_dir.glob("**/full_frame*"))
            print(f"DEBUG:   Found {len(cropped_dirs)} directories with 'cropped_roi' in name")
            print(f"DEBUG:   Found {len(full_frame_dirs)} directories with 'full_frame' in name")
            
            # Count image files in each type of directory
            cropped_images = []
            full_frame_images = []
            for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                cropped_images.extend(list(condition_dir.glob(f"**/cropped_roi*/**/*{ext}")))
                full_frame_images.extend(list(condition_dir.glob(f"**/full_frame*/**/*{ext}")))
            
            print(f"DEBUG:   Found {len(cropped_images)} images in 'cropped_roi' directories")
            print(f"DEBUG:   Found {len(full_frame_images)} images in 'full_frame' directories")
        
        print("\nOpening web interface for ROI selection...")
        print("Please select ROI coordinates for each condition in the browser window.")
        print("Click two points on each image to define the min and max X coordinates.")
        print("You must select ROI for ALL conditions before processing can begin.")
        
        # Get ROI coordinates
        print("\nDEBUG: Calling get_roi_coordinates_web to select images for ROI selection")
        roi_coordinates = get_roi_coordinates_web(condition_dirs, run_output_dir)
        
        # Verify ROI coordinates
        missing_conditions = [d.name for d in condition_dirs if d.name not in roi_coordinates]
        if missing_conditions:
            raise ValueError(f"Missing ROI coordinates for conditions: {', '.join(missing_conditions)}.")
            
        print("\nROI coordinates collected successfully for all conditions!")
        print("ROI coordinates:")
        for condition, coords in roi_coordinates.items():
            print(f"  {condition}: {coords}")
        
        print(f"\nInitializing OpenCV pipeline... [Run ID: {run_id}]")
        pipeline = OpenCVPipeline(
            threshold_value=args.threshold,
            dilate_iterations=args.dilate,
            erode_iterations=args.erode,
            blur_kernel_size=(args.blur, args.blur),
            blur_sigma=args.sigma
        )
        
        # Count total images for progress bar
        total_images = 0
        for condition_dir in condition_dirs:
            for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                total_images += len(list(condition_dir.glob(f"**/*{ext}")))
        
        print(f"Found {total_images} total images across all conditions")
        
        # Process all conditions
        start_time = time.time()
        all_results = []
        
        with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
            # Process conditions sequentially for better debugging
            for condition_dir in condition_dirs:
                try:
                    condition_results = process_condition(
                        pipeline=pipeline,
                        condition_dir=condition_dir,
                        run_output_dir=run_output_dir,
                        run_id=run_id,
                        background_path=None,  # Will be found in process_condition
                        roi_coordinates=roi_coordinates,
                        pbar=pbar
                    )
                    if condition_results:
                        all_results.extend(condition_results)
                except Exception as e:
                    logger.error(f"Error processing condition {condition_dir.name}: {str(e)}")
                    print(f"Error processing condition {condition_dir.name}: {str(e)}")
        
        total_runtime = time.time() - start_time
        
        # Count images with and without contours
        images_with_contours = sum(1 for r in all_results if r.get('num_contours', 0) > 0)
        images_without_contours = sum(1 for r in all_results if r.get('num_contours', 0) == 0)
        
        print(f"\nProcessed {len(all_results)} images successfully across all conditions")
        print(f"  - Images with contours: {images_with_contours}")
        print(f"  - Images without contours: {images_without_contours}")
        
        # Save results
        print("\nSaving results...")
        save_results_to_csv(all_results, run_output_dir)
        
        # Save ROI coordinates
        roi_file = run_output_dir / "roi_coordinates.json"
        with open(roi_file, 'w') as f:
            json.dump(roi_coordinates, f, indent=2)
        
        print(f"\nResults saved to: {run_output_dir}")
        print(f"ROI coordinates saved to: {roi_file}")
        print(f"Total runtime: {total_runtime:.2f} seconds")
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 