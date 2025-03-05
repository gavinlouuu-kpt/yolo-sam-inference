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
from typing import Tuple, Dict, List, Any
import json
import pandas as pd
import logging
from yolo_sam_inference.web.app import get_roi_coordinates_web
from yolo_sam_inference.utils import setup_logger
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from skimage.measure import regionprops, regionprops_table
from yolo_sam_inference.utils.metrics import calculate_metrics

# Set up logger with reduced verbosity
logger = setup_logger(__name__)
logger.setLevel('INFO')

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
            print(f"WARNING: Background image not found at {background_path}")
            return None
        
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        if background is None:
            print(f"WARNING: Failed to read background image at {background_path}")
            return None
        
        # Apply blur to reduce noise
        background = cv2.GaussianBlur(background, self.blur_kernel_size, self.blur_sigma)
        return background

    def process_image(self, image_path: str, background_path: str) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Process a single image and return contours and timing information."""
        start_time = time.time()
        
        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Process background
        background = self._process_background(background_path)
        
        # Timing information
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
        pre_processing_time = (pre_processing_end - pre_processing_start) * 1000

        # Find contours (optimized for memory)
        find_contours_start = time.perf_counter()
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        find_contours_end = time.perf_counter()
        find_contours_time = (find_contours_end - find_contours_start) * 1000

        total_end_time = time.perf_counter()
        total_processing_time = (total_end_time - start_time) * 1000

        times = {
            'pre_processing_time': pre_processing_time,
            'find_contours_time': find_contours_time,
            'total_processing_time': total_processing_time
        }

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
        metrics = calculate_metrics(rgb_image, mask)
        
        return metrics

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
        # Extract image name from path
        image_name = Path(image_path).stem
        image_path_str = str(image_path)
        
        # Skip background images
        if 'background' in image_name.lower():
            return None
        
        print(f"Processing image: {image_path}")
        
        # Check if this is a cropped image
        is_cropped = 'cropped_roi' in image_path_str
        
        # Process the image
        contours, times = pipeline.process_image(image_path_str, background_path)
        
        print(f"  Found {len(contours)} contours")
        
        if not contours:
            print(f"  No contours found, skipping")
            return None
        
        # Create mask from contours
        image = cv2.imread(image_path_str)
        if image is None:
            print(f"  Failed to read image for visualization: {image_path}")
            return None
            
        mask = pipeline.contours_to_mask(contours, image.shape[:2])
        
        # Apply ROI filtering
        x_min, y_min = roi['x_min'], roi['y_min']
        x_max, y_max = roi['x_max'], roi['y_max']
        
        # For cropped images, we need to adjust the ROI or use the entire image
        if is_cropped:
            # For cropped images, consider the entire image as the ROI
            print(f"  This is a cropped image, using entire image as ROI")
            filtered_mask = mask.copy()
            # Use image dimensions for visualization
            h, w = image.shape[:2]
            x_min, y_min = 0, 0
            x_max, y_max = w, h
            filtered_contours = contours
        else:
            # Filter contours based on whether they intersect with the ROI
            filtered_contours = []
            for contour in contours:
                # Create a mask for this contour
                contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
                
                # Check if any part of the contour is within the ROI
                roi_section = contour_mask[y_min:y_max, x_min:x_max]
                if np.any(roi_section > 0):
                    filtered_contours.append(contour)
            
            # Create filtered mask from filtered contours
            filtered_mask = pipeline.contours_to_mask(filtered_contours, image.shape[:2])
        
        # Calculate metrics for each contour
        all_contour_metrics = []
        for i, contour in enumerate(filtered_contours):
            metrics = pipeline.calculate_contour_metrics(contour, image)
            metrics['cell_id'] = i
            metrics['image_name'] = image_name
            metrics['is_cropped'] = is_cropped
            all_contour_metrics.append(metrics)
        
        # Save visualization
        vis_path = str(Path(output_dir) / f"{image_name}_visualization.png")
        save_visualization(image, mask, filtered_mask, times, x_min, y_min, x_max, y_max, vis_path, all_contour_metrics)
        
        # Save masks
        mask_path = str(Path(output_dir) / f"{image_name}_mask.png")
        filtered_mask_path = str(Path(output_dir) / f"{image_name}_filtered_mask.png")
        cv2.imwrite(mask_path, mask * 255)
        cv2.imwrite(filtered_mask_path, filtered_mask * 255)
        
        # Calculate metrics
        total_area = np.sum(mask > 0)
        roi_area = np.sum(filtered_mask > 0)
        
        print(f"  Processed successfully: total_area={total_area}, roi_area={roi_area}, contours={len(filtered_contours)}")
        
        return {
            'image_name': image_name,
            'is_cropped': is_cropped,
            'total_area': total_area,
            'roi_area': roi_area,
            'processing_time': times.get('total_processing_time', 0),
            'contour_metrics': all_contour_metrics
        }
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
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
        avg_deformability = np.mean([m['deformability'] for m in contour_metrics]) if contour_metrics else 0
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
    
    print(f"\nProcessing condition: {condition_name}")
    
    # Find all image files in the condition directory (recursively)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        found_files = list(condition_dir.glob(f"**/*{ext}"))
        print(f"  Found {len(found_files)} {ext} files")
        image_files.extend(found_files)
    
    # Filter out background images
    original_count = len(image_files)
    image_files = [f for f in image_files if 'background' not in f.name.lower()]
    print(f"  After filtering out background images: {len(image_files)} of {original_count} files remain")
    
    # Separate full frame and cropped images
    full_frame_images = [f for f in image_files if 'full_frames' in str(f)]
    cropped_images = [f for f in image_files if 'cropped_roi' in str(f)]
    print(f"  Full frame images: {len(full_frame_images)}")
    print(f"  Cropped ROI images: {len(cropped_images)}")
    
    # Find background images in each batch folder
    background_images = {}
    for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        for bg_file in condition_dir.glob(f"**/*background*{ext}"):
            batch_folder = bg_file.parent.parent  # Go up two levels to get the batch folder
            batch_name = batch_folder.name
            background_images[batch_name] = str(bg_file)
            print(f"  Found background image for batch {batch_name}: {bg_file}")
    
    if not background_images:
        print(f"  WARNING: No background images found")
        # Use the provided background path as fallback
        if background_path:
            print(f"  Using fallback background image: {background_path}")
    
    # Get ROI coordinates for this condition
    if condition_name not in roi_coordinates:
        raise ValueError(f"No ROI coordinates found for condition: {condition_name}")
    
    roi = roi_coordinates[condition_name]
    x_min, y_min = roi['x_min'], roi['y_min']
    x_max, y_max = roi['x_max'], roi['y_max']
    print(f"  Using ROI: x={x_min}-{x_max}, y={y_min}-{y_max}")
    
    try:
        # Prepare arguments for parallel processing
        process_args = []
        
        # Add full frame images first
        for image_path in full_frame_images:
            # Determine which background image to use
            batch_folder = image_path.parent.parent  # Go up two levels to get the batch folder
            batch_name = batch_folder.name
            img_background_path = background_images.get(batch_name, background_path)
            
            process_args.append((
                str(image_path),
                img_background_path,
                pipeline,
                condition_output_dir / "full_frames",
                roi
            ))
        
        # Then add cropped images
        for image_path in cropped_images:
            # Determine which background image to use
            batch_folder = image_path.parent.parent  # Go up two levels to get the batch folder
            batch_name = batch_folder.name
            img_background_path = background_images.get(batch_name, background_path)
            
            process_args.append((
                str(image_path),
                img_background_path,
                pipeline,
                condition_output_dir / "cropped",
                roi
            ))
        
        print(f"  Processing {len(process_args)} images")
        
        # Create output directories
        (condition_output_dir / "full_frames").mkdir(exist_ok=True)
        (condition_output_dir / "cropped").mkdir(exist_ok=True)
        
        # Process images in parallel using a ProcessPoolExecutor
        results = []
        n_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for args in process_args:
                futures.append(executor.submit(process_single_image, args))
            
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
                if pbar:
                    pbar.update(1)
        
        # Separate results by image type
        full_frame_results = [r for r in results if not r.get('is_cropped', False)]
        cropped_results = [r for r in results if r.get('is_cropped', False)]
        
        print(f"  Processed {len(full_frame_results)} full frame images successfully")
        print(f"  Processed {len(cropped_results)} cropped images successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing condition {condition_name}: {str(e)}")
        raise

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

def save_results_to_csv(results, output_dir):
    """Save processing results to CSV files focusing on cell metrics."""
    # Prepare data in memory first
    cell_metrics_data = []
    processing_times = []
    
    for result in results:
        if result is None:
            continue
            
        # Add processing time data
        processing_times.append({
            'image_name': result['image_name'],
            'is_cropped': result.get('is_cropped', False),
            'total_area': result['total_area'],
            'roi_area': result['roi_area'],
            'processing_time': result['processing_time']
        })
        
        # Add cell metrics data
        if 'contour_metrics' in result:
            for cell_metric in result['contour_metrics']:
                # Add image information to each cell metric
                cell_metric['image_name'] = result['image_name']
                cell_metric['is_cropped'] = result.get('is_cropped', False)
                cell_metrics_data.append(cell_metric)
    
    # Write data in batches
    chunk_size = 1000
    
    # Save processing times for reference
    if processing_times:
        times_df = pd.DataFrame(processing_times)
        times_df.to_csv(output_dir / 'processing_times.csv', index=False)
    
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
                'mean_deformability': group['deformability'].mean(),
                'std_deformability': group['deformability'].std(),
                'min_deformability': group['deformability'].min(),
                'max_deformability': group['deformability'].max(),
                'mean_area': group['area'].mean(),
                'total_area': group['area'].sum()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / 'deformability_summary.csv', index=False)
        
        # Log summary statistics
        logger.info(f"Saved metrics for {len(cell_metrics_data)} cells across {len(summary_data)} images")
        if summary_data:
            avg_deformability = sum(item['mean_deformability'] for item in summary_data) / len(summary_data)
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
        
        print("\nOpening web interface for ROI selection...")
        print("Please select ROI coordinates for each condition in the browser window.")
        print("Click two points on each image to define the min and max X coordinates.")
        print("You must select ROI for ALL conditions before processing can begin.")
        
        # Get ROI coordinates
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
        
        print(f"\nProcessed {len(all_results)} images successfully across all conditions")
        
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