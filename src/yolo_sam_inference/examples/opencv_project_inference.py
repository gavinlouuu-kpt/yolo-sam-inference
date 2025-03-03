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
from typing import Tuple, Dict, List
import json
import pandas as pd
import logging
from yolo_sam_inference.web.app import get_roi_coordinates_web
from yolo_sam_inference.utils import setup_logger
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

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
        """Process and cache background image."""
        if background_path not in self._cached_backgrounds:
            background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
            if background is None:
                raise ValueError(f"Failed to load background image: {background_path}")
            blurred_bg = cv2.GaussianBlur(background, self.blur_kernel_size, self.blur_sigma)
            self._cached_backgrounds[background_path] = blurred_bg
        return self._cached_backgrounds[background_path]

    def process_image(self, image_path: str, background_path: str) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Process an image to find contours using OpenCV pipeline."""
        # Load and process image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Get processed background
        blurred_bg = self._process_background(background_path)
        
        total_start_time = time.perf_counter()
        
        # Pre-processing (optimized operations)
        pre_processing_start = time.perf_counter()
        
        # Combine operations where possible to reduce memory allocations
        blurred = cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)
        bg_sub = cv2.subtract(blurred, blurred_bg)
        _, binary = cv2.threshold(bg_sub, self.threshold_value, 255, cv2.THRESH_BINARY)
        
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
        total_processing_time = (total_end_time - total_start_time) * 1000

        times = {
            'pre_processing_time': pre_processing_time,
            'find_contours_time': find_contours_time,
            'total_processing_time': total_processing_time
        }

        return contours, times
    
    def contours_to_mask(self, contours: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
        """Convert contours to a binary mask.
        
        Args:
            contours: List of contours
            shape: Shape of the output mask (height, width)
            
        Returns:
            Binary mask
        """
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask

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
        '--dilate-iterations',
        type=int,
        default=2,
        help='Number of dilation iterations'
    )
    
    parser.add_argument(
        '--erode-iterations',
        type=int,
        default=2,
        help='Number of erosion iterations'
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
    """Process a single image with the pipeline."""
    image_path, background_path, pipeline, condition_output_dir, roi = args
    try:
        # Process the image
        contours, times = pipeline.process_image(str(image_path), background_path)
        
        # Convert contours to mask
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = pipeline.contours_to_mask(contours, image.shape)
        
        # Filter contours based on ROI
        x_min, x_max = roi['x_min'], roi['x_max']
        y_min, y_max = roi['y_min'], roi['y_max']
        
        filtered_contours = []
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                if (x_min <= center_x <= x_max) and (y_min <= center_y <= y_max):
                    filtered_contours.append(contour)
                    cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Save visualization using a thread-safe approach
        vis_path = condition_output_dir / f"{Path(image_path).stem}_result.png"
        save_visualization(image, mask, filtered_mask, times, x_min, y_min, x_max, y_max, vis_path)
        
        return {
            'image_path': str(image_path),
            'condition': Path(image_path).parent.name,
            'contours': filtered_contours,
            'mask': filtered_mask,
            'times': times,
            'roi': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        }
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def save_visualization(image, mask, filtered_mask, times, x_min, y_min, x_max, y_max, vis_path):
    """Thread-safe visualization saving."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(image, cmap='gray')
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      fill=False, color='r', linestyle='--', alpha=0.5)
    ax1.add_patch(rect)
    ax1.set_title('Original Image with ROI')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Raw OpenCV Mask')
    ax2.axis('off')
    
    ax3.imshow(filtered_mask, cmap='gray')
    ax3.set_title('ROI Filtered Mask')
    ax3.axis('off')
    
    plt.suptitle(f"Processing Times: Pre={times['pre_processing_time']:.2f}ms, "
               f"Contours={times['find_contours_time']:.2f}ms, "
               f"Total={times['total_processing_time']:.2f}ms")
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()

def process_condition(pipeline, condition_dir, run_output_dir, run_id: str, background_path: str, roi_coordinates: Dict, pbar=None):
    """Process all images within a condition directory using parallel processing."""
    condition_output_dir = run_output_dir / condition_dir.name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        condition_roi = roi_coordinates.get(condition_dir.name)
        if condition_roi is None:
            logger.warning(f"No ROI coordinates found for condition {condition_dir.name}")
            return []
        
        # Get image paths
        image_mapping = collect_image_paths(condition_dir)
        temp_dir = create_hardlinks_for_batch(condition_dir, image_mapping)
        
        # Prepare arguments for parallel processing
        process_args = []
        for image_path in temp_dir.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
                process_args.append((
                    str(image_path),
                    background_path,
                    pipeline,
                    condition_output_dir,
                    condition_roi
                ))
        
        # Process images in parallel using a ProcessPoolExecutor
        results = []
        n_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for result in executor.map(process_single_image, process_args):
                if result is not None:
                    results.append(result)
                if pbar:
                    pbar.update(1)
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing condition {condition_dir.name}: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        if 'temp_dir' in locals() and temp_dir.exists():
            shutil.rmtree(temp_dir)

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
    """Save processing results to CSV files using efficient batch writing."""
    # Prepare data in memory first
    timing_data = []
    contour_data = []
    
    for result in results:
        # Add timing data
        timing_data.append({
            'image_path': result['image_path'],
            'condition': result['condition'],
            'pre_processing_time': result['times']['pre_processing_time'],
            'find_contours_time': result['times']['find_contours_time'],
            'total_processing_time': result['times']['total_processing_time']
        })
        
        # Add contour data
        for i, contour in enumerate(result['contours']):
            contour_data.append({
                'image_path': result['image_path'],
                'condition': result['condition'],
                'contour_id': i,
                'contour_points': contour.tolist()
            })
    
    # Write data in batches
    chunk_size = 1000
    
    # Save timing data
    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv(output_dir / 'processing_times.csv', index=False)
    
    # Save contour data in chunks to manage memory
    for i in range(0, len(contour_data), chunk_size):
        chunk = pd.DataFrame(contour_data[i:i + chunk_size])
        mode = 'w' if i == 0 else 'a'
        header = i == 0
        chunk.to_csv(output_dir / 'contours.csv', index=False, mode=mode, header=header)

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
            dilate_iterations=args.dilate_iterations,
            erode_iterations=args.erode_iterations
        )
        
        # Process conditions with progress tracking
        start_time = time.time()
        all_results = []
        total_images = count_total_images(condition_dirs)
        
        with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
            # Process conditions in parallel
            with ThreadPoolExecutor(max_workers=min(len(condition_dirs), 4)) as executor:
                future_to_condition = {}
                
                for condition_dir in condition_dirs:
                    # Find background image
                    background_path = next((str(f) for f in condition_dir.glob("*") 
                                         if 'background' in f.name.lower()), None)
                    
                    if not background_path:
                        logger.warning(f"No background image found in condition {condition_dir.name}")
                        continue
                    
                    print(f"\nProcessing condition: {condition_dir.name}")
                    print(f"Using ROI coordinates: {roi_coordinates[condition_dir.name]}")
                    
                    # Submit condition processing task
                    future = executor.submit(
                        process_condition,
                        pipeline=pipeline,
                        condition_dir=condition_dir,
                        run_output_dir=run_output_dir,
                        run_id=run_id,
                        background_path=background_path,
                        roi_coordinates=roi_coordinates,
                        pbar=pbar
                    )
                    future_to_condition[future] = condition_dir.name
                
                # Collect results as they complete
                for future in future_to_condition:
                    try:
                        condition_results = future.result()
                        all_results.extend(condition_results)
                    except Exception as e:
                        condition = future_to_condition[future]
                        logger.error(f"Error processing condition {condition}: {str(e)}")
        
        total_runtime = time.time() - start_time
        
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