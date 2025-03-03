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
        """Initialize OpenCV pipeline with parameters.
        
        Args:
            threshold_value: Threshold value for binary thresholding
            dilate_iterations: Number of dilation iterations
            erode_iterations: Number of erosion iterations
            blur_kernel_size: Kernel size for Gaussian blur
            blur_sigma: Sigma value for Gaussian blur
        """
        self.threshold_value = threshold_value
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def process_image(self, image_path: str, background_path: str) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """Process an image to find contours using OpenCV pipeline.
        
        Args:
            image_path: Path to the image file
            background_path: Path to the background image file
            
        Returns:
            Tuple containing list of contours and processing times
        """
        # Load images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        if background is None:
            raise ValueError(f"Failed to load background image: {background_path}")
            
        blurred_bg = cv2.GaussianBlur(background, self.blur_kernel_size, self.blur_sigma)

        total_start_time = time.perf_counter()
        
        # Pre-processing
        pre_processing_start = time.perf_counter()
        blurred = cv2.GaussianBlur(image, self.blur_kernel_size, self.blur_sigma)
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        dilate1 = cv2.dilate(binary, self.kernel, iterations=self.dilate_iterations)
        erode1 = cv2.erode(dilate1, self.kernel, iterations=self.erode_iterations)
        erode2 = cv2.erode(erode1, self.kernel, iterations=1)
        dilate2 = cv2.dilate(erode2, self.kernel, iterations=1)
        
        pre_processing_end = time.perf_counter()
        pre_processing_time = (pre_processing_end - pre_processing_start) * 1000

        # Find contours
        find_contours_start = time.perf_counter()
        contours, _ = cv2.findContours(dilate2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    """Collect all image paths from a condition directory."""
    image_mapping = {}
    
    # Get all image files (excluding background)
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        image_files.extend(list(condition_dir.glob(f"*{ext}")))
    
    # Filter out background image
    image_files = [f for f in image_files if 'background' not in f.name.lower()]
    
    # Create mapping
    for image_file in image_files:
        image_mapping[str(image_file)] = {
            'original_path': image_file,
            'new_name': image_file.name
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

def process_condition(pipeline, condition_dir, run_output_dir, run_id: str, background_path: str, roi_coordinates: Dict, pbar=None):
    """Process all images within a condition directory."""
    # Create output directory for this condition
    condition_output_dir = run_output_dir / condition_dir.name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get ROI coordinates for this condition
        condition_roi = roi_coordinates.get(condition_dir.name)
        if condition_roi is None:
            logger.warning(f"No ROI coordinates found for condition {condition_dir.name}")
            return []
        
        # Extract ROI coordinates
        x_min = condition_roi['x_min']
        x_max = condition_roi['x_max']
        y_min = condition_roi['y_min']
        y_max = condition_roi['y_max']
        
        # Get image paths
        image_mapping = collect_image_paths(condition_dir)
        
        # Create hardlinks in temporary directory
        temp_dir = create_hardlinks_for_batch(condition_dir, image_mapping)
        
        # Process all images in the temporary directory
        results = []
        for image_path in temp_dir.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
                try:
                    # Process the image
                    contours, times = pipeline.process_image(str(image_path), background_path)
                    
                    # Convert contours to mask
                    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    mask = pipeline.contours_to_mask(contours, image.shape)
                    
                    # Filter contours based on ROI
                    filtered_contours = []
                    filtered_mask = np.zeros_like(mask)
                    for contour in contours:
                        # Calculate contour center
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            center_x = int(M["m10"] / M["m00"])  # x coordinate
                            center_y = int(M["m01"] / M["m00"])  # y coordinate
                            # Check if contour center is within ROI
                            if (x_min <= center_x <= x_max) and (y_min <= center_y <= y_max):
                                filtered_contours.append(contour)
                                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    
                    # Save results
                    result = {
                        'image_path': str(image_path),
                        'condition': condition_dir.name,
                        'contours': filtered_contours,
                        'mask': filtered_mask,
                        'times': times,
                        'roi': {
                            'x_min': x_min,
                            'x_max': x_max,
                            'y_min': y_min,
                            'y_max': y_max
                        }
                    }
                    results.append(result)
                    
                    # Save visualization
                    vis_path = condition_output_dir / f"{image_path.stem}_result.png"
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Plot original image with ROI rectangle
                    ax1.imshow(image, cmap='gray')
                    # Draw ROI rectangle
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, color='r', linestyle='--', alpha=0.5)
                    ax1.add_patch(rect)
                    ax1.set_title('Original Image with ROI')
                    ax1.axis('off')
                    
                    # Plot filtered mask
                    ax2.imshow(filtered_mask, cmap='gray')
                    ax2.set_title('OpenCV Mask (ROI filtered)')
                    ax2.axis('off')
                    
                    plt.suptitle(f"Processing Times: Pre={times['pre_processing_time']:.2f}ms, "
                               f"Contours={times['find_contours_time']:.2f}ms, "
                               f"Total={times['total_processing_time']:.2f}ms")
                    plt.tight_layout()
                    plt.savefig(vis_path)
                    plt.close()
                    
                    if pbar:
                        pbar.update(1)
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {str(e)}")
                    continue
        
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
    """Save processing results to CSV files."""
    # Save timing data
    timing_data = []
    for result in results:
        timing_data.append({
            'image_path': result['image_path'],
            'condition': result['condition'],
            'pre_processing_time': result['times']['pre_processing_time'],
            'find_contours_time': result['times']['find_contours_time'],
            'total_processing_time': result['times']['total_processing_time']
        })
    
    timing_df = pd.DataFrame(timing_data)
    timing_df.to_csv(output_dir / 'processing_times.csv', index=False)
    
    # Save contour data
    contour_data = []
    for result in results:
        for i, contour in enumerate(result['contours']):
            contour_data.append({
                'image_path': result['image_path'],
                'condition': result['condition'],
                'contour_id': i,
                'contour_points': contour.tolist()
            })
    
    contour_df = pd.DataFrame(contour_data)
    contour_df.to_csv(output_dir / 'contours.csv', index=False)

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Convert paths to Path objects
        project_dir = Path(args.project_dir)
        base_output_dir = Path(args.output_dir)
        
        # Validate project directory
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
        
        # Create unique run output directory
        run_output_dir, run_id = create_run_output_dir(base_output_dir)
        
        # Get all condition directories
        condition_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        # Get ROI coordinates for each condition using web interface
        print("\nOpening web interface for ROI selection...")
        print("Please select ROI coordinates for each condition in the browser window.")
        print("Click two points on each image to define the min and max X coordinates.")
        print("You must select ROI for ALL conditions before processing can begin.")
        
        # Get ROI coordinates for all conditions
        roi_coordinates = get_roi_coordinates_web(condition_dirs, run_output_dir)
        
        # Verify ROI coordinates for all conditions
        missing_conditions = [d.name for d in condition_dirs if d.name not in roi_coordinates]
        if missing_conditions:
            raise ValueError(f"Missing ROI coordinates for conditions: {', '.join(missing_conditions)}. Please select ROI for all conditions.")
            
        print("\nROI coordinates collected successfully for all conditions!")
        print("ROI coordinates:")
        for condition, coords in roi_coordinates.items():
            print(f"  {condition}: {coords}")
        
        print(f"\nInitializing OpenCV pipeline... [Run ID: {run_id}]")
        # Initialize the OpenCV pipeline
        pipeline = OpenCVPipeline(
            threshold_value=args.threshold,
            dilate_iterations=args.dilate_iterations,
            erode_iterations=args.erode_iterations
        )
        
        # Process each condition with progress bar tracking total images
        start_time = time.time()
        all_results = []
        total_images = count_total_images(condition_dirs)
        
        with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
            for condition_dir in condition_dirs:
                # Find background image in condition directory
                background_path = None
                for file in condition_dir.glob("*"):
                    if 'background' in file.name.lower():
                        background_path = str(file)
                        break
                
                if not background_path:
                    logger.warning(f"No background image found in condition {condition_dir.name}")
                    continue
                
                print(f"\nProcessing condition: {condition_dir.name}")
                print(f"Using ROI coordinates: {roi_coordinates[condition_dir.name]}")
                
                condition_results = process_condition(
                    pipeline=pipeline,
                    condition_dir=condition_dir,
                    run_output_dir=run_output_dir,
                    run_id=run_id,
                    background_path=background_path,
                    roi_coordinates=roi_coordinates,
                    pbar=pbar
                )
                all_results.extend(condition_results)
        
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