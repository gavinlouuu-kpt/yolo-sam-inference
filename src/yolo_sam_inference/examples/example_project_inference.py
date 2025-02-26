# This example is used to run inference on a project where a project could contain multiple conditions and within each condition, there could be multiple batches.
# All the batches of a condition will be concatenated and then run through the pipeline together as a single batch.
# To avoid images within the same condition of differnet batches having the same name, we will add the folder name as a prefix to the image name.

# feature: Gate ROI of all the conditions in the beginning of the pipeline

from yolo_sam_inference import CellSegmentationPipeline
from yolo_sam_inference.pipeline import ParallelCellSegmentationPipeline
from yolo_sam_inference.utils import (
    setup_logger,
    load_model_from_mlflow,
)
from yolo_sam_inference.reporting import (
    save_results_to_csv,
    print_summary,
    save_run_summary
)
from yolo_sam_inference.pipeline import BatchProcessingResult
from pathlib import Path
import argparse
import time
import shutil
from tqdm import tqdm
from datetime import datetime
import uuid
from typing import Tuple, Dict
import cv2
import json
import pandas as pd
import logging

# Set up logger with reduced verbosity
logger = setup_logger(__name__)
logger.setLevel('INFO')

# Reduce YOLO logging verbosity
yolo_logger = logging.getLogger('ultralytics')
yolo_logger.setLevel(logging.WARNING)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Project-based cell segmentation pipeline for microscopy images.',
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
        default='D:\\code\\ai_cytometry\\yolo-sam-inference-pipeline\\project_inference_output',
        help='Directory to save output results'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default="320489803004134590",
        help='MLflow experiment ID'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default="c2fef8a01dea4fc4a8876414a90b3f69",
        help='MLflow run ID'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to run inference on'
    )
    
    parser.add_argument(
        '--num-pipelines',
        type=int,
        default=2,
        help='Number of parallel pipelines to use for processing'
    )
    
    return parser.parse_args()

def collect_images_from_batches(condition_dir):
    """Collect all images from all batches in a condition directory."""
    # Create a temporary directory for combined images
    temp_dir = condition_dir / "temp_combined_batches"
    temp_dir.mkdir(exist_ok=True)
    
    # Get all batch directories
    batch_dirs = [d for d in condition_dir.iterdir() if d.is_dir() and d.name != "temp_combined_batches"]
    
    # Copy images from each batch with batch name prefix
    for batch_dir in batch_dirs:
        image_files = list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.tiff"))
        for image_file in image_files:
            # Create new filename with batch prefix
            new_filename = f"{batch_dir.name}_{image_file.name}"
            # Copy the file to temp directory
            shutil.copy2(image_file, temp_dir / new_filename)
    
    return temp_dir

def process_condition(pipeline, condition_dir, run_output_dir, run_id: str, pbar=None):
    """Process all batches within a condition directory."""
    # Create output directory for this condition
    condition_output_dir = run_output_dir / condition_dir.name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Collect and combine all images from all batches
        temp_dir = collect_images_from_batches(condition_dir)
        
        # Process all images in the temporary directory
        batch_result = pipeline.process_directory(
            input_dir=temp_dir,
            output_dir=condition_output_dir,
            save_visualizations=True,
            pbar=pbar
        )
        
        # Add condition information to results
        for result in batch_result.results:
            result.condition = condition_dir.name
        
        # Save condition-specific results directly in the condition directory
        save_results_to_csv(batch_result, condition_output_dir)
        save_run_summary(
            batch_result,
            temp_dir,  # Use temp_dir as input dir for condition summary
            condition_output_dir,
            run_id,
            batch_result.total_timing['total_time'],
            summary_name=f"{condition_dir.name}_summary.txt",
            is_condition_summary=True
        )
            
        return batch_result
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def combine_batch_results(batch_results):
    """Combine multiple batch results into a single BatchProcessingResult."""
    all_results = []
    all_metrics = []
    all_timing = []
    total_timing = {
        "image_load": 0,
        "yolo_detection": 0,
        "sam_preprocess": 0,
        "sam_inference_total": 0,
        "sam_postprocess_total": 0,
        "metrics_total": 0,
        "visualization": 0,
        "total_time": 0,
        "total_cells": 0
    }
    
    for batch_result in batch_results:
        all_results.extend(batch_result.results)
        
        # Add condition information to metrics data
        for result in batch_result.results:
            condition = getattr(result, 'condition', 'Unknown')
            image_name = Path(result.image_path).name
            
            # Update cell metrics with image and condition info
            for cell_idx, cell_metric in enumerate(result.cell_metrics):
                cell_metric.update({
                    'condition': condition,
                    'image_name': image_name,
                    'cell_id': cell_idx
                })
                all_metrics.append(cell_metric)
            
            # Update timing data with condition info
            timing_entry = next((t for t in batch_result.timing_data if t['image_name'] == image_name), None)
            if timing_entry:
                timing_entry['condition'] = condition
                all_timing.append(timing_entry)
        
        # Aggregate timing data
        for key in total_timing:
            total_timing[key] += batch_result.total_timing[key]
    
    return BatchProcessingResult(
        results=all_results,
        total_timing=total_timing,
        metrics_data=all_metrics,
        timing_data=all_timing
    )

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
        # Get all batch directories
        batch_dirs = [d for d in condition_dir.iterdir() if d.is_dir()]
        # Count images in each batch
        for batch_dir in batch_dirs:
            total_images += len(list(batch_dir.glob("*.png")) + 
                              list(batch_dir.glob("*.jpg")) + 
                              list(batch_dir.glob("*.tiff")))
    return total_images

def get_roi_coordinates(image_path: Path) -> Tuple[int, int]:
    """Get min and max X coordinates from user using OpenCV."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create window and set mouse callback
    window_name = "Select ROI - Click two points for min and max X coordinates (Press 'r' to reset, 'c' to confirm)"
    cv2.namedWindow(window_name)
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append(x)
            # Draw vertical line at clicked point
            img_copy = image.copy()
            for px in points:
                cv2.line(img_copy, (px, 0), (px, image.shape[0]), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        # Show image
        if not points:
            cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Reset points
            points.clear()
            cv2.imshow(window_name, image)
        elif key == ord('c') and len(points) == 2:  # Confirm selection
            break
    
    cv2.destroyAllWindows()
    
    return min(points), max(points)

def save_roi_coordinates(coordinates: Dict[str, Tuple[int, int]], output_dir: Path) -> None:
    """Save ROI coordinates to a JSON file."""
    roi_file = output_dir / "roi_coordinates.json"
    with open(roi_file, 'w') as f:
        json.dump(coordinates, f, indent=2)

def filter_cells_by_roi(metrics_df: pd.DataFrame, roi_coordinates: Dict[str, Tuple[int, int]]) -> pd.DataFrame:
    """Filter cell metrics based on ROI coordinates for each condition."""
    # Create a copy of the DataFrame
    gated_df = pd.DataFrame()
    
    # Print available columns for debugging
    logger.info(f"Available columns in metrics DataFrame: {metrics_df.columns.tolist()}")
    
    # Check required columns exist
    required_columns = ['condition', 'min_y', 'max_y']
    missing_columns = [col for col in required_columns if col not in metrics_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in metrics DataFrame: {missing_columns}")
    
    # Filter cells for each condition
    for condition, (min_x, max_x) in roi_coordinates.items():
        logger.info(f"Processing condition: {condition} with ROI: min_x={min_x}, max_x={max_x}")
        
        condition_df = metrics_df[metrics_df['condition'] == condition]
        if condition_df.empty:
            logger.warning(f"No data found for condition: {condition}")
            continue
            
        try:
            # Calculate center y coordinate from bounding box (horizontal position)
            condition_df['center_y'] = (condition_df['min_y'] + condition_df['max_y']) / 2
            
            # Filter based on center y coordinate (horizontal position)
            gated_condition_df = condition_df[
                (condition_df['center_y'] >= min_x) & 
                (condition_df['center_y'] <= max_x)
            ]
            
            # Remove the temporary center_y column
            gated_condition_df = gated_condition_df.drop(columns=['center_y'])
            
            logger.info(f"Filtered {len(gated_condition_df)} cells from {len(condition_df)} for condition {condition}")
            
            gated_df = pd.concat([gated_df, gated_condition_df])
            
        except Exception as e:
            logger.error(f"Error processing condition {condition}: {str(e)}")
            raise
    
    return gated_df

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
        
        # Get ROI coordinates for each condition
        print("\nSelecting ROI coordinates for each condition...")
        roi_coordinates = {}
        for condition_dir in condition_dirs:
            print(f"\nProcessing condition: {condition_dir.name}")
            # Get first image from first batch in condition
            batch_dirs = [d for d in condition_dir.iterdir() if d.is_dir()]
            if not batch_dirs:
                continue
                
            image_files = list(batch_dirs[0].glob("*.png")) + list(batch_dirs[0].glob("*.jpg")) + list(batch_dirs[0].glob("*.tiff"))
            if not image_files:
                continue
                
            print(f"Please select ROI coordinates for condition {condition_dir.name} using the first image")
            min_x, max_x = get_roi_coordinates(image_files[0])
            roi_coordinates[condition_dir.name] = (min_x, max_x)
            print(f"ROI coordinates for {condition_dir.name}: min_x={min_x}, max_x={max_x}")
        
        # Save ROI coordinates
        save_roi_coordinates(roi_coordinates, run_output_dir)
        
        print(f"\nInitializing pipeline... [Run ID: {run_id}]")
        # Get model path from MLflow
        yolo_model_path = load_model_from_mlflow(args.experiment_id, args.run_id)
        
        # Initialize the pipeline
        pipeline = ParallelCellSegmentationPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_type="facebook/sam-vit-base",
            device=args.device,
            num_pipelines=args.num_pipelines
        )
        
        # Process each condition with progress bar tracking total images
        start_time = time.time()
        batch_results = []
        total_images = count_total_images(condition_dirs)
        
        with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
            for condition_dir in condition_dirs:
                batch_result = process_condition(
                    pipeline=pipeline,
                    condition_dir=condition_dir,
                    run_output_dir=run_output_dir,
                    run_id=run_id,
                    pbar=pbar
                )
                batch_results.append(batch_result)
        
        total_runtime = time.time() - start_time
        
        # Combine all results and generate run summary
        print("\nAggregating results and generating summary...")
        combined_results = combine_batch_results(batch_results)
        
        # Save combined run summary directly in the run output directory
        save_results_to_csv(combined_results, run_output_dir)
        
        # Create gated versions of the metrics files
        print("\nCreating gated metrics files...")
        # Read the original metrics file
        metrics_df = pd.read_csv(run_output_dir / 'cell_metrics.csv')
        
        # Debug: Print metrics file info
        print("\nMetrics file information:")
        print(f"Number of rows: {len(metrics_df)}")
        print(f"Columns: {metrics_df.columns.tolist()}")
        print("\nFirst few rows of data:")
        print(metrics_df.head())
        
        # Apply ROI filtering
        gated_metrics_df = filter_cells_by_roi(metrics_df, roi_coordinates)
        
        # Save gated metrics
        gated_metrics_df.to_csv(run_output_dir / 'gated_cell_metrics.csv', index=False)
        
        # Create condition-specific gated files
        for condition in roi_coordinates.keys():
            condition_metrics = metrics_df[metrics_df['condition'] == condition]
            gated_condition_metrics = filter_cells_by_roi(
                condition_metrics, 
                {condition: roi_coordinates[condition]}
            )
            
            # Save condition-specific files
            condition_dir = run_output_dir / condition
            gated_condition_metrics.to_csv(
                condition_dir / f'gated_cell_metrics.csv',
                index=False
            )
        
        save_run_summary(
            combined_results,
            project_dir,
            run_output_dir,
            run_id,
            total_runtime,
            summary_name="run_summary.txt"
        )
        print_summary(combined_results, total_runtime)
        
        print(f"\nResults saved to: {run_output_dir}")
        print(f"ROI coordinates saved to: {run_output_dir}/roi_coordinates.json")
        print("Gated metrics files have been created for each condition and the overall results.")
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()






