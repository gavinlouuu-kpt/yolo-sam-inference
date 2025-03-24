# This example is used to run inference on a project where a project could contain multiple conditions and within each condition, there could be multiple batches.
# All the batches of a condition will be concatenated and then run through the pipeline together as a single batch.
# To avoid images within the same condition of differnet batches having the same name, we will add the folder name as a prefix to the image name.

# feature: Gate ROI of all the conditions in the beginning of the pipeline

from yolo_sam_inference import CellSegmentationPipeline
from yolo_sam_inference.pipeline import ParallelCellSegmentationPipeline
from yolo_sam_inference.utils import (
    setup_logger,
    load_model_from_mlflow,
    load_model_from_registry,
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
import os
import mlflow
from mlflow.tracking import MlflowClient
from yolo_sam_inference.web.app import get_roi_coordinates_web

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
        default='/home/mib-p5-a5000/code/ai-cyto/output',
        help='Directory to save output results'
    )
    
    # Add new arguments for model registry
    parser.add_argument(
        '--model-name',
        type=str,
        required=False,
        help='Name of the registered model in MLflow Model Registry'
    )
    
    parser.add_argument(
        '--model-version',
        type=str,
        required=False,
        help='Version of the registered model (if not specified, latest version will be used)'
    )
    
    parser.add_argument(
        '--registry-uri',
        type=str,
        default='http://localhost:5000',
        help='URI of the MLflow Model Registry server'
    )
    
    # Add S3/MinIO credentials
    parser.add_argument(
        '--aws-access-key-id',
        type=str,
        default='mibadmin',
        help='AWS access key ID for S3/MinIO'
    )
    
    parser.add_argument(
        '--aws-secret-access-key',
        type=str,
        default='cuhkminio',
        help='AWS secret access key for S3/MinIO'
    )
    
    parser.add_argument(
        '--s3-endpoint-url',
        type=str,
        default='http://localhost:9000',
        help='S3/MinIO endpoint URL'
    )
    
    # Keep the existing arguments for backward compatibility
    parser.add_argument(
        '--experiment-id',
        type=str,
        default="320489803004134590",
        help='MLflow experiment ID (used only if model-name is not provided)'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default="c2fef8a01dea4fc4a8876414a90b3f69",
        help='MLflow run ID (used only if model-name is not provided)'
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
    
    # Add MLflow tracking arguments
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
        help='MLflow tracking server URI'
    )
    
    parser.add_argument(
        '--inference-experiment-name',
        type=str,
        default='yolo-sam-inference',
        help='MLflow experiment name for tracking inference'
    )
    
    parser.add_argument(
        '--log-to-mlflow',
        action='store_true',
        help='Enable MLflow tracking for this inference run'
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

def filter_cells_by_roi(metrics_df: pd.DataFrame, roi_coordinates: Dict[str, Dict[str, int]]) -> pd.DataFrame:
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
    for condition, roi in roi_coordinates.items():
        logger.info(f"Processing condition: {condition} with ROI: {roi}")
        
        condition_df = metrics_df[metrics_df['condition'] == condition]
        if condition_df.empty:
            logger.warning(f"No data found for condition: {condition}")
            continue
            
        try:
            # Calculate center y coordinate from bounding box (horizontal position)
            condition_df['center_y'] = (condition_df['min_y'] + condition_df['max_y']) / 2
            
            # Filter based on center y coordinate (horizontal position)
            # Note: For backward compatibility with the current pipeline, we only use x coordinates
            gated_condition_df = condition_df[
                (condition_df['center_y'] >= roi['x_min']) & 
                (condition_df['center_y'] <= roi['x_max'])
            ]
            
            # Remove the temporary center_y column
            gated_condition_df = gated_condition_df.drop(columns=['center_y'])
            
            logger.info(f"Filtered {len(gated_condition_df)} cells from {len(condition_df)} for condition {condition}")
            
            gated_df = pd.concat([gated_df, gated_condition_df])
            
        except Exception as e:
            logger.error(f"Error processing condition {condition}: {str(e)}")
            raise
    
    return gated_df

def safe_log_artifact(file_path, artifact_path=None):
    """
    Safely log an artifact to MLflow, handling path conversion between Windows and WSL.
    
    Args:
        file_path: Path to the file to log
        artifact_path: Optional subdirectory within the artifact directory to log to
    """
    path_obj = Path(file_path)
    if path_obj.exists():
        if artifact_path:
            mlflow.log_artifact(str(path_obj), artifact_path)
        else:
            mlflow.log_artifact(str(path_obj))
    else:
        # Try to convert Windows path to WSL path if needed
        if str(path_obj).startswith('D:'):
            # Convert Windows path to WSL path
            wsl_path = str(path_obj).replace('D:', '/mnt/d').replace('\\', '/')
            wsl_path_obj = Path(wsl_path)
            if wsl_path_obj.exists():
                if artifact_path:
                    mlflow.log_artifact(str(wsl_path_obj), artifact_path)
                else:
                    mlflow.log_artifact(str(wsl_path_obj))
            else:
                logger.warning(f"Artifact file not found: {file_path} or {wsl_path}")
        else:
            logger.warning(f"Artifact file not found: {file_path}")

def safe_log_image(img, artifact_path):
    """
    Safely log an image to MLflow.
    
    Args:
        img: Image data to log
        artifact_path: Path within the artifact directory to log to
    """
    try:
        mlflow.log_image(img, artifact_path)
    except Exception as e:
        logger.warning(f"Failed to log image to MLflow: {str(e)}")

def log_roi_selection_to_mlflow(condition_dirs, roi_coordinates: Dict[str, Dict[str, int]], run_output_dir: Path):
    """
    Create and log ROI selection visualizations to MLflow.
    
    Args:
        condition_dirs: List of condition directories
        roi_coordinates: Dictionary of ROI coordinates for each condition
        run_output_dir: Path to the run output directory
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Rectangle
        
        # Create directory for ROI visualizations
        roi_vis_dir = run_output_dir / "roi_visualizations"
        roi_vis_dir.mkdir(exist_ok=True)
        
        for condition_dir in condition_dirs:
            condition_name = condition_dir.name
            
            # Skip if no ROI coordinates for this condition
            if condition_name not in roi_coordinates:
                continue
                
            # Find a representative image from this condition
            image_files = list(condition_dir.glob("**/*.png")) + list(condition_dir.glob("**/*.jpg")) + list(condition_dir.glob("**/*.tiff"))
            if not image_files:
                continue
                
            # Use the first image
            image_path = image_files[0]
            
            # Read the image
            img = cv2.imread(str(image_path))
            if img is None:
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get ROI coordinates
            roi = roi_coordinates[condition_name]
            min_x, max_x = roi.get('min_x', 0), roi.get('max_x', img.shape[1])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            
            # Add ROI rectangle
            height = img.shape[0]
            rect = Rectangle((min_x, 0), max_x - min_x, height, 
                            linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add title and labels
            ax.set_title(f'ROI Selection for {condition_name}')
            ax.text(min_x + 10, height - 30, f'min_x: {min_x}', color='white', 
                   backgroundcolor='black', fontsize=12)
            ax.text(max_x - 100, height - 30, f'max_x: {max_x}', color='white', 
                   backgroundcolor='black', fontsize=12)
            
            # Save figure
            roi_vis_path = roi_vis_dir / f"{condition_name}_roi.png"
            plt.savefig(roi_vis_path)
            
            # Log to MLflow
            safe_log_artifact(roi_vis_path, "roi_visualizations")
            plt.close()
            
        return roi_vis_dir
        
    except Exception as e:
        logger.warning(f"Failed to create ROI visualizations: {str(e)}")
        return None

def log_visualizations_to_mlflow(run_output_dir: Path, max_images_per_condition: int = 5):
    """
    Log a sample of visualization images to MLflow.
    
    Args:
        run_output_dir: Path to the run output directory
        max_images_per_condition: Maximum number of images to log per condition
    """
    # Find all visualization directories
    for condition_dir in run_output_dir.iterdir():
        if not condition_dir.is_dir():
            continue
            
        # Look for visualizations directory
        vis_dir = condition_dir / "visualizations"
        if not vis_dir.exists() or not vis_dir.is_dir():
            # Try WSL path conversion if needed
            if str(vis_dir).startswith('D:'):
                wsl_vis_dir = Path(str(vis_dir).replace('D:', '/mnt/d').replace('\\', '/'))
                if wsl_vis_dir.exists() and wsl_vis_dir.is_dir():
                    vis_dir = wsl_vis_dir
                else:
                    continue
            else:
                continue
            
        # Get all visualization images
        vis_images = list(vis_dir.glob("*.png"))
        
        # Sample images if there are too many
        if len(vis_images) > max_images_per_condition:
            import random
            vis_images = random.sample(vis_images, max_images_per_condition)
        
        # Log each image
        for img_path in vis_images:
            # Read the image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Log to MLflow
            safe_log_image(img, f"visualizations/{condition_dir.name}/{img_path.name}")

def create_and_log_summary_figures(metrics_df: pd.DataFrame, gated_metrics_df: pd.DataFrame, run_output_dir: Path):
    """
    Create and log summary figures to MLflow.
    
    Args:
        metrics_df: DataFrame containing all cell metrics
        gated_metrics_df: DataFrame containing gated cell metrics
        run_output_dir: Path to the run output directory
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from matplotlib.figure import Figure
        
        # Create directory for figures
        figures_dir = run_output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # 1. Create histogram of cell areas by condition
        plt.figure(figsize=(12, 8))
        conditions = metrics_df['condition'].unique()
        
        for condition in conditions:
            condition_data = metrics_df[metrics_df['condition'] == condition]['area']
            plt.hist(condition_data, alpha=0.5, bins=30, label=condition)
        
        plt.title('Cell Area Distribution by Condition')
        plt.xlabel('Cell Area (pixels)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save and log figure
        area_hist_path = figures_dir / "cell_area_histogram.png"
        plt.savefig(area_hist_path)
        safe_log_artifact(area_hist_path, "figures")
        plt.close()
        
        # 2. Create bar chart of cell counts by condition (before and after gating)
        plt.figure(figsize=(10, 6))
        
        # Count cells by condition
        condition_counts = metrics_df['condition'].value_counts()
        gated_condition_counts = gated_metrics_df['condition'].value_counts()
        
        # Ensure all conditions are represented in gated counts
        for condition in condition_counts.index:
            if condition not in gated_condition_counts:
                gated_condition_counts[condition] = 0
        
        # Sort both series by the same order
        gated_condition_counts = gated_condition_counts.reindex(condition_counts.index)
        
        # Create bar chart
        x = np.arange(len(condition_counts))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, condition_counts, width, label='All Cells')
        ax.bar(x + width/2, gated_condition_counts, width, label='Gated Cells')
        
        ax.set_title('Cell Counts by Condition (Before and After Gating)')
        ax.set_xlabel('Condition')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.set_xticklabels(condition_counts.index)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Add count labels on top of bars
        for i, v in enumerate(condition_counts):
            ax.text(i - width/2, v + 5, str(v), ha='center')
            
        for i, v in enumerate(gated_condition_counts):
            ax.text(i + width/2, v + 5, str(v), ha='center')
        
        # Save and log figure
        count_bar_path = figures_dir / "cell_count_by_condition.png"
        plt.savefig(count_bar_path)
        safe_log_artifact(count_bar_path, "figures")
        plt.close()
        
        # 3. Create scatter plot of cell metrics (e.g., area vs. circularity)
        if 'circularity' in metrics_df.columns:
            plt.figure(figsize=(12, 8))
            
            for condition in conditions:
                condition_data = metrics_df[metrics_df['condition'] == condition]
                plt.scatter(
                    condition_data['area'], 
                    condition_data['circularity'],
                    alpha=0.5,
                    label=condition
                )
            
            plt.title('Cell Area vs. Circularity by Condition')
            plt.xlabel('Cell Area (pixels)')
            plt.ylabel('Circularity')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Save and log figure
            scatter_path = figures_dir / "area_vs_circularity.png"
            plt.savefig(scatter_path)
            safe_log_artifact(scatter_path, "figures")
            plt.close()
        
        # Return the directory containing the figures
        return figures_dir
        
    except Exception as e:
        logger.warning(f"Failed to create summary figures: {str(e)}")
        return None

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
        
        # Set up MLflow tracking if enabled
        mlflow_run = None
        if args.log_to_mlflow:
            print(f"\nSetting up MLflow tracking at {args.tracking_uri}")
            mlflow.set_tracking_uri(args.tracking_uri)
            
            # Set up S3/MinIO credentials for MLflow if provided
            if args.aws_access_key_id and args.aws_secret_access_key and args.s3_endpoint_url:
                os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id
                os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key
                os.environ["MLFLOW_S3_ENDPOINT_URL"] = args.s3_endpoint_url
                os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
            
            # Create or set experiment
            experiment = mlflow.set_experiment(args.inference_experiment_name)
            
            # Start MLflow run
            mlflow_run = mlflow.start_run(
                run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Inference on project: {project_dir.name}"
            )
            
            # Log parameters
            mlflow.log_params({
                "project_dir": str(project_dir),
                "output_dir": str(run_output_dir),
                "device": args.device,
                "num_pipelines": args.num_pipelines,
                "run_id": run_id,
            })
            
            # Log model source information
            if args.model_name:
                mlflow.log_params({
                    "model_source": "registry",
                    "model_name": args.model_name,
                    "model_version": args.model_version or "latest",
                })
            else:
                mlflow.log_params({
                    "model_source": "mlflow_run",
                    "source_experiment_id": args.experiment_id,
                    "source_run_id": args.run_id,
                })
            
            print(f"MLflow tracking initialized - Run ID: {mlflow_run.info.run_id}")
        
        # Get all condition directories
        condition_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        # Get ROI coordinates for each condition using web interface
        print("\nOpening web interface for ROI selection...")
        print("Please select ROI coordinates for each condition in the browser window.")
        print("Click two points on each image to define the min and max X coordinates.")
        roi_coordinates = get_roi_coordinates_web(condition_dirs, run_output_dir)
        print("\nROI coordinates collected successfully!")
        
        # Log ROI selection to MLflow if enabled
        if args.log_to_mlflow and mlflow_run:
            print("\nLogging ROI selection to MLflow...")
            log_roi_selection_to_mlflow(condition_dirs, roi_coordinates, run_output_dir)
        
        print(f"\nInitializing pipeline... [Run ID: {run_id}]")
        # Get model path from MLflow
        if args.model_name:
            # Use the new registry-based model loading
            yolo_model_path = load_model_from_registry(
                model_name=args.model_name,
                model_version=args.model_version,
                registry_uri=args.registry_uri,
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                s3_endpoint_url=args.s3_endpoint_url
            )
        else:
            # Fallback to the old method for backward compatibility
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
        
        # Log metrics and artifacts to MLflow if enabled
        if args.log_to_mlflow and mlflow_run:
            # Log metrics
            mlflow.log_metric("total_runtime_seconds", total_runtime)
            mlflow.log_metric("total_images_processed", total_images)
            mlflow.log_metric("total_cells_detected", combined_results.total_timing["total_cells"])
            mlflow.log_metric("avg_cells_per_image", combined_results.total_timing["total_cells"] / total_images if total_images > 0 else 0)
            
            # Log timing metrics
            for key, value in combined_results.total_timing.items():
                if key != "total_cells":  # Skip non-timing metrics
                    mlflow.log_metric(f"timing_{key}_seconds", value)
            
            # Log condition-specific metrics
            condition_counts = metrics_df['condition'].value_counts().to_dict()
            for condition, count in condition_counts.items():
                mlflow.log_metric(f"cells_in_{condition}", count)
                
                # Log gated cell counts
                gated_count = len(gated_metrics_df[gated_metrics_df['condition'] == condition])
                mlflow.log_metric(f"gated_cells_in_{condition}", gated_count)
            
            # Log artifacts with safe function
            safe_log_artifact(run_output_dir / 'cell_metrics.csv')
            safe_log_artifact(run_output_dir / 'gated_cell_metrics.csv')
            safe_log_artifact(run_output_dir / 'roi_coordinates.json')
            safe_log_artifact(run_output_dir / 'run_summary.txt')
            
            # Log condition-specific artifacts
            for condition_dir in run_output_dir.iterdir():
                if condition_dir.is_dir() and condition_dir.name in roi_coordinates:
                    # Log the condition summary file if it exists
                    summary_file = condition_dir / f"{condition_dir.name}_summary.txt"
                    safe_log_artifact(summary_file, f"conditions/{condition_dir.name}")
                    
                    # Log the condition metrics file
                    metrics_file = condition_dir / "cell_metrics.csv"
                    safe_log_artifact(metrics_file, f"conditions/{condition_dir.name}")
                    
                    # Log the gated metrics file
                    gated_metrics_file = condition_dir / "gated_cell_metrics.csv"
                    safe_log_artifact(gated_metrics_file, f"conditions/{condition_dir.name}")
            
            # Create and log summary figures
            print("\nCreating and logging summary figures...")
            figures_dir = create_and_log_summary_figures(metrics_df, gated_metrics_df, run_output_dir)
            
            # Log visualization images
            print("\nLogging visualization samples to MLflow...")
            log_visualizations_to_mlflow(run_output_dir)
            
            # End MLflow run
            mlflow.end_run()
            print(f"\nMLflow tracking completed - Run ID: {mlflow_run.info.run_id}")
            print(f"View run details at: {args.tracking_uri}/#/experiments/{experiment.experiment_id}/runs/{mlflow_run.info.run_id}")
        
        print(f"\nResults saved to: {run_output_dir}")
        print(f"ROI coordinates saved to: {run_output_dir}/roi_coordinates.json")
        print("Gated metrics files have been created for each condition and the overall results.")
        
    except Exception as e:
        # End MLflow run if active
        if 'mlflow_run' in locals() and mlflow_run:
            mlflow.end_run(status="FAILED")
        
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()






