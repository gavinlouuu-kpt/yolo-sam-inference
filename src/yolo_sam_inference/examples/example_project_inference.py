# This example is used to run inference on a project where a project could contain multiple conditions and within each condition, there could be multiple batches.
# All the batches of a condition will be concatenated and then run through the pipeline together as a single batch.
# To avoid images within the same condition of differnet batches having the same name, we will add the folder name as a prefix to the image name.

from yolo_sam_inference import CellSegmentationPipeline
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
from typing import Tuple

# Set up logger with reduced verbosity
logger = setup_logger(__name__)
logger.setLevel('INFO')

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
        
        # Create a temporary pipeline without run_id to avoid nested folders
        temp_pipeline = CellSegmentationPipeline(
            yolo_model_path=pipeline.yolo_model.model.pt_path,
            sam_model_type=pipeline.sam_model_type,
            device=pipeline.device
        )
        
        # Process all images in the temporary directory
        batch_result = temp_pipeline.process_directory(
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
    """Count total number of images across all conditions."""
    total_images = 0
    for condition_dir in condition_dirs:
        temp_dir = collect_images_from_batches(condition_dir)
        total_images += len(list(temp_dir.glob("*.png")) + list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.tiff")))
        shutil.rmtree(temp_dir)
    return total_images

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
        
        print(f"Initializing pipeline... [Run ID: {run_id}]")
        # Get model path from MLflow
        yolo_model_path = load_model_from_mlflow(args.experiment_id, args.run_id)
        
        # Initialize the pipeline
        pipeline = CellSegmentationPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_type="facebook/sam-vit-huge",
            device=args.device
        )
        
        # Get all condition directories and count total images
        condition_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        total_images = count_total_images(condition_dirs)
        
        # Process each condition with progress bar tracking total images
        start_time = time.time()
        batch_results = []
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
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()






