# This example is used to run inference on a project where a project could contain multiple conditions and within each condition, there could be multiple batches.
# All the batches of a condition will be concatenated and then run through the pipeline together as a single batch.
# To avoid images within the same condition of differnet batches having the same name, we will add the folder name as a prefix to the image name.

from yolo_sam_inference import CellSegmentationPipeline
from yolo_sam_inference.utils import (
    setup_logger,
    load_model_from_mlflow,
    calculate_summary_statistics,
    report_summary_statistics,
    report_cell_details
)
from pathlib import Path
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm

# Set up logger with reduced verbosity
logger = setup_logger(__name__)
logger.setLevel('INFO')  # Only show INFO and above

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
    """
    Collect all images from all batches in a condition directory and copy them to a temporary directory,
    prefixing each image with its batch name to avoid naming conflicts.
    
    Args:
        condition_dir: Path to condition directory
        
    Returns:
        Path to temporary directory containing all images
    """
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

def process_condition(pipeline, condition_dir, output_dir, pbar=None):
    """
    Process all batches within a condition directory as a single combined batch.
    
    Args:
        pipeline: Initialized CellSegmentationPipeline instance
        condition_dir: Path to condition directory
        output_dir: Path to save outputs
        pbar: Optional tqdm progress bar for tracking progress
        
    Returns:
        List of dictionaries containing results for all images in the condition
    """
    # Create output directory for this condition
    condition_output_dir = output_dir / condition_dir.name
    condition_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Collect and combine all images from all batches
        temp_dir = collect_images_from_batches(condition_dir)
        
        # Process all images in the temporary directory
        results = pipeline.process_directory(
            input_dir=temp_dir,
            output_dir=condition_output_dir,
            save_visualizations=True,
            pbar=pbar
        )
        
        # Add condition information
        for result in results:
            result['condition'] = condition_dir.name
            
        return results
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def aggregate_results(all_results, output_dir):
    """
    Aggregate results across all conditions and save summary statistics.
    
    Args:
        all_results: List of results from all conditions
        output_dir: Directory to save aggregated results
    """
    # Prepare data for aggregation
    metrics_data = []
    timing_data = []
    
    for result in all_results:
        # Base info for both metrics and timing
        base_info = {
            'condition': result['condition'],
            'image': Path(result['image_path']).name,
        }
        
        # Prepare timing data
        timing_info = base_info.copy()
        timing_info.update({
            'num_cells': result['num_cells'],
            'image_load_time': result['timing']['image_load'],
            'yolo_detection_time': result['timing']['yolo_detection'],
            'sam_preprocess_time': result['timing']['sam_preprocess'],
            'sam_inference_time': result['timing']['sam_inference_total'],
            'sam_postprocess_time': result['timing']['sam_postprocess_total'],
            'metrics_calculation_time': result['timing']['metrics_total'],
            'visualization_time': result['timing']['visualization'],
            'total_time': result['timing']['total_time'],
            'cells_processed': result['timing']['cells_processed']
        })
        timing_data.append(timing_info)
        
        # Prepare metrics data
        for i, cell_metrics in enumerate(result['cell_metrics']):
            cell_info = base_info.copy()
            cell_info['cell_id'] = i
            cell_info['num_cells'] = result['num_cells']
            cell_info.update(cell_metrics)
            metrics_data.append(cell_info)
    
    # Create and save metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = output_dir / 'project_cell_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    
    # Create and save timing DataFrame
    timing_df = pd.DataFrame(timing_data)
    timing_file = output_dir / 'project_timing_metrics.csv'
    timing_df.to_csv(timing_file, index=False)
    
    # Generate condition-wise summary for metrics
    condition_summary = metrics_df.groupby('condition').agg({
        'num_cells': ['mean', 'std', 'min', 'max'],
        'cell_id': 'count'
    }).round(2)
    
    # Generate condition-wise summary for timing
    timing_summary = timing_df.groupby('condition').agg({
        'total_time': ['mean', 'std', 'min', 'max'],
        'cells_processed': ['sum', 'mean']
    }).round(3)
    
    # Print final summary
    print("\nProject Summary Statistics:")
    print("=" * 50)
    print(f"Total conditions processed: {len(metrics_df['condition'].unique())}")
    print(f"Total images processed: {len(metrics_df['image'].unique())}")
    print(f"Total cells detected: {len(metrics_df)}")
    print(f"\nTiming Summary:")
    print(f"Average processing time per image: {timing_df['total_time'].mean():.3f}s")
    print(f"Average processing time per cell: {timing_df['total_time'].sum() / timing_df['cells_processed'].sum():.3f}s")
    print(f"\nResults saved to:")
    print(f"- Cell metrics: {metrics_file}")
    print(f"- Timing metrics: {timing_file}")
    
    # Print condition-wise summary
    print("\nCondition-wise Summary:")
    print("=" * 50)
    for condition in condition_summary.index:
        print(f"\nCondition: {condition}")
        print(f"Images processed: {len(timing_df[timing_df['condition'] == condition])}")
        print(f"Total cells: {condition_summary.loc[condition, ('cell_id', 'count')]:.0f}")
        print(f"Cells per image: {condition_summary.loc[condition, ('num_cells', 'mean')]:.1f} ± {condition_summary.loc[condition, ('num_cells', 'std')]:.1f}")
        print(f"Processing time per image: {timing_summary.loc[condition, ('total_time', 'mean')]:.3f}s ± {timing_summary.loc[condition, ('total_time', 'std')]:.3f}s")

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Convert paths to Path objects
        project_dir = Path(args.project_dir)
        output_dir = Path(args.output_dir)
        
        # Validate project directory
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
        
        print("Initializing pipeline...")
        # Get model path from MLflow
        yolo_model_path = load_model_from_mlflow(args.experiment_id, args.run_id)
        
        # Initialize the pipeline
        pipeline = CellSegmentationPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_type="facebook/sam-vit-huge",
            device=args.device
        )
        
        # Get all condition directories
        condition_dirs = [d for d in project_dir.iterdir() if d.is_dir()]
        
        # Count total number of images across all conditions
        total_images = 0
        for condition_dir in condition_dirs:
            temp_dir = collect_images_from_batches(condition_dir)
            total_images += len(list(temp_dir.glob("*.png")) + list(temp_dir.glob("*.jpg")) + list(temp_dir.glob("*.tiff")))
            shutil.rmtree(temp_dir)
        
        # Process each condition with progress bar tracking total images
        all_results = []
        with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
            for condition_dir in condition_dirs:
                condition_results = process_condition(pipeline, condition_dir, output_dir, pbar)
                all_results.extend(condition_results)
        
        # Aggregate and save results
        print("\nAggregating results and generating summary...")
        aggregate_results(all_results, output_dir)
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()






