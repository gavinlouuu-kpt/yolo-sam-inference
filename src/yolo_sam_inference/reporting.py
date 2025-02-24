"""Module for generating reports and summaries from pipeline results."""

from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any
from .pipeline import BatchProcessingResult

def save_results_to_csv(
    batch_result: BatchProcessingResult,
    output_dir: Path
) -> None:
    """Save metrics and timing data to CSV files."""
    if batch_result.metrics_data:
        metrics_df = pd.DataFrame(batch_result.metrics_data)
        
        # Ensure proper column ordering for metrics
        # First get the fixed columns we want at the start
        fixed_columns = ['condition', 'image_name', 'cell_id']
        # Get the fixed columns that actually exist in the DataFrame
        existing_fixed_columns = [col for col in fixed_columns if col in metrics_df.columns]
        # Then get any remaining columns
        other_columns = [col for col in metrics_df.columns if col not in fixed_columns]
        # Combine them in the desired order
        ordered_columns = existing_fixed_columns + other_columns
        # Reorder the DataFrame columns
        metrics_df = metrics_df[ordered_columns]
        
        metrics_df.to_csv(output_dir / 'cell_metrics.csv', index=False)
    
    if batch_result.timing_data:
        timing_df = pd.DataFrame(batch_result.timing_data)
        
        # Ensure proper column ordering for timing
        fixed_columns = ['condition', 'image_name', 'cells_processed']
        existing_fixed_columns = [col for col in fixed_columns if col in timing_df.columns]
        other_columns = [col for col in timing_df.columns if col not in fixed_columns]
        ordered_columns = existing_fixed_columns + other_columns
        timing_df = timing_df[ordered_columns]
        
        timing_df.to_csv(output_dir / 'processing_times.csv', index=False)

def generate_summary_text(
    batch_result: BatchProcessingResult,
    input_dir: Path,
    output_dir: Path,
    run_id: str,
    total_runtime: float,
    is_condition_summary: bool = False
) -> str:
    """Generate a comprehensive summary text."""
    num_images = len(batch_result.results)
    total_timing = batch_result.total_timing
    
    summary = []
    if is_condition_summary:
        condition_name = batch_result.results[0].condition if batch_result.results else "Unknown"
        summary.append(f"Condition Summary: {condition_name}")
        summary.append("=" * (len(summary[0])) + "\n")
    else:
        summary.append("Pipeline Run Summary")
        summary.append("==================\n")
    
    summary.append(f"Run ID: {run_id}")
    summary.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"Input Directory: {input_dir.absolute()}")
    summary.append(f"Output Directory: {output_dir.absolute()}\n")
    
    # Add condition-specific breakdown for pipeline summary
    if not is_condition_summary:
        summary.append("Condition Breakdown")
        summary.append("==================")
        conditions = {}
        for result in batch_result.results:
            condition = getattr(result, 'condition', 'Unknown')
            if condition not in conditions:
                conditions[condition] = {'images': 0, 'cells': 0}
            conditions[condition]['images'] += 1
            conditions[condition]['cells'] += result.num_cells
        
        for condition, stats in conditions.items():
            summary.append(f"Condition: {condition}")
            summary.append(f"  Images processed: {stats['images']}")
            summary.append(f"  Cells detected: {stats['cells']}")
            summary.append(f"  Average cells per image: {stats['cells']/stats['images']:.1f}\n")
    
    summary.append("Processing Statistics")
    summary.append("====================")
    summary.append(f"Total images processed: {num_images}")
    summary.append(f"Total cells detected: {total_timing['total_cells']}")
    summary.append(f"Average cells per image: {total_timing['total_cells']/num_images:.1f}\n")
    
    summary.append("Timing Statistics (averaged per image)")
    summary.append("===================================")
    summary.append(f"Image loading: {(total_timing['image_load']/num_images)*1000:.1f}ms")
    summary.append(f"YOLO detection: {(total_timing['yolo_detection']/num_images)*1000:.1f}ms")
    summary.append(f"SAM preprocessing: {(total_timing['sam_preprocess']/num_images)*1000:.1f}ms")
    summary.append(f"SAM inference: {(total_timing['sam_inference_total']/num_images)*1000:.1f}ms")
    summary.append(f"SAM postprocessing: {(total_timing['sam_postprocess_total']/num_images)*1000:.1f}ms")
    summary.append(f"Metrics calculation: {(total_timing['metrics_total']/num_images)*1000:.1f}ms")
    summary.append(f"Visualization: {(total_timing['visualization']/num_images)*1000:.1f}ms\n")
    
    summary.append("Overall Performance")
    summary.append("==================")
    summary.append(f"Total runtime: {total_runtime:.1f}s")
    summary.append(f"Average time per image: {(total_runtime/num_images):.1f}s")
    if total_timing['total_cells'] > 0:
        summary.append(f"Average time per cell: {(total_runtime/total_timing['total_cells'])*1000:.1f}ms")
    
    return "\n".join(summary)

def print_summary(batch_result: BatchProcessingResult, total_runtime: float) -> None:
    """Print a summary of the processing results to the console."""
    num_images = len(batch_result.results)
    total_timing = batch_result.total_timing
    
    print("\n" + "=" * 80)
    print("PIPELINE PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Add condition breakdown
    print("\nCondition Breakdown:")
    conditions = {}
    for result in batch_result.results:
        condition = getattr(result, 'condition', 'Unknown')
        if condition not in conditions:
            conditions[condition] = {'images': 0, 'cells': 0}
        conditions[condition]['images'] += 1
        conditions[condition]['cells'] += result.num_cells
    
    for condition, stats in conditions.items():
        print(f"\nCondition: {condition}")
        print(f"  Images processed: {stats['images']}")
        print(f"  Cells detected: {stats['cells']}")
        print(f"  Average cells per image: {stats['cells']/stats['images']:.1f}")
    
    print("\nOverall Statistics:")
    print(f"Total images processed: {num_images}")
    print(f"Total cells detected: {total_timing['total_cells']}")
    print(f"Average cells per image: {total_timing['total_cells']/num_images:.1f}")
    print(f"\nTiming Breakdown (averaged per image):")
    print(f"Image loading: {(total_timing['image_load']/num_images)*1000:.1f}ms")
    print(f"YOLO detection: {(total_timing['yolo_detection']/num_images)*1000:.1f}ms")
    print(f"SAM preprocessing: {(total_timing['sam_preprocess']/num_images)*1000:.1f}ms")
    print(f"SAM inference: {(total_timing['sam_inference_total']/num_images)*1000:.1f}ms")
    print(f"SAM postprocessing: {(total_timing['sam_postprocess_total']/num_images)*1000:.1f}ms")
    print(f"Metrics calculation: {(total_timing['metrics_total']/num_images)*1000:.1f}ms")
    print(f"Visualization: {(total_timing['visualization']/num_images)*1000:.1f}ms")
    print(f"\nTotal runtime: {total_runtime:.1f}s")
    print(f"Average time per image: {(total_runtime/num_images):.1f}s")
    if total_timing['total_cells'] > 0:
        print(f"Average time per cell: {(total_runtime/total_timing['total_cells'])*1000:.1f}ms")
    print("=" * 80)

def save_run_summary(
    batch_result: BatchProcessingResult,
    input_dir: Path,
    output_dir: Path,
    run_id: str,
    total_runtime: float,
    summary_name: str = "run_summary.txt",
    is_condition_summary: bool = False
) -> None:
    """Save a comprehensive run summary to a text file."""
    summary_text = generate_summary_text(
        batch_result,
        input_dir,
        output_dir,
        run_id,
        total_runtime,
        is_condition_summary
    )
    with open(output_dir / summary_name, "w") as f:
        f.write(summary_text) 