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

logger = setup_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cell segmentation pipeline for microscopy images.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing input images (supports .png, .jpg, .tiff)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='D:\\code\\ai_cytometry\\yolo-sam-inference-pipeline\\inference_output',
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

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Convert paths to Path objects
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        # Validate input directory
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        # Get model path from MLflow
        logger.info("Starting model loading process...")
        yolo_model_path = load_model_from_mlflow(args.experiment_id, args.run_id)
        
        # Initialize the pipeline with model paths
        logger.info("Initializing CellSegmentationPipeline...")
        logger.info("Using YOLO model path: %s", yolo_model_path)
        pipeline = CellSegmentationPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_type="facebook/sam-vit-huge",  # or sam-vit-large, sam-vit-base
            device=args.device
        )
        logger.info("Pipeline initialized successfully")
        
        logger.info(f"Processing images from directory: {input_dir}")
        logger.info(f"Output will be saved to: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images in the directory
        logger.info("Starting directory processing...")
        results = pipeline.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            save_visualizations=True
        )
        logger.info(f"Directory processing completed. Processed {len(results)} images.")
        
        # Process results for each image
        for result in results:
            logger.info(f"\nProcessing results for {result['image_path']}:")
            cell_count = len(result['cell_metrics'])
            logger.info(f"Number of cells detected: {cell_count}")
            
            if cell_count > 0:
                # Calculate and report summary statistics
                stats = calculate_summary_statistics(result['cell_metrics'])
                report_summary_statistics(stats)
                
                # Report detailed metrics for each cell
                for i, metrics in enumerate(result['cell_metrics']):
                    report_cell_details(i, metrics)

    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting cell segmentation pipeline...")
    main()
    logger.info("Pipeline execution completed.") 