from yolo_sam_inference import CellSegmentationPipeline
from yolo_sam_inference.utils import (
    setup_logger,
    load_model_from_mlflow,
    calculate_summary_statistics,
    report_summary_statistics,
    report_cell_details
)
from pathlib import Path

logger = setup_logger(__name__)

def main():
    try:
        # MLflow experiment and run IDs
        experiment_id = "320489803004134590"
        run_id = "c2fef8a01dea4fc4a8876414a90b3f69"
        
        # Get model path from MLflow
        logger.info("Starting model loading process...")
        yolo_model_path = load_model_from_mlflow(experiment_id, run_id)
        
        # Initialize the pipeline with model paths
        logger.info("Initializing CellSegmentationPipeline...")
        logger.info("Using YOLO model path: %s", yolo_model_path)
        pipeline = CellSegmentationPipeline(
            yolo_model_path=yolo_model_path,
            sam_model_type="facebook/sam-vit-huge",  # or sam-vit-large, sam-vit-base
            device="cuda"  # or "cpu" for CPU inference
        )
        logger.info("Pipeline initialized successfully")
        
        # Path containing input images
        input_dir = Path(__file__).parent / "example_image"
        output_dir = Path("D:\\code\\ai_cytometry\\yolo-sam-inference-pipeline\\inference_output")
        
        logger.info(f"Processing images from directory: {input_dir}")
        logger.info(f"Output will be saved to: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images in the directory
        logger.info("Starting directory processing...")
        results = pipeline.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            save_visualizations=True  # Save visualization of results
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