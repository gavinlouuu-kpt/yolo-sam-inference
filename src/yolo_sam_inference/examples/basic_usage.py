from yolo_sam_inference import CellSegmentationPipeline
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_from_mlflow(experiment_id: str, run_id: str) -> str:
    """Load a model from MLflow run and return the path to the model weights."""
    logger.info(f"Loading model from MLflow - Experiment ID: {experiment_id}, Run ID: {run_id}")
    
    # Set up MLflow tracking URI from environment or use default
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:///D:/code/ai_cytometry/yolo-sam-training/mlruns')
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set the experiment
    logger.info(f"Setting MLflow experiment: {experiment_id}")
    mlflow.set_experiment(experiment_id=experiment_id)
    
    # Download the model artifacts
    logger.info("Downloading model artifacts...")
    client = MlflowClient()
    try:
        local_dir = client.download_artifacts(run_id, "weights/best.pt")
        logger.info(f"Model artifacts downloaded successfully to: {local_dir}")
        return local_dir
    except Exception as e:
        logger.error(f"Error downloading model artifacts: {str(e)}")
        raise

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
        
        # Results will contain a list of dictionaries, one for each processed image
        for result in results:
            logger.info(f"\nProcessing results for {result['image_path']}:")
            cell_count = len(result['cell_metrics'])
            logger.info(f"Number of cells detected: {cell_count}")
            
            # Calculate summary statistics
            if cell_count > 0:
                areas = [m['area'] for m in result['cell_metrics']]
                circularities = [m['circularity'] for m in result['cell_metrics']]
                mean_intensities = [m['mean_intensity'] for m in result['cell_metrics']]
                
                logger.info(f"Summary statistics:")
                logger.info(f"Average area: {sum(areas)/len(areas):.2f} pixels")
                logger.info(f"Average circularity: {sum(circularities)/len(circularities):.3f}")
                logger.info(f"Average mean intensity: {sum(mean_intensities)/len(mean_intensities):.2f}")
            
            # Each cell's metrics
            for i, metrics in enumerate(result['cell_metrics']):
                logger.debug(f"\nCell {i+1} details:")
                logger.debug(f"Area: {metrics['area']} pixels")
                logger.debug(f"Circularity: {metrics['circularity']:.3f}")
                logger.debug(f"Deformability: {metrics['deformability']:.3f}")
                logger.debug(f"Mean intensity: {metrics['mean_intensity']:.2f}")
                logger.debug(f"Convex hull area: {metrics['convex_hull_area']} pixels")
                logger.debug(f"Perimeter: {metrics['perimeter']:.2f} pixels")
                
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Starting cell segmentation pipeline...")
    main()
    logger.info("Pipeline execution completed.") 