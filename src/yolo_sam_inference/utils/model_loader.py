import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from .logger import setup_logger

logger = setup_logger(__name__)

def load_model_from_mlflow(experiment_id: str, run_id: str) -> str:
    """Load a model from MLflow run and return the path to the model weights.
    
    Args:
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
        
    Returns:
        Path to the downloaded model weights
    """
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