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
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
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

def load_model_from_registry(
    model_name: str, 
    model_version: str = None, 
    registry_uri: str = "http://localhost:5000",
    aws_access_key_id: str = "mibadmin",
    aws_secret_access_key: str = "cuhkminio",
    s3_endpoint_url: str = "http://localhost:9000"
) -> str:
    """Load a model from MLflow Model Registry and return the path to the model weights.
    
    Args:
        model_name: Name of the registered model
        model_version: Version of the model to load (if None, latest version is used)
        registry_uri: URI of the MLflow Model Registry server
        aws_access_key_id: AWS access key ID for S3/MinIO
        aws_secret_access_key: AWS secret access key for S3/MinIO
        s3_endpoint_url: S3/MinIO endpoint URL
        
    Returns:
        Path to the downloaded model weights
    """
    logger.info(f"Loading model from MLflow Registry - Model: {model_name}, Version: {model_version or 'latest'}")
    
    # Set up MLflow tracking URI
    logger.info(f"Using MLflow registry URI: {registry_uri}")
    mlflow.set_tracking_uri(registry_uri)
    
    # Set up S3/MinIO credentials
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    
    # Download the model artifacts
    logger.info("Downloading model artifacts from registry...")
    client = MlflowClient()
    
    try:
        # Get the model version
        if model_version is None:
            # Get the latest version
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            
            # Sort by version number and get the latest
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            model_version = latest_version.version
            logger.info(f"Using latest model version: {model_version}")
        
        # Get the run ID for the model version
        model_details = client.get_model_version(model_name, model_version)
        run_id = model_details.run_id
        
        # Download the model artifacts
        local_dir = client.download_artifacts(run_id, "weights/best.pt")
        logger.info(f"Model artifacts downloaded successfully to: {local_dir}")
        return local_dir
    except Exception as e:
        logger.error(f"Error downloading model artifacts from registry: {str(e)}")
        raise 