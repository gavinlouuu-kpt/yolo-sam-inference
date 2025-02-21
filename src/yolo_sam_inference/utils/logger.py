import logging

def setup_logger(name: str = __name__):
    """Configure and return the logger for the pipeline.
    
    Args:
        name: Logger name, defaults to module name
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name) 