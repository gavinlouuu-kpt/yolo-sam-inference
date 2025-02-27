"""Script to process frames using YOLO model and interactive ROI selection.

This script takes a folder of frames as input, allows user to select an ROI,
and uses YOLO model to identify frames containing targets. It outputs:
1. Full frames containing targets
2. ROI-cropped frames containing targets
Each output folder includes a background frame with no target.
"""

import argparse
from pathlib import Path
import cv2
import shutil
import logging
from typing import Tuple, List
import torch
from ultralytics import YOLO
from tqdm import tqdm

from yolo_sam_inference.utils import (
    setup_logger,
    load_model_from_mlflow
)

# Set up logger
logger = setup_logger(__name__)
logger.setLevel('INFO')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process frames using YOLO model with interactive ROI selection.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        required=True,
        help='Directory containing input frames'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=False,
        help='Directory to save output results (defaults to input_dir + "_output")'
    )
    
    parser.add_argument(
        '--experiment-id',
        type=str,
        default="320489803004134590",
        help='MLflow experiment ID for loading YOLO model'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        default="c2fef8a01dea4fc4a8876414a90b3f69",
        help='MLflow run ID for loading YOLO model'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to run inference on'
    )
    
    return parser.parse_args()

def get_roi_coordinates(image_path: Path) -> Tuple[int, int, int, int]:
    """Get ROI coordinates from user using OpenCV.
    
    Returns:
        Tuple[int, int, int, int]: x_min, y_min, x_max, y_max coordinates
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create window for ROI selection
    window_name = "Select ROI - Draw rectangle and press SPACE or ENTER"
    cv2.namedWindow(window_name)
    
    # Let user draw rectangle ROI
    roi = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    
    return roi  # Returns (x_min, y_min, width, height)

def setup_output_dirs(output_base: Path) -> Tuple[Path, Path]:
    """Create and return output directory paths.
    
    Returns:
        Tuple[Path, Path]: Paths to full frames and cropped frames directories
    """
    full_frames_dir = output_base / "full_frames_with_target"
    cropped_frames_dir = output_base / "cropped_roi_with_target"
    
    full_frames_dir.mkdir(parents=True, exist_ok=True)
    cropped_frames_dir.mkdir(parents=True, exist_ok=True)
    
    return full_frames_dir, cropped_frames_dir

def process_frames(
    input_dir: Path,
    output_dir: Path,
    yolo_model,
    roi_coords: Tuple[int, int, int, int],
) -> None:
    """Process frames using YOLO model and save results.
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Base directory for outputs
        yolo_model: Loaded YOLO model
        roi_coords: (x_min, y_min, width, height) of selected ROI
        device: Device to run inference on
    """
    # Get output directories
    full_frames_dir, cropped_frames_dir = setup_output_dirs(output_dir)
    
    # Get all image files
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.tiff"))
    
    # Track frames with and without targets
    frames_with_target = []
    frames_without_target = []
    
    # Process each frame
    logger.info("Running YOLO detection on frames...")
    for img_path in tqdm(image_files, desc="Processing frames", unit="frame"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
            
        # Extract ROI
        x, y, w, h = roi_coords
        roi = img[y:y+h, x:x+w]
        
        # Run YOLO inference on ROI
        results = yolo_model(roi, verbose=False)[0]  # Get first (and only) result
        
        # Check if any detections in ROI
        if len(results.boxes) > 0:
            frames_with_target.append((img_path, img, roi))
        else:
            frames_without_target.append((img_path, img, roi))
    
    # Save frames with targets
    logger.info(f"Found {len(frames_with_target)} frames with targets")
    if frames_with_target:
        logger.info("Saving frames with targets...")
        for i, (img_path, full_frame, roi) in enumerate(tqdm(frames_with_target, desc="Saving target frames", unit="frame")):
            # Save full frame
            output_name = f"frame_with_target_{i+1}{img_path.suffix}"
            cv2.imwrite(str(full_frames_dir / output_name), full_frame)
            
            # Save cropped ROI
            cv2.imwrite(str(cropped_frames_dir / output_name), roi)
    
    # Save one background frame (without target) if available
    if frames_without_target:
        logger.info("Saving background frame...")
        bg_path, bg_frame, bg_roi = frames_without_target[0]
        bg_name = f"background{bg_path.suffix}"
        
        # Save full background frame
        cv2.imwrite(str(full_frames_dir / bg_name), bg_frame)
        
        # Save cropped background ROI
        cv2.imwrite(str(cropped_frames_dir / bg_name), bg_roi)
        
        logger.info(f"Saved background frame (from {len(frames_without_target)} available frames without targets)")
    else:
        logger.warning("No frames without targets found - background frame not available")

def main():
    """Main function to run the frame processing pipeline."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    
    # Set output directory to input directory + '_output' if not specified
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_output"
    
    # Check if input directory exists and contains images
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.tiff"))
    if not image_files:
        raise ValueError(f"No image files found in input directory: {input_dir}")
    
    # Load YOLO model
    logger.info("Loading YOLO model...")
    model_path = load_model_from_mlflow(
        experiment_id=args.experiment_id,
        run_id=args.run_id
    )
    yolo_model = YOLO(model_path)
    
    # Set model device
    if args.device == 'cuda' and torch.cuda.is_available():
        yolo_model.to('cuda')
    else:
        yolo_model.to('cpu')
    
    # Get ROI coordinates using first image
    logger.info("Please select ROI in the opened window...")
    roi_coords = get_roi_coordinates(image_files[0])
    
    # Setup output directories
    full_frames_dir, cropped_frames_dir = setup_output_dirs(output_dir)
    
    # Process frames
    logger.info("Processing frames...")
    process_frames(
        input_dir=input_dir,
        output_dir=output_dir,
        yolo_model=yolo_model,
        roi_coords=roi_coords
    )
    
    logger.info("Processing complete!")

if __name__ == '__main__':
    main()
