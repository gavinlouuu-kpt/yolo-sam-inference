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
import numpy as np

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

def draw_detections(image: np.ndarray, results) -> np.ndarray:
    """Draw YOLO detections on the image.
    
    Args:
        image: Image to draw on
        results: YOLO results object
    
    Returns:
        Image with drawn detections
    """
    img_with_boxes = image.copy()
    
    # Draw each detection
    for box in results.boxes:
        # Get coordinates and confidence
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        
        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw confidence score
        label = f"{conf:.2f}"
        cv2.putText(img_with_boxes, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes

def is_box_fully_contained(box_coords, roi_coords, margin=2):
    """Check if a bounding box is fully contained within ROI with a small margin.
    
    Args:
        box_coords: (x1, y1, x2, y2) of the bounding box
        roi_coords: (x, y, w, h) of the ROI
        margin: pixel margin to consider for boundary touch detection
    
    Returns:
        bool: True if box is fully contained within ROI
    """
    x1, y1, x2, y2 = box_coords
    roi_x, roi_y, roi_w, roi_h = roi_coords
    
    # Add margin to make sure box isn't touching boundary
    return (x1 >= roi_x + margin and 
            y1 >= roi_y + margin and 
            x2 <= roi_x + roi_w - margin and 
            y2 <= roi_y + roi_h - margin)

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
    """
    # Get output directories
    full_frames_dir, cropped_frames_dir = setup_output_dirs(output_dir)
    
    # Create directory for debug visualizations
    debug_dir = output_dir / "debug_visualizations"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.tiff"))
    
    # Track frames with and without targets
    frames_with_target = []
    frames_without_target = []
    
    # Set confidence threshold
    CONF_THRESHOLD = 0.5
    
    # Get ROI coordinates
    roi_x, roi_y, roi_w, roi_h = roi_coords
    
    # Process each frame
    logger.info("Running YOLO detection on frames...")
    for img_path in tqdm(image_files, desc="Processing frames", unit="frame"):
        # Read image and convert to RGB (YOLO expects RGB)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference on full image
        results = yolo_model(img_rgb, verbose=False)[0]
        
        # Filter detections by confidence and ROI
        confident_boxes = []
        boxes_touch_boundary = False
        
        for box in results.boxes:
            if float(box.conf[0]) >= CONF_THRESHOLD:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Check if box center is within ROI
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                
                if (roi_x <= box_center_x <= roi_x + roi_w and 
                    roi_y <= box_center_y <= roi_y + roi_h):
                    
                    # Check if box touches ROI boundary
                    if is_box_fully_contained((x1, y1, x2, y2), roi_coords):
                        confident_boxes.append(box)
                    else:
                        boxes_touch_boundary = True
                        logger.debug(f"Box touches boundary in {img_path.name}")
        
        # Extract ROI for visualization and saving
        roi_bgr = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Create visualization
        vis_img = img.copy()
        # Draw ROI rectangle
        cv2.rectangle(vis_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        
        # Draw all detections in red, confident ROI detections in green, boundary-touching in yellow
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            # Determine box color based on its status
            is_confident_roi = any(
                np.array_equal(box.xyxy[0].cpu().numpy(), b.xyxy[0].cpu().numpy())
                for b in confident_boxes
            )
            
            is_touching = (not is_confident_roi and 
                         not is_box_fully_contained((x1, y1, x2, y2), roi_coords) and
                         roi_x <= (x1 + x2)/2 <= roi_x + roi_w and
                         roi_y <= (y1 + y2)/2 <= roi_y + roi_h)
            
            # Green: valid detection, Yellow: touching boundary, Red: outside or low confidence
            color = (0, 255, 0) if is_confident_roi else (0, 255, 255) if is_touching else (0, 0, 255)
            
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            label = f"{conf:.2f}"
            cv2.putText(vis_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save debug visualization
        debug_name = f"debug_{img_path.stem}_detections.jpg"
        cv2.imwrite(str(debug_dir / debug_name), vis_img)
        
        # Only accept frames with exactly one valid detection
        if len(confident_boxes) == 1 and not boxes_touch_boundary:
            frames_with_target.append((img_path, img, roi_bgr))
            logger.debug(f"Found single valid detection in {img_path.name}")
        else:
            frames_without_target.append((img_path, img, roi_bgr))
            if len(confident_boxes) > 1:
                logger.debug(f"Multiple detections ({len(confident_boxes)}) in {img_path.name}")
            elif boxes_touch_boundary:
                logger.debug(f"Detection touches boundary in {img_path.name}")
            else:
                logger.debug(f"No valid detections in {img_path.name}")
    
    # Save frames with targets
    logger.info(f"Found {len(frames_with_target)} frames with targets")
    if frames_with_target:
        logger.info("Saving frames with targets...")
        for img_path, full_frame, roi in tqdm(frames_with_target, desc="Saving target frames", unit="frame"):
            # Keep original name and append suffix
            output_name = f"{img_path.stem}_with_target{img_path.suffix}"
            
            # Save full frame
            cv2.imwrite(str(full_frames_dir / output_name), full_frame)
            
            # Save cropped ROI
            cv2.imwrite(str(cropped_frames_dir / output_name), roi)
    
    # Save one background frame (without target) if available
    if frames_without_target:
        logger.info("Saving background frame...")
        bg_path, bg_frame, bg_roi = frames_without_target[0]
        # Keep original name and append suffix
        bg_name = f"{bg_path.stem}_background{bg_path.suffix}"
        
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
