from pathlib import Path
from typing import List, Dict, Union, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
from .utils.metrics import calculate_metrics
import matplotlib
# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure
from datetime import datetime
import uuid
import pandas as pd
import time
import warnings
from .utils.image_utils import save_optimized_tiff, save_mask_as_tiff
from tqdm import tqdm

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

class CellSegmentationPipeline:
    def __init__(
        self,
        yolo_model_path: Union[str, Path],
        sam_model_type: str = "facebook/sam-vit-huge",
        device: str = "cuda"
    ):
        """
        Initialize the cell segmentation pipeline.
        
        Args:
            yolo_model_path: Path to the YOLO model weights
            sam_model_type: HuggingFace model identifier for SAM
                          (e.g., 'facebook/sam-vit-huge', 'facebook/sam-vit-large', 'facebook/sam-vit-base')
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.sam_model_type = sam_model_type
        
        # Initialize YOLO model with verbose=False to disable progress output
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.args['verbose'] = False  # Set verbose in args dictionary
        
        # Initialize SAM model and processor from HuggingFace
        self.sam_model = SamModel.from_pretrained(sam_model_type).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_type)
        
        # Generate a unique run ID for this pipeline instance
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualizations: bool = True,
        pbar: tqdm = None
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            pbar: Optional tqdm progress bar for tracking progress
            
        Returns:
            List of dictionaries containing results for each image
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images first to gather statistics
        results = []
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.tiff"))
        
        # Initialize timing aggregation
        total_timing = {
            "image_load": 0,
            "yolo_detection": 0,
            "sam_preprocess": 0,
            "sam_inference_total": 0,
            "sam_postprocess_total": 0,
            "metrics_total": 0,
            "visualization": 0,
            "total_time": 0,
            "total_cells": 0
        }
        
        # Prepare data for CSV
        all_metrics_data = []
        all_timing_data = []
        
        start_time = time.time()
        for image_path in image_files:
            result = self.process_single_image(
                image_path,
                output_dir / image_path.name,
                save_visualizations
            )
            results.append(result)
            
            # Aggregate timing data
            timing = result['timing']
            for key in total_timing.keys():
                if key == "total_cells":
                    total_timing[key] += timing["cells_processed"]
                elif key in timing:
                    total_timing[key] += timing[key]
            
            # Update progress bar if provided
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({'cells': timing["cells_processed"]}, refresh=True)
            
            # Add timing data for this image
            all_timing_data.append({
                "image_name": image_path.name,
                "cells_processed": timing["cells_processed"],
                "image_load_ms": timing["image_load"] * 1000,
                "yolo_detection_ms": timing["yolo_detection"] * 1000,
                "sam_preprocess_ms": timing["sam_preprocess"] * 1000,
                "sam_inference_ms": timing["sam_inference_total"] * 1000,
                "sam_postprocess_ms": timing["sam_postprocess_total"] * 1000,
                "metrics_calculation_ms": timing["metrics_total"] * 1000,
                "visualization_ms": timing["visualization"] * 1000,
                "total_time_ms": timing["total_time"] * 1000,
                "avg_time_per_cell_ms": (timing["total_time"] * 1000 / timing["cells_processed"]) if timing["cells_processed"] > 0 else 0
            })
            
            # Add metrics to CSV data
            for cell_idx, metrics in enumerate(result['cell_metrics']):
                metrics_row = {
                    'image_name': image_path.name,
                    'cell_id': cell_idx,
                    'deformability': metrics['deformability'],
                    'area': metrics['area'],
                    'area_ratio': metrics['area_ratio'],
                    'circularity': metrics['circularity'],
                    'convex_hull_area': metrics['convex_hull_area'],
                    'mask_x_length': metrics['mask_x_length'],
                    'mask_y_length': metrics['mask_y_length'],
                    'min_x': metrics['min_x'],
                    'min_y': metrics['min_y'],
                    'max_x': metrics['max_x'],
                    'max_y': metrics['max_y'],
                    'mean_brightness': metrics['mean_brightness'],
                    'brightness_std': metrics['brightness_std'],
                    'perimeter': metrics['perimeter'],
                    'aspect_ratio': metrics['aspect_ratio']
                }
                all_metrics_data.append(metrics_row)
        
        total_runtime = time.time() - start_time
        
        # Calculate and print aggregate statistics
        num_images = len(image_files)
        print("\n" + "=" * 80)
        print("PIPELINE PERFORMANCE SUMMARY")
        print("=" * 80)
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
        
        # Save metrics and timing data to CSV
        if all_metrics_data:
            metrics_df = pd.DataFrame(all_metrics_data)
            metrics_df.to_csv(output_dir / 'cell_metrics.csv', index=False)
        
        timing_df = pd.DataFrame(all_timing_data)
        timing_df.to_csv(output_dir / 'processing_times.csv', index=False)
        
        # Save comprehensive run information
        with open(output_dir / "run_summary.txt", "w") as f:
            f.write(f"Pipeline Run Summary\n")
            f.write(f"==================\n\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input Directory: {input_dir.absolute()}\n")
            f.write(f"Output Directory: {output_dir.absolute()}\n\n")
            
            f.write(f"Processing Statistics\n")
            f.write(f"====================\n")
            f.write(f"Total images processed: {num_images}\n")
            f.write(f"Total cells detected: {total_timing['total_cells']}\n")
            f.write(f"Average cells per image: {total_timing['total_cells']/num_images:.1f}\n\n")
            
            f.write(f"Timing Statistics (averaged per image)\n")
            f.write(f"===================================\n")
            f.write(f"Image loading: {(total_timing['image_load']/num_images)*1000:.1f}ms\n")
            f.write(f"YOLO detection: {(total_timing['yolo_detection']/num_images)*1000:.1f}ms\n")
            f.write(f"SAM preprocessing: {(total_timing['sam_preprocess']/num_images)*1000:.1f}ms\n")
            f.write(f"SAM inference: {(total_timing['sam_inference_total']/num_images)*1000:.1f}ms\n")
            f.write(f"SAM postprocessing: {(total_timing['sam_postprocess_total']/num_images)*1000:.1f}ms\n")
            f.write(f"Metrics calculation: {(total_timing['metrics_total']/num_images)*1000:.1f}ms\n")
            f.write(f"Visualization: {(total_timing['visualization']/num_images)*1000:.1f}ms\n\n")
            
            f.write(f"Overall Performance\n")
            f.write(f"==================\n")
            f.write(f"Total runtime: {total_runtime:.1f}s\n")
            f.write(f"Average time per image: {(total_runtime/num_images):.1f}s\n")
            if total_timing['total_cells'] > 0:
                f.write(f"Average time per cell: {(total_runtime/total_timing['total_cells'])*1000:.1f}ms\n")
        
        return results
    
    def process_single_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Path to save outputs
            save_visualizations: Whether to save visualization images
            
        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()
        
        # Read image
        image_load_start = time.time()
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        image_load_time = time.time() - image_load_start
        
        # Run YOLO detection
        yolo_start = time.time()
        yolo_results = self.yolo_model(image)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        yolo_time = time.time() - yolo_start
        
        cell_metrics = []
        masks = []
        raw_sam_masks = []
        
        # Get SAM transformed image once for visualization and reuse
        sam_preprocess_start = time.time()
        sam_inputs = self.sam_processor(image, return_tensors="pt")
        processed_pixel_values = sam_inputs['pixel_values'].to(self.device)
        sam_preprocess_time = time.time() - sam_preprocess_start
        
        # Process each detected cell
        sam_inference_total = 0
        sam_postprocess_total = 0
        metrics_total = 0
        
        for box in boxes:
            # Process box with SAM
            box_start = time.time()
            inputs = self.sam_processor(
                image,
                input_boxes=[[box.tolist()]],
                return_tensors="pt"
            )
            
            # Move inputs to device
            processed_boxes = inputs['input_boxes'].to(self.device)
            box_process_time = time.time() - box_start
            
            # Generate mask prediction
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.sam_model(
                    pixel_values=processed_pixel_values,  # Reuse processed image
                    input_boxes=processed_boxes,
                    multimask_output=False,
                )
            inference_time = time.time() - inference_start
            sam_inference_total += inference_time
            
            # Post-process masks
            postprocess_start = time.time()
            # Save raw SAM mask and ensure proper shape handling
            raw_mask = outputs.pred_masks.cpu()
            if len(raw_mask.shape) == 5:  # [B, max_masks, 1, H, W]
                raw_mask = raw_mask.squeeze(2)
            raw_mask = raw_mask[0].squeeze().numpy() > 0.5
            raw_sam_masks.append(raw_mask)
            
            # Post-process the masks
            processed_masks = self.sam_processor.post_process_masks(
                masks=outputs.pred_masks.cpu(),
                original_sizes=inputs["original_sizes"].cpu(),
                reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
                return_tensors="pt"
            )

            # Handle the list output from post_process_masks
            if isinstance(processed_masks, list):
                processed_masks = torch.stack(processed_masks, dim=0)
            
            # Convert mask to numpy and binarize
            mask = processed_masks[0].squeeze().numpy() > 0.5
            masks.append(mask)
            postprocess_time = time.time() - postprocess_start
            sam_postprocess_total += postprocess_time
            
            # Calculate metrics
            metrics_start = time.time()
            metrics = calculate_metrics(image, mask)
            cell_metrics.append(metrics)
            metrics_time = time.time() - metrics_start
            metrics_total += metrics_time
        
        # Save visualizations if requested
        vis_time = 0
        if save_visualizations:
            vis_start = time.time()
            self._save_visualizations(image, masks, boxes, cell_metrics, output_path)
            vis_time = time.time() - vis_start
        
        total_time = time.time() - start_time
            
        return {
            "image_path": str(image_path),
            "cell_metrics": cell_metrics,
            "num_cells": len(cell_metrics),
            "timing": {
                "image_load": image_load_time,
                "yolo_detection": yolo_time,
                "sam_preprocess": sam_preprocess_time,
                "sam_inference_total": sam_inference_total,
                "sam_postprocess_total": sam_postprocess_total,
                "metrics_total": metrics_total,
                "visualization": vis_time,
                "total_time": total_time,
                "cells_processed": len(boxes)
            }
        }
    
    def _save_visualizations(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        boxes: np.ndarray,
        cell_metrics: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save visualization of the segmentation results in separate folders using optimized TIFF format.
        
        Args:
            image: Original input image
            masks: List of processed masks
            boxes: YOLO detection boxes
            cell_metrics: List of metrics for each cell
            output_path: Path to save outputs
        """
        try:
            output_path = Path(output_path)
            base_dir = output_path.parent
            
            # Create separate directories for each output type
            dirs = {
                'original': base_dir / "1_original_images",
                'yolo': base_dir / "2_yolo_detections",
                'processed': base_dir / "3_processed_masks",
                'processed_masks': base_dir / "3_processed_masks/masks",
                'processed_overlays': base_dir / "3_processed_masks/overlay_images",
                'convex_hull': base_dir / "3_processed_masks/convex_hull_overlay",
                'combined': base_dir / "4_combined_visualization"
            }
            
            # Create all directories
            for dir_path in dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 1. Save original image
            save_optimized_tiff(
                image,
                dirs['original'] / f"{output_path.stem}_original.tiff",
                compression='zlib',
                compression_level=6
            )
            
            # 2. Save YOLO detections visualization
            yolo_vis = image.copy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(yolo_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            save_optimized_tiff(
                yolo_vis,
                dirs['yolo'] / f"{output_path.stem}_yolo.tiff",
                compression='zlib'
            )
            
            # 3. Save individual masks and overlays
            for i, (mask, metrics) in enumerate(zip(masks, cell_metrics)):
                # Save processed mask
                save_mask_as_tiff(
                    mask,
                    dirs['processed_masks'] / f"{output_path.stem}_mask_{i}.tiff"
                )
                
                # Create and save mask overlay
                overlay = image.copy()
                overlay[mask] = overlay[mask] * 0.7 + np.array([255, 0, 0]) * 0.3
                save_optimized_tiff(
                    overlay,
                    dirs['processed_overlays'] / f"{output_path.stem}_mask_{i}_overlay.tiff"
                )
                
                # Create and save convex hull overlay
                hull_overlay = image.copy()
                if 'convex_hull_coords' in metrics and len(metrics['convex_hull_coords']) > 0:
                    hull_coords = metrics['convex_hull_coords'].astype(np.int32)
                    cv2.polylines(hull_overlay, [hull_coords], True, (0, 255, 0), 2)
                    cv2.fillPoly(hull_overlay, [hull_coords], (0, 255, 0, 64))
                save_optimized_tiff(
                    hull_overlay,
                    dirs['convex_hull'] / f"{output_path.stem}_mask_{i}_convex_hull.tiff"
                )
            
            # 4. Save combined visualization
            combined_vis = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
            # Original with YOLO boxes
            combined_vis[:, :image.shape[1]] = yolo_vis
            # Masks and hulls overlay
            overlay_vis = image.copy()
            for i, (mask, metrics) in enumerate(zip(masks, cell_metrics)):
                # Add mask overlay
                overlay_vis[mask] = overlay_vis[mask] * 0.8 + np.array([255, 0, 0]) * 0.2
                # Add hull if available
                if 'convex_hull_coords' in metrics and len(metrics['convex_hull_coords']) > 0:
                    hull_coords = metrics['convex_hull_coords'].astype(np.int32)
                    cv2.polylines(overlay_vis, [hull_coords], True, (0, 255, 0), 2)
            combined_vis[:, image.shape[1]:] = overlay_vis
            
            save_optimized_tiff(
                combined_vis,
                dirs['combined'] / f"{output_path.stem}_combined.tiff",
                compression='zlib'
            )
     
            
        except Exception as e:
            print(f"Warning: Error during visualization saving: {str(e)}")
            # Continue processing even if visualization fails 