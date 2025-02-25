from pathlib import Path
from typing import List, Dict, Union, Any, Optional, Tuple
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
from dataclasses import dataclass
import logging

# Set up logger
logger = logging.getLogger(__name__)

# Filter out specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

@dataclass
class ProcessingResult:
    """Data class to store processing results for a single image."""
    image_path: str
    cell_metrics: List[Dict[str, Any]]
    num_cells: int
    timing: Dict[str, float]

@dataclass
class BatchProcessingResult:
    """Data class to store processing results for a batch of images."""
    results: List[ProcessingResult]
    total_timing: Dict[str, float]
    metrics_data: List[Dict[str, Any]]
    timing_data: List[Dict[str, Any]]

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
        
        self._initialize_models(yolo_model_path)
        self.run_id = self._generate_run_id()
    
    def _initialize_models(self, yolo_model_path: Union[str, Path]) -> None:
        """Initialize YOLO and SAM models."""
        # Initialize YOLO model with verbose=False to disable progress output
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.args['verbose'] = False  # Set verbose in args dictionary
        
        # Initialize SAM model and processor from HuggingFace
        self.sam_model = SamModel.from_pretrained(self.sam_model_type).to(self.device)
        self.sam_processor = SamProcessor.from_pretrained(self.sam_model_type)
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID for this pipeline instance."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _detect_cells(self, image: np.ndarray) -> np.ndarray:
        """Run YOLO detection on an image."""
        yolo_results = self.yolo_model(image)[0]
        return yolo_results.boxes.xyxy.cpu().numpy()
    
    def _process_sam_mask(
        self,
        image: np.ndarray,
        box: np.ndarray,
        processed_pixel_values: torch.Tensor
    ) -> Tuple[np.ndarray, float, float]:
        """Process a single box with SAM model."""
        # Prepare inputs
        inputs = self.sam_processor(
            image,
            input_boxes=[[box.tolist()]],
            return_tensors="pt"
        )
        processed_boxes = inputs['input_boxes'].to(self.device)

        # Generate mask prediction
        with torch.no_grad():
            outputs = self.sam_model(
                pixel_values=processed_pixel_values,
                input_boxes=processed_boxes,
                multimask_output=False,
            )

        # Post-process masks
        processed_masks = self.sam_processor.post_process_masks(
            masks=outputs.pred_masks.cpu(),
            original_sizes=inputs["original_sizes"].cpu(),
            reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
            return_tensors="pt"
        )

        if isinstance(processed_masks, list):
            processed_masks = torch.stack(processed_masks, dim=0)

        mask = processed_masks[0].squeeze().numpy() > 0.5
        return mask
    
    def process_single_image(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        save_visualizations: bool = True
    ) -> ProcessingResult:
        """
        Process a single image through the pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Path to save outputs
            save_visualizations: Whether to save visualization images
            
        Returns:
            Dictionary containing processing results
        """
        timings = {}
        
        # Load and preprocess image
        start_time = time.time()
        image = self._load_image(str(image_path))
        timings['image_load'] = time.time() - start_time

        # Detect cells with YOLO
        start_time = time.time()
        boxes = self._detect_cells(image)
        timings['yolo_detection'] = time.time() - start_time

        # Initialize empty lists for results
        masks = []
        cell_metrics = []
        sam_times = {'inference': 0.0, 'postprocess': 0.0}

        # Only process with SAM if YOLO detected cells
        if len(boxes) > 0:
            logger.info(f"YOLO detected {len(boxes)} cells in {Path(image_path).name}")
            # Prepare SAM inputs
            start_time = time.time()
            sam_inputs = self.sam_processor(image, return_tensors="pt")
            processed_pixel_values = sam_inputs['pixel_values'].to(self.device)
            timings['sam_preprocess'] = time.time() - start_time

            # Process each detected cell
            for box in boxes:
                mask = self._process_sam_mask(image, box, processed_pixel_values)
                masks.append(mask)
                
                metrics = calculate_metrics(image, mask)
                cell_metrics.append(metrics)
        else:
            logger.info(f"No cells detected by YOLO in {Path(image_path).name} - skipping SAM processing")
            # No cells detected, set SAM preprocessing time to 0
            timings['sam_preprocess'] = 0.0

        timings.update(sam_times)

        # Save visualizations if requested and there are detections
        if save_visualizations:
            start_time = time.time()
            self._save_visualizations(image, masks, boxes, cell_metrics, output_path)
            timings['visualization'] = time.time() - start_time

        # Add total time and cells processed
        total_time = time.time() - start_time
        timings.update({
            'total_time': total_time,
            'cells_processed': len(boxes)
        })

        # Log processing summary
        logger.info(f"Processed {Path(image_path).name}: {len(boxes)} cells detected, total time: {total_time:.2f}s")

        return ProcessingResult(
            image_path=str(image_path),
            cell_metrics=cell_metrics,
            num_cells=len(cell_metrics),
            timing=timings
        )
    
    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        """Load and preprocess an image."""
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualizations: bool = True,
        pbar: Optional[tqdm] = None
    ) -> BatchProcessingResult:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            pbar: Optional tqdm progress bar for tracking progress
            
        Returns:
            BatchProcessingResult containing results for all images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = self._get_image_files(input_dir)
        
        # Process images and collect results
        results = []
        metrics_data = []
        timing_data = []
        total_timing = self._initialize_timing_dict()

        for image_path in image_files:
            result = self.process_single_image(
                image_path,
                output_dir / image_path.name,
                save_visualizations
            )
            results.append(result)
            
            # Update progress and collect data
            self._update_progress(pbar, result)
            self._collect_metrics_data(metrics_data, result)
            self._collect_timing_data(timing_data, result)
            self._update_total_timing(total_timing, result.timing)

        return BatchProcessingResult(
            results=results,
            total_timing=total_timing,
            metrics_data=metrics_data,
            timing_data=timing_data
        )
    
    @staticmethod
    def _get_image_files(directory: Path) -> List[Path]:
        """Get all image files from a directory."""
        return list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + \
               list(directory.glob("*.tiff"))
    
    @staticmethod
    def _initialize_timing_dict() -> Dict[str, float]:
        """Initialize timing dictionary with zero values."""
        return {
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
    
    @staticmethod
    def _update_progress(pbar: Optional[tqdm], result: ProcessingResult) -> None:
        """Update progress bar if provided."""
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({'cells': result.num_cells}, refresh=True)
    
    @staticmethod
    def _collect_metrics_data(
        metrics_data: List[Dict[str, Any]],
        result: ProcessingResult
    ) -> None:
        """Collect metrics data from processing result."""
        for cell_idx, metrics in enumerate(result.cell_metrics):
            metrics_row = {
                'image_name': Path(result.image_path).name,
                'cell_id': cell_idx,
                **metrics
            }
            metrics_data.append(metrics_row)
    
    @staticmethod
    def _collect_timing_data(
        timing_data: List[Dict[str, Any]],
        result: ProcessingResult
    ) -> None:
        """Collect timing data from processing result."""
        timing_data.append({
            'image_name': Path(result.image_path).name,
            'cells_processed': result.timing["cells_processed"],
            **{f"{k}_ms": v * 1000 for k, v in result.timing.items() if k != "cells_processed"}
        })
    
    @staticmethod
    def _update_total_timing(
        total_timing: Dict[str, float],
        timing: Dict[str, float]
    ) -> None:
        """Update total timing dictionary with new timing data."""
        for key in total_timing:
            if key == "total_cells":
                total_timing[key] += timing["cells_processed"]
            elif key in timing:
                total_timing[key] += timing[key]
    
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

class ParallelCellSegmentationPipeline:
    def __init__(
        self,
        yolo_model_path: Union[str, Path],
        sam_model_type: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        num_pipelines: int = 2
    ):
        """
        Initialize multiple cell segmentation pipelines for parallel processing.
        
        Args:
            yolo_model_path: Path to the YOLO model weights
            sam_model_type: HuggingFace model identifier for SAM
            device: Device to run models on ('cuda' or 'cpu')
            num_pipelines: Number of parallel pipelines to create
        """
        self.device = device
        self.sam_model_type = sam_model_type
        self.num_pipelines = num_pipelines
        
        # Create multiple pipeline instances
        self.pipelines = [
            CellSegmentationPipeline(yolo_model_path, sam_model_type, device)
            for _ in range(num_pipelines)
        ]
        
        self.run_id = self._generate_run_id()
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID for this pipeline instance."""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualizations: bool = True,
        pbar: Optional[tqdm] = None
    ) -> BatchProcessingResult:
        """
        Process all images in a directory using multiple pipelines in parallel.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            pbar: Optional tqdm progress bar for tracking progress
            
        Returns:
            BatchProcessingResult containing results for all images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = self._get_image_files(input_dir)
        
        # Split images among pipelines
        from concurrent.futures import ThreadPoolExecutor
        import math
        
        batch_size = math.ceil(len(image_files) / self.num_pipelines)
        image_batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        # Process batches in parallel
        results = []
        metrics_data = []
        timing_data = []
        total_timing = self._initialize_timing_dict()
        
        def process_batch(pipeline, batch):
            batch_results = []
            for image_path in batch:
                result = pipeline.process_single_image(
                    image_path,
                    output_dir / image_path.name,
                    save_visualizations
                )
                batch_results.append(result)
                if pbar:
                    pbar.update(1)
            return batch_results
        
        with ThreadPoolExecutor(max_workers=self.num_pipelines) as executor:
            futures = [
                executor.submit(process_batch, pipeline, batch)
                for pipeline, batch in zip(self.pipelines, image_batches)
            ]
            
            # Collect results from all futures
            for future in futures:
                batch_results = future.result()
                results.extend(batch_results)
                
                # Collect metrics and timing data
                for result in batch_results:
                    self._collect_metrics_data(metrics_data, result)
                    self._collect_timing_data(timing_data, result)
                    self._update_total_timing(total_timing, result.timing)
        
        return BatchProcessingResult(
            results=results,
            total_timing=total_timing,
            metrics_data=metrics_data,
            timing_data=timing_data
        )
    
    @staticmethod
    def _get_image_files(directory: Path) -> List[Path]:
        """Get all image files from a directory."""
        return list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + \
               list(directory.glob("*.tiff"))
    
    @staticmethod
    def _initialize_timing_dict() -> Dict[str, float]:
        """Initialize timing dictionary with zero values."""
        return {
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
    
    @staticmethod
    def _collect_metrics_data(
        metrics_data: List[Dict[str, Any]],
        result: ProcessingResult
    ) -> None:
        """Collect metrics data from processing result."""
        for cell_idx, metrics in enumerate(result.cell_metrics):
            metrics_row = {
                'image_name': Path(result.image_path).name,
                'cell_id': cell_idx,
                **metrics
            }
            metrics_data.append(metrics_row)
    
    @staticmethod
    def _collect_timing_data(
        timing_data: List[Dict[str, Any]],
        result: ProcessingResult
    ) -> None:
        """Collect timing data from processing result."""
        timing_data.append({
            'image_name': Path(result.image_path).name,
            'cells_processed': result.timing["cells_processed"],
            **{f"{k}_ms": v * 1000 for k, v in result.timing.items() if k != "cells_processed"}
        })
    
    @staticmethod
    def _update_total_timing(
        total_timing: Dict[str, float],
        timing: Dict[str, float]
    ) -> None:
        """Update total timing dictionary with new timing data."""
        for key in total_timing:
            if key == "total_cells":
                total_timing[key] += timing["cells_processed"]
            elif key in timing:
                total_timing[key] += timing[key] 