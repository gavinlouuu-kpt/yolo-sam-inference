from pathlib import Path
from typing import List, Dict, Union, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from transformers import SamModel, SamProcessor
from .utils import calculate_metrics
import matplotlib.pyplot as plt
from skimage import measure

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
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize SAM model and processor from HuggingFace
        self.sam_model = SamModel.from_pretrained(sam_model_type).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_type)
    
    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        save_visualizations: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_visualizations: Whether to save visualization images
            
        Returns:
            List of dictionaries containing results for each image
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.tiff"))
        
        for image_path in image_files:
            result = self.process_single_image(
                image_path,
                output_dir / image_path.name,
                save_visualizations
            )
            results.append(result)
        
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
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Run YOLO detection
        yolo_results = self.yolo_model(image)[0]
        boxes = yolo_results.boxes.xyxy.cpu().numpy()
        
        cell_metrics = []
        masks = []
        raw_sam_masks = []
        
        # Get SAM transformed image once for visualization
        sam_inputs = self.sam_processor(image, return_tensors="pt")
        sam_transformed_image = sam_inputs['pixel_values'].squeeze().permute(1, 2, 0).cpu().numpy()
        
        # Process each detected cell
        for box in boxes:
            # Get input size before processing
            inputs = self.sam_processor(
                image,
                input_boxes=[[box.tolist()]],
                return_tensors="pt"
            )
            
            # Get the processed image dimensions
            processed_pixel_values = inputs['pixel_values'].to(self.device)
            processed_boxes = inputs['input_boxes'].to(self.device)
            input_size = processed_pixel_values.shape[-2:]  # (H, W)
            
            # Generate mask prediction
            with torch.no_grad():
                outputs = self.sam_model(
                    pixel_values=processed_pixel_values,
                    input_boxes=processed_boxes,
                    multimask_output=False,
                )
            
            # Save raw SAM mask and ensure proper shape handling
            raw_mask = outputs.pred_masks.cpu()
            if len(raw_mask.shape) == 5:  # [B, max_masks, 1, H, W]
                raw_mask = raw_mask.squeeze(2)
            raw_mask = raw_mask[0].squeeze().numpy() > 0.5
            raw_sam_masks.append(raw_mask)
            
            # Post-process the masks with proper shape handling
            # Get the input image size from processed_pixel_values
            input_h, input_w = processed_pixel_values.shape[-2:]
            
            processed_masks = self.sam_processor.post_process_masks(
                masks=outputs.pred_masks.cpu(),
                original_sizes=inputs["original_sizes"].cpu(),  # Use sizes from processor
                reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),  # Use sizes from processor
                return_tensors="pt"
            )

            # Handle the list output from post_process_masks
            if isinstance(processed_masks, list):
                # Convert list of tensors to a single tensor
                processed_masks = torch.stack(processed_masks, dim=0)
            
            # Convert mask to numpy and binarize with proper threshold
            mask = processed_masks[0]  # Take first mask since multimask_output=False
            if len(mask.shape) > 2:  # Remove any extra dimensions
                mask = mask.squeeze()
            mask = mask.numpy() > 0.5  # Binarize with 0.5 threshold
            masks.append(mask)
            
            # Calculate metrics using the original image and properly processed mask
            metrics = calculate_metrics(image, mask)
            cell_metrics.append(metrics)
        
        # Save all visualizations if requested
        if save_visualizations:
            self._save_visualizations(image, masks, boxes, output_path)
            
        return {
            "image_path": str(image_path),
            "cell_metrics": cell_metrics,
            "num_cells": len(cell_metrics)
        }
    
    def _save_visualizations(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        boxes: np.ndarray,
        output_path: Union[str, Path]
    ) -> None:
        """
        Save visualization of the segmentation results in separate folders.
        
        Args:
            image: Original input image
            masks: List of processed masks
            boxes: YOLO detection boxes
            output_path: Path to save outputs
        """
        output_path = Path(output_path)
        base_dir = output_path.parent
        
        # Create separate directories for each output type
        dirs = {
            'original': base_dir / "1_original_images",
            'yolo': base_dir / "2_yolo_detections",
            'processed': base_dir / "3_processed_masks",
            'combined': base_dir / "4_combined_visualization"
        }
        
        # Create all directories
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 1. Save original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.savefig(str(dirs['original'] / f"{output_path.stem}_original.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3. Save YOLO detections
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for box in boxes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=1, alpha=0.3))
        plt.axis('off')
        plt.savefig(str(dirs['yolo'] / f"{output_path.stem}_yolo.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        
        # 4 & 5. Save individual masks (both raw and processed)
        for i, mask in enumerate(masks):
            # Save processed mask
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.savefig(str(dirs['processed'] / f"{output_path.stem}_mask_{i}.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # Save mask overlay on original image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.3, cmap='Reds')
            plt.axis('off')
            plt.savefig(str(dirs['processed'] / f"{output_path.stem}_mask_{i}_overlay.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 6. Save combined visualization
        plt.figure(figsize=(20, 10))
        
        # Original with YOLO boxes
        plt.subplot(121)
        plt.imshow(image)
        for box in boxes:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color='red', linewidth=1, alpha=0.3))
        plt.title('Original Image with YOLO Detections')
        plt.axis('off')
        
        # All masks overlay
        plt.subplot(122)
        plt.imshow(image)
        colors = [(1,0,0,0.2), (0,1,0,0.2), (0,0,1,0.2)]
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.3, cmap=plt.cm.get_cmap('Reds'))
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            for contour in contours:
                plt.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1, alpha=0.3)
        plt.title('All Masks Overlay')
        plt.axis('off')
        
        plt.savefig(str(dirs['combined'] / f"{output_path.stem}_combined.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close() 