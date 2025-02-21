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
from datetime import datetime
import uuid
import pandas as pd

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
        
        # Initialize YOLO model
        self.yolo_model = YOLO(yolo_model_path)
        
        # Initialize SAM model and processor from HuggingFace
        self.sam_model = SamModel.from_pretrained(sam_model_type).to(device)
        self.sam_processor = SamProcessor.from_pretrained(sam_model_type)
        
        # Generate a unique run ID for this pipeline instance
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
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
        output_dir = Path(output_dir) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images first to gather statistics
        results = []
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + \
                     list(input_dir.glob("*.tiff"))
        
        # Prepare data for CSV
        all_metrics_data = []
        
        for image_path in image_files:
            result = self.process_single_image(
                image_path,
                output_dir / image_path.name,
                save_visualizations
            )
            results.append(result)
            
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
        
        # Save metrics to CSV
        if all_metrics_data:
            metrics_df = pd.DataFrame(all_metrics_data)
            metrics_df.to_csv(output_dir / 'cell_metrics.csv', index=False)
        
        # Calculate run statistics
        total_cells = sum(result['num_cells'] for result in results)
        avg_cells_per_image = total_cells / len(results) if results else 0
        
        # Aggregate cell metrics across all images
        all_metrics = []
        for result in results:
            all_metrics.extend(result['cell_metrics'])
            
        # Calculate aggregate statistics if we have any cells
        if all_metrics:
            avg_area = sum(m['area'] for m in all_metrics) / len(all_metrics)
            avg_circularity = sum(m['circularity'] for m in all_metrics) / len(all_metrics)
            avg_brightness = sum(m['mean_brightness'] for m in all_metrics) / len(all_metrics)
            avg_perimeter = sum(m['perimeter'] for m in all_metrics) / len(all_metrics)
        
        # Save comprehensive run information
        with open(output_dir / "run_info.txt", "w") as f:
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Input Directory: {input_dir.absolute()}\n")
            f.write(f"Images Processed: {len(results)}\n")
            f.write(f"Total Cells: {total_cells}\n")
            f.write(f"Average Cells/Image: {avg_cells_per_image:.1f}\n\n")
            
            if all_metrics:
                f.write("Average Cell Metrics:\n")
                f.write(f"  Area: {avg_area:.1f} pixels\n")
                f.write(f"  Circularity: {avg_circularity:.3f}\n")
                f.write(f"  Brightness: {avg_brightness:.1f}\n")
                f.write(f"  Perimeter: {avg_perimeter:.1f} pixels\n")
        
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
            self._save_visualizations(image, masks, boxes, cell_metrics, output_path)
            
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
        cell_metrics: List[Dict[str, Any]],
        output_path: Union[str, Path]
    ) -> None:
        """
        Save visualization of the segmentation results in separate folders.
        
        Args:
            image: Original input image
            masks: List of processed masks
            boxes: YOLO detection boxes
            cell_metrics: List of metrics for each cell
            output_path: Path to save outputs
        """
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
        
        # Define color scheme for masks and convex hulls
        colors = [
            ('Reds', '#00ff00'),      # Red mask, Green hull
            ('Blues', '#ff8c00'),     # Blue mask, Orange hull
            ('Greens', '#ff00ff'),    # Green mask, Magenta hull
            ('Purples', '#ffff00'),   # Purple mask, Yellow hull
            ('Oranges', '#00ffff'),   # Orange mask, Cyan hull
        ]
        
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
        
        # 2. Save YOLO detections
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
        
        # 3. Save individual masks and overlays
        for i, (mask, metrics) in enumerate(zip(masks, cell_metrics)):
            mask_cmap, hull_color = colors[i % len(colors)]
            
            # Save processed mask
            plt.figure(figsize=(10, 10))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.savefig(str(dirs['processed_masks'] / f"{output_path.stem}_mask_{i}.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # Save mask overlay on original image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.3, cmap=mask_cmap)
            plt.axis('off')
            plt.savefig(str(dirs['processed_overlays'] / f"{output_path.stem}_mask_{i}_overlay.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
            
            # Save convex hull overlay
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # Plot the mask with lower alpha
            plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.2, cmap=mask_cmap)
            # Plot the convex hull outline with higher alpha and thicker line
            if 'convex_hull_coords' in metrics and len(metrics['convex_hull_coords']) > 0:
                hull_coords = metrics['convex_hull_coords']
                plt.plot(hull_coords[:, 1], hull_coords[:, 0], color=hull_color, 
                        linewidth=3, alpha=1.0, label=f'Convex Hull {i+1}')
                # Fill the convex hull with a different pattern
                plt.fill(hull_coords[:, 1], hull_coords[:, 0], color=hull_color, 
                        alpha=0.1, hatch='///')
            plt.legend()
            plt.axis('off')
            plt.savefig(str(dirs['convex_hull'] / f"{output_path.stem}_mask_{i}_convex_hull.png"), 
                       bbox_inches='tight', dpi=150)
            plt.close()
        
        # 4. Save combined visualization
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
        
        # All masks and convex hulls overlay
        plt.subplot(122)
        plt.imshow(image)
        legend_elements = []
        for i, (mask, metrics) in enumerate(zip(masks, cell_metrics)):
            mask_cmap, hull_color = colors[i % len(colors)]
            # Plot mask with lower alpha
            plt.imshow(np.ma.masked_where(~mask, mask), alpha=0.2, cmap=mask_cmap)
            # Plot convex hull with higher alpha and pattern
            if 'convex_hull_coords' in metrics and len(metrics['convex_hull_coords']) > 0:
                hull_coords = metrics['convex_hull_coords']
                # Fill the convex hull with a pattern
                plt.fill(hull_coords[:, 1], hull_coords[:, 0], color=hull_color, 
                        alpha=0.1, hatch='///')
                # Draw the outline
                hull_line = plt.plot(hull_coords[:, 1], hull_coords[:, 0], 
                                   color=hull_color, linewidth=3, alpha=1.0)[0]
                legend_elements.append((f'Cell {i+1} Mask', plt.Rectangle((0,0), 1, 1, fc=plt.cm.get_cmap(mask_cmap)(0.5), alpha=0.2)))
                legend_elements.append((f'Cell {i+1} Hull', hull_line))
        
        if legend_elements:
            plt.legend([patch if isinstance(patch, plt.Rectangle) else line 
                       for patch, line in legend_elements],
                      [label for label, _ in legend_elements],
                      loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Masks and Convex Hulls Overlay')
        plt.axis('off')
        
        plt.savefig(str(dirs['combined'] / f"{output_path.stem}_combined.png"), 
                   bbox_inches='tight', dpi=150, bbox_extra_artists=[patch for _, patch in legend_elements])
        plt.close() 