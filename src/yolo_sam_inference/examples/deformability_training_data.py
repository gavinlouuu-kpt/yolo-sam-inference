#!/usr/bin/env python3
# This script will process an inferenced project, group cells by deformability percentiles,
# and create a training data folder with images from each percentile group.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import shutil
import argparse

# Reuse functions from plot_scatter_example.py
def find_timestamp_folder(condition_path):
    """Find the timestamp folder within a condition directory."""
    timestamp_folders = list(Path(condition_path).glob("2*"))
    if timestamp_folders:
        return timestamp_folders[0]
    return None

def get_image_path(project_path, condition, image_name):
    """Construct the correct image path based on the actual directory structure."""
    condition_path = os.path.join(project_path, condition)
    timestamp_folder = find_timestamp_folder(condition_path)
    if timestamp_folder:
        # Convert image name from CSV to actual image name format
        base_name = os.path.splitext(image_name)[0]  # Remove .png
        image_path = os.path.join(timestamp_folder, "1_original_images", f"{base_name}_original.tiff")
        return image_path
    return None

def load_project_data(project_path):
    """Load data from all condition folders in the project."""
    project_path = Path(project_path)
    all_data = []
    
    print("\nScanning project directory:")
    print(f"Project path: {project_path}")
    
    # List all items in the project directory
    all_items = list(project_path.iterdir())
    print("\nAll items found in directory:")
    for item in all_items:
        print(f"- {item.name} ({'directory' if item.is_dir() else 'file'})")
    
    # Get all condition folders (excluding timestamp folders)
    condition_folders = [d for d in all_items if d.is_dir() and not d.name.startswith('202')]
    
    print("\nIdentified condition folders:")
    for folder in condition_folders:
        print(f"- {folder.name}")
    
    for condition_folder in condition_folders:
        condition_name = condition_folder.name
        metrics_file = condition_folder / 'gated_cell_metrics.csv'
        
        print(f"\nProcessing condition: {condition_name}")
        print(f"Looking for metrics file: {metrics_file}")
        
        if metrics_file.exists():
            print(f"Found metrics file for {condition_name}")
            try:
                # Read the CSV file
                df = pd.read_csv(metrics_file)
                print(f"Loaded {len(df)} rows from {condition_name}")
                
                # Explicitly set the condition column
                df['condition'] = condition_name
                print(f"Set condition column to: {condition_name}")
                
                # Verify condition column
                unique_conditions = df['condition'].unique()
                print(f"Verified conditions in DataFrame: {unique_conditions}")
                
                all_data.append(df)
            except Exception as e:
                print(f"Error processing {condition_name}: {str(e)}")
        else:
            print(f"Warning: No metrics file found for condition {condition_name}")
    
    if not all_data:
        raise ValueError("No data found in any condition folder!")
    
    # Combine all dataframes and verify the result
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Debug information about the combined dataset
    print(f"\nCombined dataset information:")
    print(f"Total rows: {len(combined_df)}")
    print("Rows per condition:")
    for condition in combined_df['condition'].unique():
        count = len(combined_df[combined_df['condition'] == condition])
        print(f"- {condition}: {count} rows")
    
    return combined_df

def get_cropped_image(image_path, min_x, min_y, max_x, max_y):
    """Load and crop image, then return the PIL Image object."""
    try:
        if not os.path.exists(image_path):
            return None
            
        img = Image.open(image_path)
        
        # The coordinates in the CSV are flipped (x is y and y is x)
        min_x_img = int(min_y)  # CSV's min_y becomes image's min_x
        max_x_img = int(max_y)  # CSV's max_y becomes image's max_x
        min_y_img = int(min_x)  # CSV's min_x becomes image's min_y
        max_y_img = int(max_x)  # CSV's max_x becomes image's max_y
        
        # Calculate the center of the bounding box
        center_x = (min_x_img + max_x_img) // 2
        center_y = (min_y_img + max_y_img) // 2
        
        # Calculate the size of the original bounding box
        width = max_x_img - min_x_img
        height = max_y_img - min_y_img
        
        # Expand the crop area by 2x
        expand_factor = 2.0
        new_width = int(width * expand_factor)
        new_height = int(height * expand_factor)
        
        # Calculate new coordinates while keeping the center
        min_x_img = center_x - (new_width // 2)
        max_x_img = center_x + (new_width // 2)
        min_y_img = center_y - (new_height // 2)
        max_y_img = center_y + (new_height // 2)
        
        # Ensure coordinates are within bounds
        min_x_img = max(0, min(min_x_img, img.size[0]-1))
        max_x_img = max(min_x_img+1, min(max_x_img, img.size[0]))
        min_y_img = max(0, min(min_y_img, img.size[1]-1))
        max_y_img = max(min_y_img+1, min(max_y_img, img.size[1]))
        
        # Convert to RGB if needed
        if img.mode in ['I;16', 'I']:
            img_array = np.array(img)
            img_normalized = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(np.uint8)
            img = Image.fromarray(img_normalized, mode='L')
        
        if img.mode == 'L':
            img = Image.merge('RGB', (img, img, img))
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Crop the image with expanded coordinates
        cropped = img.crop((min_x_img, min_y_img, max_x_img, max_y_img))
        
        return cropped
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def save_as_png(image, output_path):
    """Save image as PNG with optimized settings."""
    try:
        # Ensure the image is in RGB mode for consistent PNG output
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save with optimized settings for training data
        image.save(
            output_path, 
            format='PNG',
            optimize=True,
            compress_level=6  # Balance between file size and quality (0-9)
        )
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {str(e)}")
        return False

def create_training_data(project_path, output_dir=None):
    """
    Create training data by grouping cells by deformability percentiles.
    
    Args:
        project_path: Path to the project folder containing condition folders
        output_dir: Path to the output directory for training data (default: project_path/training_data)
    """
    # Load data
    df = load_project_data(project_path)
    
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.join(project_path, 'training_data')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate deformability percentiles
    print("\nCalculating deformability percentiles...")
    df['deformability_percentile'] = pd.qcut(df['deformability'], 5, labels=False)
    
    # Create a mapping from percentile to group name
    percentile_groups = {
        0: 'very_low_deformability',
        1: 'low_deformability',
        2: 'medium_deformability',
        3: 'high_deformability',
        4: 'very_high_deformability'
    }
    
    # Map percentile to group name
    df['deformability_group'] = df['deformability_percentile'].map(percentile_groups)
    
    # Print summary of percentile groups
    print("\nDeformability percentile groups:")
    for group_id, group_name in percentile_groups.items():
        group_df = df[df['deformability_percentile'] == group_id]
        min_val = group_df['deformability'].min()
        max_val = group_df['deformability'].max()
        count = len(group_df)
        print(f"- {group_name}: {count} cells, deformability range: {min_val:.4f} to {max_val:.4f}")
    
    # Create directories for each percentile group
    for group_name in percentile_groups.values():
        group_dir = os.path.join(output_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)
    
    # Process each row and save cropped images to appropriate directories
    print("\nProcessing images and saving to training data directories...")
    processed_count = 0
    skipped_count = 0
    
    for _, row in df.iterrows():
        # Get image path
        image_path = get_image_path(project_path, row['condition'], row['image_name'])
        if image_path is None:
            skipped_count += 1
            continue
        
        # Get cropped image
        cropped_img = get_cropped_image(
            image_path,
            row['min_x'], row['min_y'], row['max_x'], row['max_y']
        )
        
        if cropped_img is None:
            skipped_count += 1
            continue
        
        # Get group name and create output path
        group_name = row['deformability_group']
        group_dir = os.path.join(output_dir, group_name)
        
        # Create a unique filename
        condition = row['condition']
        image_name = os.path.splitext(row['image_name'])[0]
        cell_id = processed_count  # Use a counter as a unique identifier
        output_filename = f"{condition}_{image_name}_cell{cell_id}.png"
        output_path = os.path.join(group_dir, output_filename)
        
        # Save the image as PNG
        if save_as_png(cropped_img, output_path):
            processed_count += 1
            
            # Print progress every 100 images
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images...")
        else:
            skipped_count += 1
    
    # Print summary
    print("\nTraining data creation complete!")
    print(f"Total processed images: {processed_count}")
    print(f"Total skipped images: {skipped_count}")
    print(f"Training data saved to: {output_dir}")
    print(f"All images saved in PNG format")
    
    # Create a metadata CSV file with information about each image
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create training data from cell metrics by deformability percentiles')
    parser.add_argument('project_path', help='Path to the project folder containing condition folders')
    parser.add_argument('--output-dir', help='Path to the output directory for training data (default: project_path/training_data)')
    args = parser.parse_args()
    
    create_training_data(args.project_path, args.output_dir) 