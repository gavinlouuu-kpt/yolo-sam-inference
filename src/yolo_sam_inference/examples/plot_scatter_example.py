# This script will plot an inferenced project as a scatter plot
# User may include whatever inferenced data into a folder as a project and point the script to that folder
# Within each 'condition/' folder there is a file called 'gated_cell_metrics.csv'
# This script will gather all the conditions within the project and plot [deformability] (y-axis) against the [convex_hull_area] (x-axis)
# The [condition] column will tell which condition folder the point belongs to and the [image_name] will tell which image the point belongs to
# The script will use bokeh to plot the scatter plot and allow for hover tooltips to show the cropped original image retrieved with image_name and condition.
# The image will be cropped with min_x, min_y, max_x, max_y from the gated_cell_metrics.csv file that are shown in matrix form.
# Use bokeh_example_script.py as a reference for the plot.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from bokeh.plotting import figure, show, save, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Spectral11
from bokeh.embed import file_html
from bokeh.resources import CDN
import glob

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
    
    # Get all condition folders
    condition_folders = [d for d in project_path.iterdir() if d.is_dir()]
    
    for condition_folder in condition_folders:
        condition_name = condition_folder.name
        metrics_file = condition_folder / 'gated_cell_metrics.csv'
        
        if metrics_file.exists():
            # Read the CSV file
            df = pd.read_csv(metrics_file)
            # Add condition column if not present
            if 'condition' not in df.columns:
                df['condition'] = condition_name
            all_data.append(df)
    
    # Combine all dataframes
    return pd.concat(all_data, ignore_index=True)

def get_cropped_image_base64(image_path, min_x, min_y, max_x, max_y):
    """Load and crop image, then convert to base64 for Bokeh tooltip."""
    try:
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        # Open and convert TIFF to ensure proper handling
        img = Image.open(image_path)
        
        # Print image details for debugging
        print(f"Processing image: {image_path}")
        print(f"Image mode: {img.mode}")
        print(f"Image size: {img.size}")
        
        # The coordinates in the CSV are flipped (x is y and y is x)
        # We need to swap them back
        min_x_img = int(min_y)  # CSV's min_y becomes image's min_x
        max_x_img = int(max_y)  # CSV's max_y becomes image's max_x
        min_y_img = int(min_x)  # CSV's min_x becomes image's min_y
        max_y_img = int(max_x)  # CSV's max_x becomes image's max_y
        
        print(f"Original CSV coordinates: x=({min_x}, {max_x}), y=({min_y}, {max_y})")
        print(f"Swapped coordinates for image: x=({min_x_img}, {max_x_img}), y=({min_y_img}, {max_y_img})")
        
        # Ensure coordinates are within bounds
        min_x_img = max(0, min(min_x_img, img.size[0]-1))
        max_x_img = max(min_x_img+1, min(max_x_img, img.size[0]))
        min_y_img = max(0, min(min_y_img, img.size[1]-1))
        max_y_img = max(min_y_img+1, min(max_y_img, img.size[1]))
        
        # Convert to RGB if needed
        if img.mode in ['I;16', 'I']:
            # For 16-bit images, normalize to 8-bit
            img_array = np.array(img)
            img_normalized = ((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min()))).astype(np.uint8)
            img = Image.fromarray(img_normalized, mode='L')
        
        # Convert grayscale to RGB
        if img.mode == 'L':
            img = Image.merge('RGB', (img, img, img))
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Crop the image with swapped coordinates
        cropped = img.crop((min_x_img, min_y_img, max_x_img, max_y_img))
        print(f"Cropped size: {cropped.size}")
        
        # Add padding to make the crop square if needed
        if cropped.size[0] != cropped.size[1]:
            max_dim = max(cropped.size)
            padded = Image.new('RGB', (max_dim, max_dim), (128, 128, 128))
            paste_x = (max_dim - cropped.size[0]) // 2
            paste_y = (max_dim - cropped.size[1]) // 2
            padded.paste(cropped, (paste_x, paste_y))
            cropped = padded
            print(f"Padded to square: {cropped.size}")
        
        # Resize if too large for tooltip - increased size
        max_size = (400, 400)  # Increased from 200x200
        cropped.thumbnail(max_size, Image.Resampling.LANCZOS)
        print(f"Final thumbnail size: {cropped.size}")
        
        # Debug: Save a test image to check content
        test_output_dir = os.path.join(os.path.dirname(image_path), "debug_crops")
        os.makedirs(test_output_dir, exist_ok=True)
        test_output_path = os.path.join(test_output_dir, f"crop_{os.path.basename(image_path)}")
        cropped.save(test_output_path)
        print(f"Saved debug crop to: {test_output_path}")
        
        # Convert to base64
        import base64
        from io import BytesIO
        buffer = BytesIO()
        cropped.save(buffer, format='PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_scatter_plot(project_path):
    """Create interactive scatter plot with image tooltips."""
    # Load data
    df = load_project_data(project_path)
    
    # Create color map for conditions
    conditions = df['condition'].unique()
    color_map = dict(zip(conditions, Spectral11[:len(conditions)]))
    df['color'] = df['condition'].map(color_map)
    
    # Add cropped image data
    df['image_data'] = df.apply(
        lambda row: get_cropped_image_base64(
            get_image_path(project_path, row['condition'], row['image_name']),
            row['min_x'], row['min_y'], row['max_x'], row['max_y']
        ),
        axis=1
    )
    
    # Remove rows where image processing failed
    df = df[df['image_data'].notna()]
    
    if df.empty:
        print("No valid data points with images found!")
        return
    
    # Create ColumnDataSource for Bokeh
    source = ColumnDataSource(df)
    
    # Set up the output file with a proper title
    output_file(os.path.join(project_path, 'scatter_plot.html'), 
                title="Cell Metrics Analysis")
    
    # Create figure
    p = figure(width=800, height=600, 
              title="Cell Metrics Scatter Plot",
              tools="pan,box_zoom,reset,save,wheel_zoom")
    p.xaxis.axis_label = 'Convex Hull Area'
    p.yaxis.axis_label = 'Deformability'
    
    # Add scatter points
    scatter = p.scatter(
        'convex_hull_area', 'deformability',
        size=8, alpha=0.6,
        color='color',
        legend_field='condition',
        source=source
    )
    
    # Configure legend
    p.legend.title = 'Conditions'
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    # Add hover tooltip with larger image and improved styling
    hover = HoverTool(
        renderers=[scatter],
        tooltips="""
        <div style="background-color: rgba(255, 255, 255, 0.95); padding: 10px; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.1);">
            <div style="text-align: center; margin-bottom: 10px;">
                <img src="@image_data" style="max-width: 400px; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <div style="margin: 5px 0;">
                <span style="font-size: 14px; color: #666; font-weight: bold;">Condition:</span>
                <span style="font-size: 14px;">@condition</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="font-size: 14px; color: #666; font-weight: bold;">Image:</span>
                <span style="font-size: 14px;">@image_name</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="font-size: 14px; color: #666; font-weight: bold;">Area:</span>
                <span style="font-size: 14px;">@convex_hull_area{0.00}</span>
            </div>
            <div style="margin: 5px 0;">
                <span style="font-size: 14px; color: #666; font-weight: bold;">Deformability:</span>
                <span style="font-size: 14px;">@deformability{0.00}</span>
            </div>
        </div>
        """
    )
    p.add_tools(hover)
    
    # Save the plot
    save(p)
    print(f"Plot saved to: {os.path.join(project_path, 'scatter_plot.html')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create scatter plot from cell metrics data')
    parser.add_argument('project_path', help='Path to the project folder containing condition folders')
    args = parser.parse_args()
    
    create_scatter_plot(args.project_path)

