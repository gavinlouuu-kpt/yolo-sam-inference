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
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Spectral11
from bokeh.embed import file_html
from bokeh.resources import CDN
from scipy.stats import gaussian_kde
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

def get_cropped_image_base64(image_path, min_x, min_y, max_x, max_y):
    """Load and crop image, then convert to base64 for Bokeh tooltip."""
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
        
        # Resize for tooltip (larger size)
        max_size = (600, 600)
        cropped.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        import base64
        from io import BytesIO
        buffer = BytesIO()
        cropped.save(buffer, format='PNG', optimize=True)
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception:
        return None

def create_scatter_plot(project_path):
    """Create interactive scatter plot with image tooltips and density coloring."""
    # Load data
    df = load_project_data(project_path)
    
    # Print debugging information
    print("\nData Summary:")
    print(f"Total number of rows: {len(df)}")
    print("\nConditions found:")
    for condition in df['condition'].unique():
        count = len(df[df['condition'] == condition])
        print(f"- {condition}: {count} cells")
    
    # Create color map for conditions
    conditions = df['condition'].unique()
    color_map = dict(zip(conditions, Spectral11[:len(conditions)]))
    print("\nColor mapping:")
    for condition, color in color_map.items():
        print(f"- {condition}: {color}")
    
    # Set up the output file
    output_file(os.path.join(project_path, 'scatter_plot.html'), 
                title="Cell Metrics Analysis")
    
    # Create figure
    p = figure(width=800, height=600,
              title="Cell Metrics Scatter Plot",
              tools="pan,box_zoom,reset,save,wheel_zoom")
    p.xaxis.axis_label = 'Convex Hull Area'
    p.yaxis.axis_label = 'Deformability'
    
    # Create hover tool first (we'll use the same one for all renderers)
    hover = HoverTool(
        tooltips="""
        <div style="background-color: rgba(255, 255, 255, 0.98); padding: 15px; border-radius: 8px; box-shadow: 0 2px 15px rgba(0,0,0,0.15); max-width: 650px;">
            <div style="text-align: center; margin-bottom: 15px;">
                <img src="@image_data" style="max-width: 600px; width: 100%; height: auto; border: 2px solid #eee; border-radius: 8px;">
            </div>
            <div style="margin: 8px 0;">
                <span style="font-size: 15px; color: #555; font-weight: bold;">Condition:</span>
                <span style="font-size: 15px;">@condition</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="font-size: 15px; color: #555; font-weight: bold;">Image:</span>
                <span style="font-size: 15px;">@image_name</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="font-size: 15px; color: #555; font-weight: bold;">Area:</span>
                <span style="font-size: 15px;">@convex_hull_area{0.00}</span>
            </div>
            <div style="margin: 8px 0;">
                <span style="font-size: 15px; color: #555; font-weight: bold;">Deformability:</span>
                <span style="font-size: 15px;">@deformability{0.00}</span>
            </div>
        </div>
        """
    )
    p.add_tools(hover)
    
    # Store all renderers to attach to hover tool
    renderers = []
    
    # Plot each condition separately with density coloring
    for condition in conditions:
        print(f"\nProcessing condition: {condition}")
        condition_data = df[df['condition'] == condition].copy()
        print(f"Number of cells in condition: {len(condition_data)}")
        
        # Calculate 2D KDE for density coloring
        if len(condition_data) > 5:  # Need enough points for meaningful KDE
            x = condition_data['convex_hull_area'].values
            y = condition_data['deformability'].values
            xy = np.vstack([x, y])
            
            try:
                kde = gaussian_kde(xy)
                density = kde(xy)
                
                # Normalize density to [0.2, 0.8] range for alpha
                min_density = density.min()
                max_density = density.max()
                if min_density != max_density:
                    alpha_values = 0.2 + 0.6 * (density - min_density) / (max_density - min_density)
                else:
                    alpha_values = 0.6 * np.ones_like(density)
                
                # Add alpha values to the dataframe
                condition_data['point_alpha'] = alpha_values
                
                print("Adding image data...")
                # Add cropped image data for this condition
                condition_data['image_data'] = condition_data.apply(
                    lambda row: get_cropped_image_base64(
                        get_image_path(project_path, row['condition'], row['image_name']),
                        row['min_x'], row['min_y'], row['max_x'], row['max_y']
                    ),
                    axis=1
                )
                
                # Remove rows where image processing failed
                condition_data = condition_data[condition_data['image_data'].notna()]
                print(f"Cells with valid images: {len(condition_data)}")
                
                if not condition_data.empty:
                    source = ColumnDataSource(condition_data)
                    
                    # Add scatter points with density-based alpha from the source
                    scatter = p.scatter(
                        'convex_hull_area', 'deformability',
                        size=8,
                        color=color_map[condition],
                        alpha='point_alpha',
                        legend_label=condition,
                        source=source,
                        name=condition  # Add name for identification
                    )
                    renderers.append(scatter)
                    print(f"Added scatter plot for {condition}")
                else:
                    print(f"Warning: No valid data points for {condition}")
            
            except np.linalg.LinAlgError:
                print(f"KDE failed for {condition}, using simple scatter plot")
                # Fallback to simple scatter plot if KDE fails
                condition_data['point_alpha'] = 0.6
                condition_data['image_data'] = condition_data.apply(
                    lambda row: get_cropped_image_base64(
                        get_image_path(project_path, row['condition'], row['image_name']),
                        row['min_x'], row['min_y'], row['max_x'], row['max_y']
                    ),
                    axis=1
                )
                condition_data = condition_data[condition_data['image_data'].notna()]
                if not condition_data.empty:
                    source = ColumnDataSource(condition_data)
                    scatter = p.scatter(
                        'convex_hull_area', 'deformability',
                        size=8,
                        color=color_map[condition],
                        alpha='point_alpha',
                        legend_label=condition,
                        source=source,
                        name=condition
                    )
                    renderers.append(scatter)
        else:
            print(f"Warning: Not enough points for KDE in {condition}")
    
    # Update hover tool with all renderers
    hover.renderers = renderers
    
    # Configure legend
    p.legend.title = 'Conditions'
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    # Save the plot
    save(p)
    print(f"\nPlot saved to: {os.path.join(project_path, 'scatter_plot.html')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create scatter plot from cell metrics data')
    parser.add_argument('project_path', help='Path to the project folder containing condition folders')
    args = parser.parse_args()
    
    create_scatter_plot(args.project_path)

