import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os

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

def load_first_image_and_coords(project_path):
    """Load the first available image and its coordinates from the project."""
    # Get first condition folder
    project_path = Path(project_path)
    condition_folders = [d for d in project_path.iterdir() if d.is_dir()]
    
    if not condition_folders:
        raise ValueError("No condition folders found")
        
    # Get first condition's data
    condition_folder = condition_folders[0]
    metrics_file = condition_folder / 'gated_cell_metrics.csv'
    
    if not metrics_file.exists():
        raise ValueError(f"No metrics file found in {condition_folder}")
        
    # Read CSV and get first row
    df = pd.read_csv(metrics_file)
    first_row = df.iloc[0]
    
    # Get image path
    image_path = get_image_path(project_path, condition_folder.name, first_row['image_name'])
    
    if not image_path:
        raise ValueError(f"Could not construct image path for {first_row['image_name']}")
        
    # Swap x and y coordinates from CSV since they are flipped
    return {
        'image_path': image_path,
        'min_x': first_row['min_y'],  # Swapped!
        'min_y': first_row['min_x'],  # Swapped!
        'max_x': first_row['max_y'],  # Swapped!
        'max_y': first_row['max_x'],  # Swapped!
        'image_name': first_row['image_name']
    }

def draw_rect(event, x, y, flags, param):
    """Mouse callback function for drawing rectangle."""
    global ix, iy, drawing, img, img_copy, rect_coords
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_copy = img.copy()
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        rect_coords = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

def main(project_path):
    global img, drawing, rect_coords
    
    # Load first image and its coordinates
    data = load_first_image_and_coords(project_path)
    print(f"Loading image: {data['image_path']}")
    print(f"CSV coordinates (after x/y swap): ({data['min_x']}, {data['min_y']}, {data['max_x']}, {data['max_y']})")
    print(f"Original CSV coordinates were: y1={data['min_y']}, x1={data['min_x']}, y2={data['max_y']}, x2={data['max_x']}")
    
    # Read image with OpenCV
    img = cv2.imread(str(data['image_path']))
    if img is None:
        raise ValueError(f"Could not load image: {data['image_path']}")
    
    img_copy = img.copy()
    drawing = False
    rect_coords = None
    
    # Create window and set mouse callback
    window_name = 'Debug Crop'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rect)
    
    print("\nInstructions:")
    print("1. Click and drag to draw a rectangle")
    print("2. Press 'r' to reset")
    print("3. Press 'c' to compare coordinates")
    print("4. Press 'q' to quit")
    
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):  # Reset
            img = img_copy.copy()
            rect_coords = None
        
        elif key == ord('c') and rect_coords:  # Compare coordinates
            print("\nComparison:")
            print(f"Your coordinates:   {rect_coords}")
            print(f"CSV coordinates:    ({data['min_x']}, {data['min_y']}, {data['max_x']}, {data['max_y']})")
            
            # Show both crops side by side
            your_crop = img_copy[rect_coords[1]:rect_coords[3], rect_coords[0]:rect_coords[2]]
            csv_crop = img_copy[int(data['min_y']):int(data['max_y']), 
                              int(data['min_x']):int(data['max_x'])]
            
            # Display crops
            cv2.imshow('Your Crop', your_crop)
            cv2.imshow('CSV Crop', csv_crop)
        
        elif key == ord('q'):  # Quit
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Debug image cropping')
    parser.add_argument('project_path', help='Path to the project folder')
    args = parser.parse_args()
    
    main(args.project_path) 