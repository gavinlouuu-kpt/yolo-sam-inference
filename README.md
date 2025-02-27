This pipeline will take a path as input (containing images .png, .tiff, .jpg etc.) and return the following:
1. Through yolo will return A list of bounding boxes (essentially identifying background and images with target)
2. The bounding box will be used as box prompt in sam (the bounding box will be paired with image)

The output of this package will give:
1. Segmented mask of all the images in a folder
2. Original image cropped by bounding box
3. Metrics including the following: Mask area, mask circularity, mask deformability, convex hull of the mask, distribution of pixel intensity within the mask area from the original image

## Interactive Visualization

The package includes an interactive scatter plot visualization tool that allows you to explore the cell metrics data. The scatter plot is generated using Bokeh and provides the following features:

### Scatter Plot Tool (`plot_scatter_example.py`)

This tool creates an interactive HTML visualization that plots cell metrics with the following features:

- **Plot Configuration**:
  - X-axis: Convex Hull Area
  - Y-axis: Deformability
  - Points are colored by condition
  - Density-based transparency shows clustering of cells

- **Interactive Features**:
  - Hover tooltips showing:
    - Large (600x600) cell image crop
    - Condition name
    - Image name
    - Area and deformability values
  - Pan and zoom capabilities
  - Legend with hide/show functionality for each condition
  - Save plot functionality

### Usage

To generate the scatter plot, run:

```bash
python -m yolo_sam_inference.examples.plot_scatter_example /path/to/project/folder
```

The script will:
1. Load data from all condition folders in the project
2. Process the gated_cell_metrics.csv files
3. Generate an interactive HTML plot saved as scatter_plot.html

### Project Structure Requirements

Your project folder should have the following structure:
```
project_folder/
├── condition_a/
│   ├── gated_cell_metrics.csv
│   └── timestamp_folder/
│       └── 1_original_images/
│           └── *_original.tiff
├── condition_b/
│   ├── gated_cell_metrics.csv
│   └── timestamp_folder/
│       └── 1_original_images/
│           └── *_original.tiff
└── scatter_plot.html (generated)
```

The `gated_cell_metrics.csv` files should contain columns for:
- convex_hull_area
- deformability
- image_name
- min_x, min_y, max_x, max_y (cropping coordinates)

### Output

The tool generates a self-contained HTML file (`scatter_plot.html`) that can be opened in any modern web browser. The visualization is fully interactive and doesn't require any additional software to view.

## YOLO Frame Cleaner Tool

The package includes a frame processing tool (`yolo_frame_cleaner.py`) that helps clean and organize image datasets using YOLO object detection. This tool is particularly useful for preparing training data by filtering frames based on specific criteria within a user-defined Region of Interest (ROI).

### Features

- **Interactive ROI Selection**: Users can draw a rectangular ROI on the first frame
- **Intelligent Frame Filtering**:
  - Detects objects using YOLO model
  - Accepts only frames with exactly one detection fully within the ROI
  - Rejects frames where detections touch ROI boundaries
  - Saves one background frame (without detections) for reference
- **Output Organization**:
  - Preserves original filenames with descriptive suffixes
  - Saves both full frames and ROI-cropped versions
  - Includes debug visualizations for verification

### Usage

```bash
python -m yolo_sam_inference.examples.yolo_frame_cleaner \
    --input-dir /path/to/input/frames \
    --output-dir /path/to/output \
    --experiment-id "YOUR_MLFLOW_EXPERIMENT_ID" \
    --run-id "YOUR_MLFLOW_RUN_ID" \
    --device cuda  # or cpu
```

#### Arguments

- `--input-dir`, `-i`: Directory containing input frames (required)
- `--output-dir`, `-o`: Output directory (defaults to input_dir + "_output")
- `--experiment-id`: MLflow experiment ID for loading YOLO model
- `--run-id`: MLflow run ID for loading YOLO model
- `--device`: Device to run inference on ('cuda' or 'cpu')

### Output Structure

```
output_dir/
├── full_frames_with_target/
│   ├── original_name_with_target.jpg
│   └── original_name_background.jpg
├── full_frames_with_target_png/
│   ├── original_name_with_target.png
│   └── original_name_background.png
├── cropped_roi_with_target/
│   ├── original_name_with_target.jpg
│   └── original_name_background.jpg
├── cropped_roi_with_target_png/
│   ├── original_name_with_target.png
│   └── original_name_background.png
└── debug_visualizations/
    └── debug_original_name_detections.jpg
```

The tool saves outputs in both the original format and PNG format:
- `full_frames_with_target/`: Full frames in original format
- `full_frames_with_target_png/`: Full frames in PNG format
- `cropped_roi_with_target/`: Cropped ROIs in original format
- `cropped_roi_with_target_png/`: Cropped ROIs in PNG format
- `debug_visualizations/`: Debug images with color-coded detections

### Debug Visualizations

The tool generates debug visualizations with color-coded bounding boxes:
- **Green**: Valid detections (fully contained within ROI)
- **Yellow**: Detections touching ROI boundary (rejected)
- **Red**: Outside ROI or low confidence detections
- **Blue**: ROI boundary
