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
