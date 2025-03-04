# Deformability Training Data Preparation

This tool helps prepare training data for machine learning models by grouping cells based on their deformability characteristics.

## Overview

The `deformability_training_data.py` script:

1. Loads cell metrics data from an inferenced project
2. Groups cells into 5 percentile categories based on deformability
3. Extracts and saves cropped cell images into category-specific folders
4. Converts all images to optimized PNG format
5. Creates a metadata file with all cell information

## Usage

```bash
python deformability_training_data.py /path/to/project/folder [--output-dir /path/to/output/directory]
```

### Arguments

- `project_path`: Path to the project folder containing condition folders (required)
- `--output-dir`: Path to the output directory for training data (optional, default: project_path/training_data)

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── very_low_deformability/
│   ├── condition1_image1_cell0.png
│   ├── condition2_image3_cell5.png
│   └── ...
├── low_deformability/
│   └── ...
├── medium_deformability/
│   └── ...
├── high_deformability/
│   └── ...
├── very_high_deformability/
│   └── ...
└── metadata.csv
```

Each image is named using the pattern: `{condition}_{image_name}_cell{id}.png`

The `metadata.csv` file contains all the original cell metrics data with additional columns:
- `deformability_percentile`: The percentile group (0-4)
- `deformability_group`: The descriptive name of the percentile group

## Example

```bash
# Basic usage
python deformability_training_data.py /data/my_cell_project

# Specify custom output directory
python deformability_training_data.py /data/my_cell_project --output-dir /data/training_sets/deformability_dataset
```

## Requirements

This script requires:
- Python 3.6+
- pandas
- numpy
- Pillow (PIL)
- pathlib

## Notes

- The script expects each condition folder to contain a `gated_cell_metrics.csv` file
- Images are expected to be in a timestamp subfolder under each condition folder
- The script will automatically find the timestamp folder and locate the original images
- Images are cropped around the cell with a 2x expansion factor to include context
- All images are converted to PNG format with optimized settings for machine learning (RGB mode, compression level 6)
- Original TIFF images are not modified 