# TIFF to PNG Conversion Tool

This directory contains utility tools for the YOLO-SAM inference pipeline.

## tiff2png.py

A script to convert TIFF images to PNG format. It supports recursive directory traversal to process nested folders and includes progress tracking with tqdm.

### Requirements

- Python 3.6+
- Pillow (PIL Fork)
- tqdm (for progress bars)

You can install the required packages with:

```bash
pip install pillow tqdm
```

### Features

- Convert TIFF images (.tif, .tiff, .TIF, .TIFF) to PNG format
- Process directories recursively with the `--recursive` flag
- Specify custom output directory
- Maintain directory structure when using output directory
- Progress tracking with tqdm progress bars
- Detailed logging with the `--verbose` flag

### Usage

Basic usage:

```bash
python tiff2png.py /path/to/tiff/directory
```

With recursive flag to process subdirectories:

```bash
python tiff2png.py /path/to/tiff/directory --recursive
```

Specify an output directory:

```bash
python tiff2png.py /path/to/tiff/directory --output /path/to/output/directory
```

Enable verbose output:

```bash
python tiff2png.py /path/to/tiff/directory --verbose
```

### Command-line Arguments

- `directory`: Directory containing TIFF images (required)
- `--recursive`, `-r`: Process subdirectories recursively
- `--output`, `-o`: Output directory for PNG images
- `--verbose`, `-v`: Enable verbose output

### Examples

Convert all TIFF images in the current directory:

```bash
python tiff2png.py .
```

Convert all TIFF images in a directory and its subdirectories:

```bash
python tiff2png.py /data/images --recursive
```

Convert all TIFF images and save them to a specific output directory:

```bash
python tiff2png.py /data/images --output /data/converted
```

When using the `--output` option with `--recursive`, the script will maintain the same directory structure in the output directory. 