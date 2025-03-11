#!/usr/bin/env python3
"""
Script to convert TIFF images to PNG format.
Supports recursive directory traversal with the --recursive flag.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_tiff_to_png(tiff_path, output_dir=None):
    """
    Convert a TIFF image to PNG format.
    
    Args:
        tiff_path (Path): Path to the TIFF image
        output_dir (Path, optional): Directory to save the PNG image. If None,
                                    saves in the same directory as the TIFF.
    
    Returns:
        Path: Path to the saved PNG image
    """
    try:
        # Open the TIFF image
        with Image.open(tiff_path) as img:
            # Determine output path
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                png_path = output_dir / f"{tiff_path.stem}.png"
            else:
                png_path = tiff_path.with_suffix('.png')
            
            # Save as PNG
            img.save(png_path, "PNG")
            logger.debug(f"Converted: {tiff_path} -> {png_path}")
            return png_path
    except Exception as e:
        logger.error(f"Error converting {tiff_path}: {e}")
        return None

def find_all_tiff_files(directory, recursive=False):
    """
    Find all TIFF files in a directory and optionally its subdirectories.
    
    Args:
        directory (Path): Directory to search
        recursive (bool): Whether to search subdirectories
    
    Returns:
        list: List of Path objects for all TIFF files found
    """
    directory = Path(directory)
    if not directory.exists() or not directory.is_dir():
        logger.error(f"Directory not found: {directory}")
        return []
    
    tiff_extensions = ['.tif', '.tiff', '.TIF', '.TIFF']
    tiff_files = []
    
    # Find all TIFF files in the current directory
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix in tiff_extensions:
            tiff_files.append(file_path)
    
    # Find TIFF files in subdirectories if recursive flag is set
    if recursive:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                tiff_files.extend(find_all_tiff_files(subdir, recursive))
    
    return tiff_files

def process_directory(directory, recursive=False, output_dir=None):
    """
    Process all TIFF images in a directory.
    
    Args:
        directory (Path): Directory containing TIFF images
        recursive (bool): Whether to process subdirectories
        output_dir (Path, optional): Directory to save PNG images
    
    Returns:
        int: Number of successfully converted images
    """
    # Find all TIFF files
    tiff_files = find_all_tiff_files(directory, recursive)
    
    if not tiff_files:
        logger.info(f"No TIFF files found in {directory}")
        return 0
    
    logger.info(f"Found {len(tiff_files)} TIFF files to convert")
    
    # Process files with progress bar
    converted_count = 0
    for tiff_path in tqdm(tiff_files, desc="Converting", unit="file"):
        # Determine output directory structure if specified
        if output_dir:
            # Maintain directory structure relative to input directory
            rel_path = tiff_path.parent.relative_to(directory)
            file_output_dir = Path(output_dir) / rel_path
        else:
            file_output_dir = None
        
        # Convert the file
        if convert_tiff_to_png(tiff_path, file_output_dir):
            converted_count += 1
    
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="Convert TIFF images to PNG format")
    parser.add_argument("directory", help="Directory containing TIFF images")
    parser.add_argument("--recursive", "-r", action="store_true", 
                        help="Process subdirectories recursively")
    parser.add_argument("--output", "-o", help="Output directory for PNG images")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Process the directory
    start_dir = Path(args.directory)
    logger.info(f"Processing directory: {start_dir}")
    logger.info(f"Recursive mode: {'enabled' if args.recursive else 'disabled'}")
    
    converted = process_directory(start_dir, args.recursive, args.output)
    
    logger.info(f"Conversion complete. {converted} images converted.")

if __name__ == "__main__":
    main()
