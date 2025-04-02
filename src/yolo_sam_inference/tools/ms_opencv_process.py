''' DO NOT REMOVE THIS COMMENT
given a path take the same logic as mib_batch_readout to search for images.bin, background_clean.tiff and roi.csv in each batch folder
the images.bin can be parsed by the following function:

void convertSavedImagesToStandardFormat(const std::string &binaryImageFile, const std::string &outputDirectory)
{
    std::ifstream imageFile(binaryImageFile, std::ios::binary);
    std::filesystem::create_directories(outputDirectory);

    int imageCount = 0;
    while (imageFile.good())
    {
        int rows, cols, type;
        imageFile.read(reinterpret_cast<char *>(&rows), sizeof(int));
        imageFile.read(reinterpret_cast<char *>(&cols), sizeof(int));
        imageFile.read(reinterpret_cast<char *>(&type), sizeof(int));

        if (imageFile.eof())
            break;

        cv::Mat image(rows, cols, type);
        imageFile.read(reinterpret_cast<char *>(image.data), rows * cols * image.elemSize());

        std::string outputPath = outputDirectory + "/image_" + std::to_string(imageCount++) + ".tiff";
        cv::imwrite(outputPath, image);
    }

    std::cout << "Converted " << imageCount << " images to TIFF format in " << outputDirectory << std::endl;
}

we do not actually want to save the images to tiff, but just use them as input for the OpenCV pipeline

the roi.csv is in the following format:
x,y,width,height

all the images (target and background) read from images.bin will be cropped to the roi and then processed by the OpenCV pipeline as cropped images similar to image_processing_core.cpp

the output of the pipeline should be a csv file with deformability and area from all batches saved at project-dir

the metrics calculated will always use convex hull parameters

'''

import os
import cv2
import numpy as np
import pandas as pd
import struct
import glob
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json


def read_images_bin(file_path: str) -> List[np.ndarray]:
    """
    Read images from the binary format exactly as implemented in image_processing_utils.cpp
    
    Args:
        file_path: Path to the images.bin file
    
    Returns:
        List of OpenCV images read from the binary file
    """
    images = []
    
    with open(file_path, 'rb') as f:
        image_count = 0
        while True:
            # Read metadata: rows, cols, type
            header_data = f.read(3 * 4)  # 3 integers (4 bytes each)
            if len(header_data) < 12:  # End of file
                break
                
            rows, cols, cv_type = struct.unpack('<iii', header_data)
            
            print(f"Image metadata: rows={rows}, cols={cols}, type={cv_type}")
            
            # Sanity check for unreasonable dimensions
            if rows <= 0 or cols <= 0 or rows > 10000 or cols > 10000:
                print(f"Warning: Skipping image with unusual dimensions: {rows}x{cols}")
                break
                
            # Calculate element size according to OpenCV's Mat.elemSize() function
            # In OpenCV type encoding:
            # - bits 0-2: depth (0=CV_8U, 1=CV_8S, 2=CV_16U, 3=CV_16S, 4=CV_32S, 5=CV_32F, 6=CV_64F)
            # - bits 3-11: number of channels (shifted by 3)
            depth = cv_type & 7
            channels = (cv_type >> 3) + 1  # OpenCV type channels are 1-indexed (stored as n-1)
            
            print(f"Decoded: depth={depth}, channels={channels}")
            
            # Calculate element size (bytes per pixel)
            if depth == 0:  # CV_8U
                bytes_per_element = 1
            elif depth == 1:  # CV_8S
                bytes_per_element = 1
            elif depth == 2:  # CV_16U
                bytes_per_element = 2
            elif depth == 3:  # CV_16S
                bytes_per_element = 2
            elif depth == 4:  # CV_32S
                bytes_per_element = 4
            elif depth == 5:  # CV_32F
                bytes_per_element = 4
            elif depth == 6:  # CV_64F
                bytes_per_element = 8
            else:
                print(f"Warning: Unsupported OpenCV depth: {depth}")
                break
                
            # Total element size = bytes per element * channels
            elem_size = bytes_per_element * channels
            
            # Calculate total data size (rows * cols * elemSize)
            data_size = rows * cols * elem_size
            
            print(f"Calculated data size: {data_size} bytes (elem_size={elem_size})")
            
            if data_size <= 0:
                print(f"Error: Invalid data size: {data_size} bytes")
                break
                
            try:
                # Read image data directly into a NumPy array
                data = f.read(data_size)
                
                if len(data) < data_size:
                    print(f"Warning: Incomplete image data. Expected {data_size} bytes, got {len(data)}")
                    break
                    
                # Create the image based on type
                if depth == 0:  # CV_8U
                    dtype = np.uint8
                elif depth == 1:  # CV_8S
                    dtype = np.int8
                elif depth == 2:  # CV_16U
                    dtype = np.uint16
                elif depth == 3:  # CV_16S
                    dtype = np.int16
                elif depth == 4:  # CV_32S
                    dtype = np.int32
                elif depth == 5:  # CV_32F
                    dtype = np.float32
                elif depth == 6:  # CV_64F
                    dtype = np.float64
                else:
                    print(f"Warning: Unsupported depth {depth}, skipping image")
                    continue
                
                # Create the image - note we need to match OpenCV's memory layout 
                img_data = np.frombuffer(data, dtype=dtype)
                
                # Reshape based on channels
                if channels == 1:
                    img = img_data.reshape((rows, cols))
                else:
                    img = img_data.reshape((rows, cols, channels))
                
                images.append(img)
                image_count += 1
                print(f"Successfully read image {image_count}: shape={img.shape}, dtype={img.dtype}")
                
            except Exception as e:
                print(f"Error reading image data: {str(e)}")
                break
    
    print(f"Total images read: {len(images)}")
    return images


def read_roi_csv(file_path: str) -> Tuple[int, int, int, int]:
    """
    Read ROI from CSV file in the format: x,y,width,height
    
    Args:
        file_path: Path to the ROI CSV file
    
    Returns:
        Tuple of (x, y, width, height)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            # Check if the first line contains headers
            first_line = lines[0].strip()
            if first_line.lower().startswith('x') or ',' in first_line and any(h.lower() in ['x', 'y', 'width', 'height'] for h in first_line.split(',')):
                # Skip header line
                data_line = lines[1].strip() if len(lines) > 1 else ""
            else:
                data_line = first_line
                
            if not data_line:
                print(f"Warning: No data found in ROI file {file_path}")
                return 0, 0, -1, -1
                
            # Parse the data line
            values = data_line.split(',')
            if len(values) < 4:
                print(f"Warning: Not enough values in ROI file {file_path}")
                return 0, 0, -1, -1
                
            x, y, width, height = map(int, values[:4])
            return x, y, width, height
    except Exception as e:
        print(f"Error reading ROI file {file_path}: {str(e)}")
        # Return a default ROI (full image)
        return 0, 0, -1, -1


def process_frame(target_image: np.ndarray, background_image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Process a frame using OpenCV operations, similar to image_processing_core.cpp
    
    Args:
        target_image: The target image to process
        background_image: The background image for subtraction
        config: Configuration parameters for processing
    
    Returns:
        Processed binary image
    """
    # Apply Gaussian blur to reduce noise (same as applied to background)
    blurred_target = cv2.GaussianBlur(
        target_image, 
        (config['gaussian_blur_size'], config['gaussian_blur_size']), 
        0
    )
    
    # Apply simple contrast enhancement if enabled
    if config['enable_contrast_enhancement']:
        # Use the formula: new_pixel = alpha * old_pixel + beta
        enhanced = cv2.convertScaleAbs(
            blurred_target, 
            alpha=config['contrast_alpha'], 
            beta=config['contrast_beta']
        )
        
        # Perform background subtraction with the enhanced image
        bg_sub = cv2.subtract(enhanced, background_image)
    else:
        # Simple background subtraction without contrast enhancement
        bg_sub = cv2.subtract(blurred_target, background_image)
    
    # Apply threshold to create binary image
    _, binary = cv2.threshold(
        bg_sub, 
        config['bg_subtract_threshold'], 
        255, 
        cv2.THRESH_BINARY
    )
    
    # Create structural element for morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, 
        (config['morph_kernel_size'], config['morph_kernel_size'])
    )
    
    # Apply morphological operations to clean up the image
    # First, close operation (dilate then erode)
    morphed = cv2.morphologyEx(
        binary, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=config['morph_iterations']
    )
    
    # Then, open operation (erode then dilate)
    morphed = cv2.morphologyEx(
        morphed, 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=config['morph_iterations']
    )
    
    return morphed


def find_contours(processed_image: np.ndarray) -> Tuple[List[np.ndarray], bool, List[np.ndarray]]:
    """
    Find contours in the processed image, similar to the C++ function
    
    Args:
        processed_image: Binary processed image
    
    Returns:
        Tuple of (filtered_contours, has_nested_contours, inner_contours)
    """
    # Find contours
    contours, hierarchy = cv2.findContours(
        processed_image, 
        cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter out small noise contours
    filtered_contours = []
    filtered_hierarchy = []
    
    # Minimum area threshold to filter out noise
    min_noise_area = 10.0
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area >= min_noise_area:
            filtered_contours.append(contour)
            if hierarchy is not None and i < len(hierarchy[0]):
                filtered_hierarchy.append(hierarchy[0][i])
    
    # Check if there are nested contours by examining the hierarchy
    has_nested_contours = False
    inner_contours = []
    
    # Process hierarchy to find inner contours
    if len(filtered_hierarchy) > 0:
        for i, h in enumerate(filtered_hierarchy):
            # h[3] > -1 means this contour has a parent (it's an inner contour)
            if h[3] > -1:
                has_nested_contours = True
                inner_contours.append(filtered_contours[i])
    
    return filtered_contours, has_nested_contours, inner_contours


def calculate_metrics(contour: np.ndarray) -> Tuple[float, float]:
    """
    Calculate deformability and area metrics for a contour
    
    Args:
        contour: Contour points
    
    Returns:
        Tuple of (deformability, area)
    """
    # Calculate moments
    m = cv2.moments(contour)
    area = m['m00']
    
    # Calculate perimeter
    perimeter = cv2.arcLength(contour, True)
    
    # Calculate circularity: sqrt(4 * pi * area) / perimeter
    if perimeter > 0:
        circularity = math.sqrt(4 * math.pi * area) / perimeter
    else:
        circularity = 0.0
        
    # Calculate deformability: 1.0 - circularity
    deformability = 1.0 - circularity
    
    return deformability, area


def filter_processed_image(processed_image: np.ndarray, config: Dict[str, Any]) -> Tuple[float, float]:
    """
    Filter the processed image and extract metrics, similar to the C++ function
    
    Args:
        processed_image: Binary processed image
        config: Configuration parameters for filtering
    
    Returns:
        Tuple of (deformability, area) or (0, 0) if invalid
    """
    # Find contours
    contours, has_nested_contours, inner_contours = find_contours(processed_image)
    
    # Check if we have a single inner contour
    has_single_inner_contour = len(inner_contours) == 1
    
    # If we require a single inner contour and don't have exactly one, return invalid result
    if config['require_single_inner_contour'] and not has_single_inner_contour:
        return 0.0, 0.0
    
    # If we have a single inner contour, use it for metrics
    if has_single_inner_contour:
        # Calculate contour area
        contour_area = cv2.contourArea(inner_contours[0])
        
        # Calculate convex hull
        hull = cv2.convexHull(inner_contours[0])
        hull_area = cv2.contourArea(hull)
        
        # Calculate metrics
        deformability, area = calculate_metrics(inner_contours[0])
        
        # Check area range if enabled
        if not config['enable_area_range_check'] or (
            area >= config['area_threshold_min'] and 
            area <= config['area_threshold_max']
        ):
            return deformability, area
    
    # If no inner contours but we have contours, use the largest one
    elif contours and not config['require_single_inner_contour']:
        # Find the largest contour
        largest_idx = 0
        largest_area = 0.0
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_idx = i
        
        # Calculate metrics for the largest contour
        deformability, area = calculate_metrics(contours[largest_idx])
        
        # Check area range if enabled
        if not config['enable_area_range_check'] or (
            area >= config['area_threshold_min'] and 
            area <= config['area_threshold_max']
        ):
            return deformability, area
    
    # Return invalid result
    return 0.0, 0.0


def read_batch_config(batch_dir: str, default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read batch-specific processing configuration if available
    
    Args:
        batch_dir: Path to the batch directory
        default_config: Default configuration to fall back on
        
    Returns:
        Configuration dictionary for the batch
    """
    config_path = os.path.join(batch_dir, 'processing_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                batch_config = json.load(f)
            
            # Create a new config dictionary based on the default
            config = default_config.copy()
            
            # Update with batch-specific values if they exist
            if 'gaussian_blur_size' in batch_config:
                config['gaussian_blur_size'] = batch_config['gaussian_blur_size']
            if 'bg_subtract_threshold' in batch_config:
                config['bg_subtract_threshold'] = batch_config['bg_subtract_threshold']
            if 'morph_kernel_size' in batch_config:
                config['morph_kernel_size'] = batch_config['morph_kernel_size']
            if 'morph_iterations' in batch_config:
                config['morph_iterations'] = batch_config['morph_iterations']
                
            # Handle filters and contrast enhancement if they exist in a nested structure
            if 'filters' in batch_config:
                filters = batch_config['filters']
                if 'enable_border_check' in filters:
                    config['enable_border_check'] = filters['enable_border_check']
                if 'enable_area_range_check' in filters:
                    config['enable_area_range_check'] = filters['enable_area_range_check']
                if 'require_single_inner_contour' in filters:
                    config['require_single_inner_contour'] = filters['require_single_inner_contour']
                    
            if 'area_threshold_min' in batch_config:
                config['area_threshold_min'] = batch_config['area_threshold_min']
            if 'area_threshold_max' in batch_config:
                config['area_threshold_max'] = batch_config['area_threshold_max']
                
            if 'contrast_enhancement' in batch_config:
                contrast = batch_config['contrast_enhancement']
                if 'enable_contrast' in contrast:
                    config['enable_contrast_enhancement'] = contrast['enable_contrast']
                if 'alpha' in contrast:
                    config['contrast_alpha'] = contrast['alpha']
                if 'beta' in contrast:
                    config['contrast_beta'] = contrast['beta']
            
            print(f"Using batch-specific configuration from {config_path}")
            return config
        except Exception as e:
            print(f"Error reading batch configuration: {str(e)}")
            print(f"Using default configuration")
            return default_config
    else:
        print(f"No batch-specific configuration found, using default")
        return default_config


def process_batch(batch_dir: str, default_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process a single batch directory
    
    Args:
        batch_dir: Path to the batch directory
        default_config: Default processing configuration
    
    Returns:
        List of dictionaries with metrics from processed images
    """
    results = []
    
    # Find images.bin, background_clean.tiff, and roi.csv
    images_bin_path = os.path.join(batch_dir, 'images.bin')
    background_path = os.path.join(batch_dir, 'background_clean.tiff')
    roi_path = os.path.join(batch_dir, 'roi.csv')
    
    # Check if required files exist
    if not os.path.exists(images_bin_path):
        print(f"Missing images.bin in {batch_dir}")
        return results
    
    # Read batch-specific configuration
    batch_config = read_batch_config(batch_dir, default_config)
    
    # Read ROI if available
    if os.path.exists(roi_path):
        x, y, width, height = read_roi_csv(roi_path)
        print(f"Using ROI from {roi_path}: x={x}, y={y}, width={width}, height={height}")
    else:
        # Default ROI (full image)
        x, y, width, height = 0, 0, -1, -1
        print(f"No ROI file found at {roi_path}, using full image")
    
    # Read background image if available
    if os.path.exists(background_path):
        background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
        print(f"Using background from {background_path}")
    else:
        background = None
        print(f"Warning: No background image found at {background_path}")
        return results
    
    # Process the batch efficiently
    print(f"Processing batch: {batch_dir}")
    print("Reading and processing binary image file...")
    
    # Extract batch name from directory
    batch_name = os.path.basename(batch_dir)
    
    # Process images directly from binary file to avoid loading all 17000 images into memory
    with open(images_bin_path, 'rb') as f:
        image_index = 0
        processed_count = 0
        
        # Loop until end of file
        while True:
            # Read metadata: rows, cols, type
            header_data = f.read(3 * 4)  # 3 integers (4 bytes each)
            if len(header_data) < 12:  # End of file
                break
                
            rows, cols, cv_type = struct.unpack('<iii', header_data)
            
            # Sanity check for unreasonable dimensions
            if rows <= 0 or cols <= 0 or rows > 10000 or cols > 10000:
                print(f"Warning: Skipping image with unusual dimensions: {rows}x{cols}")
                continue
                
            # Calculate element size according to OpenCV's Mat.elemSize() function
            depth = cv_type & 7
            channels = (cv_type >> 3) + 1  # OpenCV type channels are 1-indexed (stored as n-1)
            
            # Calculate element size (bytes per pixel)
            if depth == 0:      # CV_8U
                bytes_per_element = 1
            elif depth == 1:    # CV_8S
                bytes_per_element = 1
            elif depth == 2:    # CV_16U
                bytes_per_element = 2
            elif depth == 3:    # CV_16S
                bytes_per_element = 2
            elif depth == 4:    # CV_32S
                bytes_per_element = 4
            elif depth == 5:    # CV_32F
                bytes_per_element = 4
            elif depth == 6:    # CV_64F
                bytes_per_element = 8
            else:
                print(f"Warning: Unsupported OpenCV depth: {depth}")
                # Skip this image by reading its data
                dummy_data_size = rows * cols * bytes_per_element * channels
                f.read(dummy_data_size)
                image_index += 1
                continue
                
            # Total element size = bytes per element * channels
            elem_size = bytes_per_element * channels
            
            # Calculate total data size (rows * cols * elemSize)
            data_size = rows * cols * elem_size
            
            if data_size <= 0:
                print(f"Error: Invalid data size: {data_size} bytes")
                image_index += 1
                continue
            
            try:
                # Read image data directly into a NumPy array
                data = f.read(data_size)
                
                if len(data) < data_size:
                    print(f"Warning: Incomplete image data. Expected {data_size} bytes, got {len(data)}")
                    break
                
                # Create the image based on type
                if depth == 0:      # CV_8U
                    dtype = np.uint8
                elif depth == 1:    # CV_8S
                    dtype = np.int8
                elif depth == 2:    # CV_16U
                    dtype = np.uint16
                elif depth == 3:    # CV_16S
                    dtype = np.int16
                elif depth == 4:    # CV_32S
                    dtype = np.int32
                elif depth == 5:    # CV_32F
                    dtype = np.float32
                elif depth == 6:    # CV_64F
                    dtype = np.float64
                else:
                    # Should not reach here due to earlier check
                    image_index += 1
                    continue
                
                # Create the image - note we need to match OpenCV's memory layout 
                img_data = np.frombuffer(data, dtype=dtype)
                
                # Reshape based on channels
                if channels == 1:
                    image = img_data.reshape((rows, cols))
                else:
                    image = img_data.reshape((rows, cols, channels))
                
                # Apply ROI if valid
                if width > 0 and height > 0:
                    # Make sure ROI is within image bounds
                    if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
                        # Adjust ROI to fit within image bounds
                        valid_x = max(0, x)
                        valid_y = max(0, y)
                        valid_width = min(image.shape[1] - valid_x, width - (valid_x - x))
                        valid_height = min(image.shape[0] - valid_y, height - (valid_y - y))
                        
                        if valid_width <= 0 or valid_height <= 0:
                            print(f"Warning: Invalid ROI dimensions for image {image_index}, using full image")
                            roi_image = image
                        else:
                            roi_image = image[valid_y:valid_y+valid_height, valid_x:valid_x+valid_width]
                    else:
                        roi_image = image[y:y+height, x:x+width]
                else:
                    roi_image = image
                    
                # Apply ROI to background if needed
                if width > 0 and height > 0:
                    # Check if ROI is valid for background
                    if x < 0 or y < 0 or x + width > background.shape[1] or y + height > background.shape[0]:
                        # Adjust ROI to fit within background bounds
                        valid_x = max(0, x)
                        valid_y = max(0, y)
                        valid_width = min(background.shape[1] - valid_x, width - (valid_x - x))
                        valid_height = min(background.shape[0] - valid_y, height - (valid_y - y))
                        
                        if valid_width <= 0 or valid_height <= 0:
                            # Resize entire background to match ROI image size
                            roi_background = cv2.resize(background, (roi_image.shape[1], roi_image.shape[0]))
                        else:
                            roi_background = background[valid_y:valid_y+valid_height, valid_x:valid_x+valid_width]
                            
                            # If sizes still don't match, resize
                            if roi_background.shape != roi_image.shape:
                                roi_background = cv2.resize(roi_background, (roi_image.shape[1], roi_image.shape[0]))
                    else:
                        roi_background = background[y:y+height, x:x+width]
                else:
                    # Make sure background has the same size as the image
                    if background.shape != image.shape:
                        # Resize background to match image size
                        roi_background = cv2.resize(background, (image.shape[1], image.shape[0]))
                    else:
                        roi_background = background
                
                # Process the image using batch-specific configuration
                processed = process_frame(roi_image, roi_background, batch_config)
                
                # Calculate metrics
                deformability, area = filter_processed_image(processed, batch_config)
                
                # Add to results if valid (non-zero)
                if deformability > 0 or area > 0:
                    results.append({
                        'batch': batch_name,
                        'image_index': image_index,
                        'deformability': deformability,
                        'area': area
                    })
                    processed_count += 1
                
                # Print progress every 1000 images
                if (image_index + 1) % 1000 == 0:
                    print(f"Processed {image_index + 1} images, found {processed_count} valid results")
                
                image_index += 1
                
            except Exception as e:
                print(f"Error processing image {image_index}: {str(e)}")
                image_index += 1
                continue
    
    print(f"Batch processing complete. Total images processed: {image_index}, valid results: {len(results)}")
    return results


def find_batch_directories(project_dir: str) -> List[str]:
    """
    Find all batch directories in the project
    
    Args:
        project_dir: Root project directory
    
    Returns:
        List of batch directory paths
    """
    # Look for directories containing images.bin files
    batch_dirs = []
    
    for root, dirs, files in os.walk(project_dir):
        if 'images.bin' in files:
            batch_dirs.append(root)
    
    return batch_dirs


def main(project_dir: str):
    """
    Main function to process all batches in a project
    
    Args:
        project_dir: Root project directory
    """
    # Default configuration based on ProcessingConfig in image_processing.h
    # This will be used only if a batch doesn't have its own config
    default_config = {
        'gaussian_blur_size': 3,
        'bg_subtract_threshold': 8,
        'morph_kernel_size': 3,
        'morph_iterations': 1,
        'area_threshold_min': 250,
        'area_threshold_max': 1200,
        'enable_border_check': True,
        'enable_area_range_check': True,
        'require_single_inner_contour': True,
        'enable_contrast_enhancement': True,
        'contrast_alpha': 1.2,
        'contrast_beta': 10
    }
    
    print(f"Starting batch processing in {project_dir}")
    print(f"Using default configuration for batches without config: {default_config}")
    
    # Find all batch directories
    batch_dirs = find_batch_directories(project_dir)
    
    if not batch_dirs:
        print(f"No batch directories found in {project_dir}")
        return
    
    print(f"Found {len(batch_dirs)} batch directories to process")
    
    # Process each batch
    all_results = []
    for i, batch_dir in enumerate(batch_dirs):
        print(f"\nProcessing batch {i+1}/{len(batch_dirs)}: {batch_dir}")
        batch_results = process_batch(batch_dir, default_config)
        all_results.extend(batch_results)
        print(f"Completed batch {i+1}/{len(batch_dirs)}, total results so far: {len(all_results)}")
    
    # Save results to CSV
    if all_results:
        output_path = os.path.join(project_dir, 'deformability_results.csv')
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(all_results)} results to {output_path}")
    else:
        print("No valid results found")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ms_opencv_process.py <project_dir>")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    main(project_dir)
