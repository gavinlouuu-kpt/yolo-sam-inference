import numpy as np
from PIL import Image
import tifffile
from pathlib import Path
from typing import Union, Optional, Tuple
import warnings

def save_optimized_tiff(
    image: Union[np.ndarray, Image.Image],
    output_path: Union[str, Path],
    compression: str = 'zlib',
    compression_level: int = 6,
    tile_size: Tuple[int, int] = (256, 256),
    bigtiff: bool = False,
    metadata: Optional[dict] = None
) -> None:
    """
    Save an image as TIFF with optimized settings for faster writing and smaller file size.
    
    Args:
        image: Input image as numpy array or PIL Image
        output_path: Path to save the TIFF file
        compression: Compression method ('zlib', 'lzw', 'jpeg', 'packbits', or None)
        compression_level: Compression level (0-9) for zlib compression
        tile_size: Size of tiles for tiled TIFF (None for non-tiled)
        bigtiff: Whether to use BigTIFF format for large files
        metadata: Optional metadata dictionary to include in the TIFF file
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            if image.dtype == np.bool_:
                image = image.astype(np.uint8) * 255
            else:
                image = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
        
        # Ensure proper channel order for RGB images
        if len(image.shape) == 3 and image.shape[-1] == 3:
            # Image is already in the correct format (H, W, C)
            pass
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # Convert from (C, H, W) to (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'ImageDescription': 'Created with optimized TIFF saver',
            'Software': 'YoloSAM Pipeline'
        })
        
        # Save the image
        tifffile.imwrite(
            output_path,
            image,
            compression=compression if compression else None,
            compressionargs={'level': compression_level} if compression == 'zlib' else None,
            tile=tile_size,
            bigtiff=bigtiff,
            metadata=metadata,
            photometric='rgb' if len(image.shape) == 3 and image.shape[-1] == 3 else 'minisblack',
            planarconfig='contig'
        )
    except Exception as e:
        raise IOError(f"Failed to save TIFF file: {str(e)}")

def save_mask_as_tiff(
    mask: np.ndarray,
    output_path: Union[str, Path],
    compress: bool = True
) -> None:
    """
    Optimized function for saving binary masks as TIFF files.
    
    Args:
        mask: Binary mask as numpy array
        output_path: Path to save the TIFF file
        compress: Whether to use compression
    """
    try:
        # Ensure mask is binary uint8
        if mask.dtype != np.uint8:
            if mask.dtype == np.bool_:
                mask = mask.astype(np.uint8) * 255
            else:
                mask = (mask > 0).astype(np.uint8) * 255
        
        # Save the mask
        tifffile.imwrite(
            output_path,
            mask,
            compression='zlib' if compress else None,
            compressionargs={'level': 1} if compress else None,
            tile=(512, 512),
            photometric='minisblack',
            planarconfig='contig'
        )
    except Exception as e:
        raise IOError(f"Failed to save mask TIFF file: {str(e)}") 