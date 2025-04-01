"""
Utilities for encoding binary masks for efficient storage in JSONB format.
"""

import numpy as np
import base64
import zlib
from typing import Dict, Any, Union

def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode a binary mask for efficient storage.
    
    Args:
        mask: Binary mask as numpy array
        
    Returns:
        Dict with encoding type and encoded data
    """
    # Ensure the mask is binary
    binary_mask = mask.astype(bool)
    
    # Compress the mask using zlib
    compressed_bytes = zlib.compress(np.packbits(binary_mask))
    
    # Encode as base64 for JSON compatibility
    encoded_data = base64.b64encode(compressed_bytes).decode('ascii')
    
    # Include the original shape for reconstruction
    return {
        "encoding_type": "compressed_binary",
        "shape": binary_mask.shape,
        "data": encoded_data
    }

def decode_binary_mask(encoded: Dict[str, Any]) -> np.ndarray:
    """
    Decode a binary mask from its encoded form.
    
    Args:
        encoded: Dict with encoding type and encoded data
        
    Returns:
        Binary mask as numpy array
    """
    if encoded.get("encoding_type") != "compressed_binary":
        raise ValueError(f"Unsupported encoding type: {encoded.get('encoding_type')}")
    
    # Get the shape and encoded data
    shape = encoded.get("shape")
    encoded_data = encoded.get("data")
    
    # Decode from base64
    compressed_bytes = base64.b64decode(encoded_data)
    
    # Decompress
    unpacked_bytes = zlib.decompress(compressed_bytes)
    
    # Convert back to boolean array
    bits = np.unpackbits(np.frombuffer(unpacked_bytes, dtype=np.uint8))
    
    # Reshape and truncate to the original shape
    total_bits = shape[0] * shape[1]
    mask = bits[:total_bits].reshape(shape).astype(bool)
    
    return mask 