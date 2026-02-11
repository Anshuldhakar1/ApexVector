"""Raster image ingestion with color space conversion."""
import os
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from PIL import ImageOps

from apexvec.types import IngestResult, VectorizationError


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to linear RGB.
    
    Args:
        srgb: sRGB values in range [0, 1] or [0, 255]
        
    Returns:
        Linear RGB values
    """
    if srgb.max() > 1.0:
        srgb = srgb / 255.0
    
    # Apply sRGB EOTF (Electro-Optical Transfer Function)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )
    
    return linear


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """
    Convert linear RGB to sRGB.
    
    Args:
        linear: Linear RGB values in range [0, 1]
        
    Returns:
        sRGB values in range [0, 1]
    """
    # Apply sRGB OETF (Opto-Electrical Transfer Function)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1.0 / 2.4)) - 0.055
    )
    
    return srgb


def ingest(path: Union[str, Path]) -> IngestResult:
    """
    Ingest a raster image file.
    
    Loads image and converts to both linear RGB (for processing)
    and sRGB (for output comparison).
    
    Args:
        path: Path to image file
        
    Returns:
        IngestResult with both linear and sRGB representations
        
    Raises:
        FileNotFoundError: If file doesn't exist
        VectorizationError: If file cannot be loaded
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not path.is_file():
        raise VectorizationError(f"Path is not a file: {path}")
    
    try:
        # Load image with PIL
        with Image.open(path) as img:
            # Apply EXIF orientation transformation to handle rotation
            img = ImageOps.exif_transpose(img)

            # Convert to RGB if necessary
            if img.mode == 'RGBA':
                has_alpha = True
                # Composite on white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                has_alpha = img.mode in ('RGBA', 'LA', 'P')
                img = img.convert('RGB')
            else:
                has_alpha = False
            
            # Get dimensions
            width, height = img.size
            
            # Convert to numpy array
            image_srgb = np.array(img).astype(np.float32) / 255.0
            
            # Convert to linear RGB for processing
            image_linear = srgb_to_linear(image_srgb)
            
            return IngestResult(
                image_linear=image_linear,
                image_srgb=image_srgb,
                original_path=str(path),
                width=width,
                height=height,
                has_alpha=has_alpha
            )
            
    except (IOError, OSError) as e:
        raise VectorizationError(f"Failed to load image {path}: {e}")
    except Exception as e:
        raise VectorizationError(f"Unexpected error loading image {path}: {e}")


def ingest_from_array(image: np.ndarray, path: str = "") -> IngestResult:
    """
    Create IngestResult from numpy array.
    
    Args:
        image: Image array in sRGB space (H, W, 3) or (H, W, 4)
        path: Optional path for reference
        
    Returns:
        IngestResult
    """
    if image.ndim == 2:
        # Grayscale - convert to RGB
        image = np.stack([image] * 3, axis=-1)
    
    if image.ndim != 3:
        raise VectorizationError(f"Expected 3D array, got {image.ndim}D")
    
    if image.shape[2] == 4:
        # RGBA - composite on white
        has_alpha = True
        alpha = image[..., 3:4]
        rgb = image[..., :3]
        image_srgb = rgb * alpha + (1 - alpha)
    elif image.shape[2] == 3:
        has_alpha = False
        image_srgb = image
    else:
        raise VectorizationError(f"Expected 3 or 4 channels, got {image.shape[2]}")
    
    # Normalize to [0, 1] if needed
    if image_srgb.max() > 1.0:
        image_srgb = image_srgb / 255.0
    
    height, width = image_srgb.shape[:2]
    
    # Convert to linear
    image_linear = srgb_to_linear(image_srgb)
    
    return IngestResult(
        image_linear=image_linear,
        image_srgb=image_srgb,
        original_path=path,
        width=width,
        height=height,
        has_alpha=has_alpha
    )
