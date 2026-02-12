"""Layer extraction module for separating color regions."""

from typing import List, Tuple

import numpy as np

from .types import ImageArray, Color


def extract_color_layers(image: ImageArray, palette: np.ndarray) -> List[Tuple[Color, np.ndarray]]:
    """Extract binary masks for each color in the palette.
    
    Creates binary masks where pixels matching each color are True.
    This separates the image into distinct layers, one per color.
    
    Args:
        image: Quantized image (H, W, C) with values 0-255
        palette: Array of colors (n_colors, C)
        
    Returns:
        List of tuples (color, mask) where:
        - color: The color tuple (R, G, B) or (R, G, B, A)
        - mask: Binary mask (H, W) with True for matching pixels
    """
    layers = []
    
    for color in palette:
        # Create binary mask for this color
        mask = create_color_mask(image, color)
        layers.append((tuple(color), mask))
    
    return layers


def create_color_mask(image: ImageArray, color: Color) -> np.ndarray:
    """Create a binary mask for pixels matching the given color.
    
    Args:
        image: Image array (H, W, C)
        color: Color to match (tuple of values)
        
    Returns:
        Binary mask (H, W) with True for matching pixels
    """
    # Handle both RGB and RGBA
    if len(image.shape) == 3:
        mask = np.all(image == color, axis=2)
    else:
        mask = image == color[0]
    
    return mask


def clean_mask(mask: np.ndarray, min_area: int = 10) -> np.ndarray:
    """Clean mask by removing small noise regions.
    
    Args:
        mask: Binary mask
        min_area: Minimum area (in pixels) to keep
        
    Returns:
        Cleaned mask
    """
    from scipy import ndimage
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Remove small regions
    cleaned_mask = mask.copy()
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        if np.sum(region_mask) < min_area:
            cleaned_mask[region_mask] = False
    
    return cleaned_mask
