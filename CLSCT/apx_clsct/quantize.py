"""Color quantization module using K-means clustering."""

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans

from .types import ImageArray, QuantizationError


def quantize_colors(image: ImageArray, n_colors: int, random_state: int = 42) -> Tuple[ImageArray, np.ndarray]:
    """Quantize image colors using K-means clustering.
    
    Reduces the image to a limited palette (e.g., 6-12 colors) creating
    distinct color regions similar to a posterized effect.
    
    Args:
        image: Input image as numpy array (H, W, C) with values 0-255
        n_colors: Number of colors to reduce to (must be >= 2)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (quantized_image, color_palette)
        - quantized_image: Image with reduced colors (H, W, C)
        - color_palette: Array of colors (n_colors, C)
        
    Raises:
        QuantizationError: If quantization fails
        ValueError: If n_colors < 2
    """
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")
    
    if image.size == 0:
        raise QuantizationError("Cannot quantize empty image")
    
    try:
        # Store original shape
        h, w = image.shape[:2]
        
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, image.shape[2]) if len(image.shape) == 3 else image.reshape(-1, 1)
        
        # Convert to float for K-means
        pixels = np.float32(pixels)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Get the color palette (cluster centers)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        
        # Reconstruct quantized image
        quantized = palette[labels].reshape(image.shape)
        
        return quantized, palette
        
    except Exception as e:
        raise QuantizationError(f"Color quantization failed: {e}") from e


def get_dominant_colors(image: ImageArray, n_colors: int) -> np.ndarray:
    """Get dominant colors without applying quantization.
    
    Args:
        image: Input image as numpy array
        n_colors: Number of dominant colors to find (must be >= 2)
        
    Returns:
        Array of dominant colors (n_colors, C)
    """
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")
    _, palette = quantize_colors(image, n_colors)
    return palette
