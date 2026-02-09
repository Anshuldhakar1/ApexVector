"""Color quantization using K-means in LAB color space."""
from typing import Tuple
import numpy as np
from sklearn.cluster import KMeans
from skimage import color

from vectorizer.types import VectorizationError


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to LAB color space.
    
    Args:
        rgb: RGB array with values in [0, 1]
        
    Returns:
        LAB array
    """
    return color.rgb2lab(rgb)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert LAB to RGB color space.
    
    Args:
        lab: LAB array
        
    Returns:
        RGB array with values in [0, 1]
    """
    return color.lab2rgb(lab)


def quantize_colors(
    image: np.ndarray,
    n_colors: int = 12,
    n_init: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors using K-means in LAB space.
    
    Args:
        image: Input image in linear RGB space (H, W, 3)
        n_colors: Number of colors to quantize to
        n_init: Number of K-means++ initializations
        
    Returns:
        Tuple of (label_map, palette)
        - label_map: (H, W) array of color indices
        - palette: (n_colors, 3) array of RGB colors in [0, 1]
        
    Raises:
        VectorizationError: If quantization fails
    """
    try:
        h, w = image.shape[:2]
        
        # Convert to LAB for perceptually uniform clustering
        lab_image = rgb_to_lab(image)
        
        # Reshape to pixels
        pixels_lab = lab_image.reshape(-1, 3)
        
        # K-means++ clustering in LAB space
        kmeans = KMeans(
            n_clusters=n_colors,
            init='k-means++',
            n_init=n_init,
            random_state=42
        )
        labels = kmeans.fit_predict(pixels_lab)
        
        # Reshape labels back to image shape
        label_map = labels.reshape(h, w)
        
        # Convert centroids back to RGB
        palette_lab = kmeans.cluster_centers_
        palette_rgb = lab_to_rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        
        # Clip to valid range
        palette_rgb = np.clip(palette_rgb, 0, 1)
        
        return label_map, palette_rgb
        
    except Exception as e:
        raise VectorizationError(f"Color quantization failed: {e}")
