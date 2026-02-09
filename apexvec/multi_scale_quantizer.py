"""Multi-scale color quantization using K-means clustering."""
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab


def kmeans_quantize(image: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors using K-means clustering in LAB space.
    
    Args:
        image: Input image in linear RGB space (H, W, 3)
        n_colors: Number of colors to quantize to
        
    Returns:
        Tuple of (label_map, palette)
        - label_map: (H, W) array of color labels
        - palette: (n_colors, 3) array of RGB colors in [0, 1]
    """
    # Reshape image to (N_pixels, 3)
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    
    # Convert to LAB for better perceptual clustering
    # First ensure values are in [0, 1]
    pixels_rgb = np.clip(pixels, 0, 1)
    pixels_lab = rgb2lab(pixels_rgb.reshape(h, w, 3)).reshape(-1, 3)
    
    # Run K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_lab)
    
    # Get cluster centers in LAB and convert back to RGB
    centers_lab = kmeans.cluster_centers_
    
    # Convert centers back to RGB
    from skimage.color import lab2rgb
    centers_rgb = lab2rgb(centers_lab.reshape(1, n_colors, 3)).reshape(n_colors, 3)
    
    # Reshape labels back to image dimensions
    label_map = labels.reshape(h, w)
    
    return label_map, np.clip(centers_rgb, 0, 1)


def multi_scale_quantize(
    image: np.ndarray,
    scales: List[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate hierarchical label maps at multiple quantization scales.
    
    Uses multi-scale quantization with coarse, medium, and fine scales
    to capture large regions, mid-size features, and small details.
    
    Args:
        image: Input image in linear RGB space (H, W, 3)
        scales: List of color counts for each scale (default: [6, 10, 18])
            - Coarse: 4-6 colors for large regions (sky, ground)
            - Medium: 8-12 colors for mid-size features (mountains)
            - Fine: 16-24 colors for small details (trees, textures)
            
    Returns:
        List of (label_map, palette) tuples, one per scale
    """
    if scales is None:
        scales = [6, 10, 18]
    
    label_maps = []
    for n_colors in scales:
        labels, palette = kmeans_quantize(image, n_colors)
        label_maps.append((labels, palette))
    
    return label_maps


def get_scale_info(scale_idx: int, n_colors: int) -> str:
    """
    Get descriptive information about a quantization scale.
    
    Args:
        scale_idx: Index of the scale (0=coarse, 1=medium, 2=fine)
        n_colors: Number of colors at this scale
        
    Returns:
        Description string
    """
    descriptions = {
        0: f"Coarse ({n_colors} colors) - Large regions, sky, ground",
        1: f"Medium ({n_colors} colors) - Mid-size features, mountains",
        2: f"Fine ({n_colors} colors) - Small details, trees, textures"
    }
    return descriptions.get(scale_idx, f"Scale {scale_idx} ({n_colors} colors)")
