"""SLIC superpixel quantization for spatially coherent color quantization."""
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from skimage.segmentation import slic
    SKIMAGE_AVAILABLE = True
except ImportError:
    logger.warning("scikit-image not available, SLIC quantization will not work")
    SKIMAGE_AVAILABLE = False


def slic_quantization(
    image: np.ndarray,
    n_segments: int = 100,
    compactness: float = 10.0,
    sigma: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform SLIC superpixel quantization on an image.
    
    SLIC (Simple Linear Iterative Clustering) creates spatially coherent
    superpixels that respect image boundaries, providing better spatial
    coherence than traditional color quantization.
    
    Args:
        image: Input image as HxWx3 uint8 array
        n_segments: Approximate number of superpixels to create
        compactness: Balance between color proximity and spatial proximity.
                    Higher values favor spatial coherence.
        sigma: Standard deviation for Gaussian smoothing before segmentation
        random_state: Random seed for reproducible results
        
    Returns:
        Tuple of (label_map, palette):
        - label_map: HxW array of superpixel labels (int)
        - palette: Nx3 array of dominant colors for each superpixel (uint8)
        
    Raises:
        ImportError: If scikit-image is not available
        ValueError: If image format is invalid
    """
    if not SKIMAGE_AVAILABLE:
        raise ImportError("scikit-image is required for SLIC quantization")
    
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be HxWx3 array")
    
    if image.dtype != np.uint8:
        raise ValueError("Input must be uint8 array")
    
    h, w = image.shape[:2]
    
    # Convert to [0, 1] range for scikit-image
    image_float = image.astype(np.float32) / 255.0
    
    # Apply SLIC superpixel segmentation
    logger.info(f"Running SLIC with {n_segments} segments, compactness={compactness}")
    
    labels = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=-1,  # Use last axis for channels (replaces multichannel=True)
        convert2lab=True,  # Use LAB color space for better color clustering
        random_state=random_state
    )
    
    # Ensure labels are contiguous from 0 to n_segments-1
    unique_labels = np.unique(labels)
    label_map = np.zeros_like(labels, dtype=np.int32)
    
    for new_label, old_label in enumerate(unique_labels):
        label_map[labels == old_label] = new_label
    
    actual_segments = len(unique_labels)
    
    # Compute dominant color for each superpixel
    palette = np.zeros((actual_segments, 3), dtype=np.uint8)
    
    for label_idx in range(actual_segments):
        mask = (label_map == label_idx)
        if np.any(mask):
            # Get mean color of all pixels in this superpixel
            mean_color = np.mean(image[mask], axis=0)
            palette[label_idx] = mean_color.astype(np.uint8)
        else:
            # This shouldn't happen but set a default color
            palette[label_idx] = [128, 128, 128]
    
    logger.info(f"SLIC created {actual_segments} superpixels")
    
    return label_map, palette


def slic_quantize(
    image: np.ndarray,
    n_segments: int = 12,
    compactness: float = 10.0,
    sigma: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for slic_quantization for backward compatibility."""
    return slic_quantization(image, n_segments, compactness, sigma, random_state)


def fallback_quantization(
    image: np.ndarray,
    n_colors: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback quantization using simple color clustering when SLIC is unavailable.
    
    This is a simple fallback that provides basic color quantization
    without spatial coherence. It's better than nothing but doesn't
    provide the benefits of SLIC.
    
    Args:
        image: Input image as HxWx3 uint8 array
        n_colors: Number of colors to quantize to
        
    Returns:
        Tuple of (label_map, palette) same format as slic_quantization
    """
    from sklearn.cluster import MiniBatchKMeans
    
    # Reshape image for clustering
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    
    # Use MiniBatchKMeans for speed
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=3,
        batch_size=10000,
        max_iter=100
    )
    
    logger.info(f"Running fallback K-means with {n_colors} colors")
    kmeans.fit(pixels)
    
    # Get labels and palette
    labels = kmeans.predict(pixels)
    label_map = labels.reshape(h, w)
    
    # Convert cluster centers to uint8 palette
    palette = (np.clip(kmeans.cluster_centers_, 0, 255)).astype(np.uint8)
    
    return label_map, palette