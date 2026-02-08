"""Perceptual loss computation for comparing images."""
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

from vectorizer.types import VectorizationError
from vectorizer.compute_backend import rgb_to_lab, delta_e_2000
from vectorizer.raster_ingest import linear_to_srgb


def compute_ssim(image1: np.ndarray, image2: np.ndarray, multichannel: bool = True) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: First image (H, W, C) or (H, W)
        image2: Second image (H, W, C) or (H, W)
        multichannel: Whether images have multiple channels
        
    Returns:
        SSIM score in range [-1, 1] (1 = identical)
    """
    if image1.shape != image2.shape:
        raise VectorizationError(f"Shape mismatch: {image1.shape} vs {image2.shape}")
    
    # Convert to float if needed
    if image1.dtype != np.float64:
        img1 = image1.astype(np.float64)
    else:
        img1 = image1
        
    if image2.dtype != np.float64:
        img2 = image2.astype(np.float64)
    else:
        img2 = image2
    
    # Normalize to [0, 1] if needed
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0
    
    # Compute SSIM
    if multichannel and len(img1.shape) == 3:
        score = ssim(img1, img2, channel_axis=2, data_range=1.0)
    else:
        if len(img1.shape) == 3:
            # Convert to grayscale for SSIM
            img1_gray = np.mean(img1, axis=2)
            img2_gray = np.mean(img2, axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        score = ssim(img1_gray, img2_gray, data_range=1.0)
    
    return float(score)


def compute_delta_e_map(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Delta E 2000 between two images.
    
    Args:
        image1: First image in RGB
        image2: Second image in RGB
        
    Returns:
        Array of Delta E values (H, W)
    """
    if image1.shape != image2.shape:
        raise VectorizationError(f"Shape mismatch: {image1.shape} vs {image2.shape}")
    
    # Convert to LAB
    lab1 = rgb_to_lab(image1)
    lab2 = rgb_to_lab(image2)
    
    # Compute Delta E for each pixel
    height, width = image1.shape[:2]
    delta_e_map = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            delta_e_map[i, j] = delta_e_2000(lab1[i, j], lab2[i, j])
    
    return delta_e_map


def mean_delta_e(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute mean Delta E 2000 between two images.
    
    Args:
        image1: First image in RGB
        image2: Second image in RGB
        
    Returns:
        Mean Delta E value
    """
    delta_e_values = compute_delta_e_map(image1, image2)
    return float(np.mean(delta_e_values))


def perceptual_loss(image1: np.ndarray, image2: np.ndarray, ssim_weight: float = 0.5) -> float:
    """
    Compute combined perceptual loss between two images.
    
    Combines SSIM (structure) and Delta E (color) metrics.
    
    Args:
        image1: First image in RGB [0, 1]
        image2: Second image in RGB [0, 1]
        ssim_weight: Weight for SSIM vs Delta E (0 = Delta E only, 1 = SSIM only)
        
    Returns:
        Perceptual loss score (lower is better)
    """
    # SSIM component (convert to loss, 0 = identical)
    ssim_score = compute_ssim(image1, image2)
    ssim_loss = (1.0 - ssim_score) / 2.0  # Map [-1, 1] to [0, 1]
    
    # Delta E component (normalize, 0 = identical)
    delta_e_score = mean_delta_e(image1, image2)
    delta_e_loss = min(delta_e_score / 100.0, 1.0)  # Normalize to [0, 1]
    
    # Combine
    loss = ssim_weight * ssim_loss + (1.0 - ssim_weight) * delta_e_loss
    
    return float(loss)


def rasterize_svg(svg_string: str, width: int, height: int) -> np.ndarray:
    """
    Rasterize an SVG string to a numpy array.
    
    Args:
        svg_string: SVG XML string
        width: Target width
        height: Target height
        
    Returns:
        RGB array (H, W, 3) in range [0, 1]
    """
    try:
        import cairosvg
        
        # Rasterize SVG
        png_data = cairosvg.svg2png(
            bytestring=svg_string.encode('utf-8'),
            output_width=width,
            output_height=height
        )
        
        # Load PNG data
        from PIL import Image
        import io
        
        image = Image.open(io.BytesIO(png_data))
        image_rgb = image.convert('RGB')
        
        # Convert to numpy array
        array = np.array(image_rgb).astype(np.float32) / 255.0
        
        return array
        
    except ImportError:
        # Fallback: return blank image
        return np.ones((height, width, 3)) * 0.5
    except Exception as e:
        raise VectorizationError(f"Failed to rasterize SVG: {e}")


def compute_loss_from_rasterized(
    original: np.ndarray,
    svg_string: str,
    ssim_weight: float = 0.5
) -> float:
    """
    Compute perceptual loss between original and rasterized SVG.
    
    Args:
        original: Original image (H, W, 3)
        svg_string: SVG to rasterize and compare
        ssim_weight: Weight for SSIM component
        
    Returns:
        Perceptual loss score
    """
    height, width = original.shape[:2]
    
    # Rasterize SVG
    rasterized = rasterize_svg(svg_string, width, height)
    
    # Compute loss
    return perceptual_loss(original, rasterized, ssim_weight)


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to image.
    
    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image
    """
    if image.ndim == 3:
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[..., c] = gaussian_filter(image[..., c], sigma=sigma)
        return blurred
    else:
        return gaussian_filter(image, sigma=sigma)
