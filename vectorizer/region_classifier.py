"""Region classification for routing to appropriate vectorization strategy."""
from typing import List
import numpy as np

from vectorizer.types import Region, RegionKind, AdaptiveConfig
from vectorizer.compute_backend import (
    compute_edge_density,
    compute_gradient_direction
)


def classify(regions: List[Region], image: np.ndarray, config: AdaptiveConfig) -> List[Region]:
    """
    Classify regions by type for strategy routing.
    
    Args:
        regions: List of regions to classify
        image: Original image
        config: Adaptive configuration
        
    Returns:
        List of regions with kind attribute set
    """
    classified_regions = []
    
    for region in regions:
        # Compute features
        edge_density = compute_edge_density(region.mask, image)
        gradient_dir, gradient_consistency = compute_gradient_direction(image, region.mask)
        color_variance = _compute_color_variance(region, image)
        complexity = _compute_complexity(region, image)
        
        # Classify based on features
        if edge_density > config.edge_density_threshold:
            # High edge density - use edge strategy
            region.kind = RegionKind.EDGE
        elif gradient_consistency > config.gradient_threshold and color_variance > 0.01:
            # Consistent gradient direction with some color variation
            region.kind = RegionKind.GRADIENT
        elif complexity > config.detail_complexity_threshold:
            # Complex pattern - use detail strategy
            region.kind = RegionKind.DETAIL
        else:
            # Uniform color - use flat strategy
            region.kind = RegionKind.FLAT
        
        classified_regions.append(region)
    
    return classified_regions


def _compute_color_variance(region: Region, image: np.ndarray) -> float:
    """Compute color variance within a region."""
    masked_pixels = image[region.mask]
    
    if len(masked_pixels) == 0:
        return 0.0
    
    # Compute variance for each channel
    variances = np.var(masked_pixels, axis=0)
    
    # Return mean variance across channels
    return float(np.mean(variances))


def _compute_complexity(region: Region, image: np.ndarray) -> float:
    """
    Compute complexity score for a region.
    
    Based on entropy and local variation.
    """
    from scipy.stats import entropy
    
    # Extract region pixels
    masked_pixels = image[region.mask]
    
    if len(masked_pixels) == 0:
        return 0.0
    
    # Quantize to 8 bins per channel
    quantized = (masked_pixels * 7).astype(int)
    
    # Compute histogram
    hist, _ = np.histogramdd(quantized, bins=8, range=[(0, 7)] * 3)
    
    # Compute entropy
    hist_flat = hist.flatten()
    hist_normalized = hist_flat / np.sum(hist_flat) if np.sum(hist_flat) > 0 else hist_flat
    
    # Remove zeros to avoid log issues
    hist_normalized = hist_normalized[hist_normalized > 0]
    
    if len(hist_normalized) == 0:
        return 0.0
    
    entropy_val = entropy(hist_normalized)
    
    # Normalize to [0, 1]
    max_entropy = np.log(8 * 8 * 8)  # Maximum entropy for 3D histogram
    normalized_entropy = entropy_val / max_entropy
    
    return float(normalized_entropy)


def classify_gradient_type(region: Region, image: np.ndarray) -> str:
    """
    Classify gradient type for gradient regions.
    
    Returns 'linear', 'radial', or 'mesh'.
    """
    gradient_dir, gradient_consistency = compute_gradient_direction(image, region.mask)
    
    if gradient_consistency > 0.8:
        # Highly consistent direction - linear gradient
        return 'linear'
    elif gradient_consistency > 0.5:
        # Moderate consistency - could be radial
        # Check if gradient radiates from center
        if _is_radial_gradient(region, image, gradient_dir):
            return 'radial'
        else:
            return 'linear'
    else:
        # Low consistency - mesh gradient
        return 'mesh'


def _is_radial_gradient(region: Region, image: np.ndarray, gradient_dir: np.ndarray) -> bool:
    """Check if gradient appears to be radial (centered)."""
    # Get region center
    center_x, center_y = region.centroid
    
    # Sample points and check if gradient direction points away from center
    coords = np.where(region.mask)
    if len(coords[0]) < 10:
        return False
    
    # Sample every Nth point
    step = max(1, len(coords[0]) // 50)
    sample_indices = range(0, len(coords[0]), step)
    
    consistent_count = 0
    total_count = 0
    
    for i in sample_indices:
        y, x = coords[0][i], coords[1][i]
        
        # Vector from center to point
        to_center = np.array([center_x - x, center_y - y])
        to_center_norm = np.linalg.norm(to_center)
        
        if to_center_norm > 0:
            to_center = to_center / to_center_norm
            
            # Get gradient at this point
            if y > 0 and y < image.shape[0] - 1 and x > 0 and x < image.shape[1] - 1:
                # Compute local gradient
                dx = image[y, x + 1] - image[y, x - 1]
                dy = image[y + 1, x] - image[y - 1, x]
                local_grad = np.array([np.mean(dx), np.mean(dy)])
                
                if np.linalg.norm(local_grad) > 0:
                    local_grad = local_grad / np.linalg.norm(local_grad)
                    
                    # Check if gradient points toward or away from center
                    dot_product = np.dot(local_grad, to_center)
                    if abs(dot_product) > 0.5:
                        consistent_count += 1
                    total_count += 1
    
    if total_count == 0:
        return False
    
    # If > 60% of points have gradient consistent with radial pattern
    return (consistent_count / total_count) > 0.6
