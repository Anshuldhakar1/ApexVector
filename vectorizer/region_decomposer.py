"""Region decomposition using SLIC segmentation."""
import logging
from typing import List, Tuple
import numpy as np
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours

from vectorizer.types import Region, AdaptiveConfig
from vectorizer.compute_backend import slic_superpixels
from vectorizer.debug_utils import audit_regions_at_extraction

logger = logging.getLogger(__name__)


def decompose(image: np.ndarray, config: AdaptiveConfig) -> List[Region]:
    """
    Decompose image into regions using SLIC segmentation.
    
    Args:
        image: Input image (H, W, 3) in linear RGB
        config: Adaptive configuration
        
    Returns:
        List of Region objects
    """
    # Compute SLIC superpixels
    segments = slic_superpixels(
        image,
        n_segments=config.slic_segments,
        compactness=config.slic_compactness,
        sigma=config.slic_sigma,
        channel_axis=-1
    )
    
    # Extract regions from segments
    regions = _segments_to_regions(segments, image)
    
    # Merge small regions
    regions = _merge_small_regions(regions, config.min_region_size, image)
    
    # Build neighbor relationships
    regions = _compute_neighbors(regions, segments)
    
    # Audit regions after extraction
    audit_regions_at_extraction(regions, phase="after_decomposition")
    
    return regions


def _segments_to_regions(segments: np.ndarray, image: np.ndarray) -> List[Region]:
    """Convert SLIC segments to Region objects."""
    regions = []
    unique_labels = np.unique(segments)
    
    for label in unique_labels:
        # Create binary mask for this segment
        mask = (segments == label)
        
        # Compute mean color
        mean_color = np.mean(image[mask], axis=0)
        
        # Create region
        region = Region(
            mask=mask,
            label=int(label),
            mean_color=mean_color
        )
        
        regions.append(region)
    
    return regions


def _merge_small_regions(
    regions: List[Region],
    min_size: int,
    image: np.ndarray
) -> List[Region]:
    """Merge regions smaller than min_size with their neighbors."""
    merged = True
    
    while merged:
        merged = False
        to_remove = set()
        
        for i, region in enumerate(regions):
            if i in to_remove:
                continue
                
            # Count pixels in region
            pixel_count = np.sum(region.mask)
            
            if pixel_count < min_size:
                # Find best neighbor to merge with
                best_neighbor = _find_best_merge_neighbor(region, regions, to_remove, image)
                
                if best_neighbor is not None:
                    # Merge regions
                    _merge_two_regions(region, best_neighbor, image)
                    to_remove.add(regions.index(best_neighbor))
                    merged = True
        
        # Remove merged regions
        regions = [r for i, r in enumerate(regions) if i not in to_remove]
    
    return regions


def _find_best_merge_neighbor(
    region: Region,
    regions: List[Region],
    to_remove: set,
    image: np.ndarray
) -> Region:
    """Find the best neighbor to merge a small region with."""
    from vectorizer.compute_backend import rgb_to_lab, delta_e_2000
    
    best_neighbor = None
    best_score = float('inf')
    
    region_lab = rgb_to_lab(region.mean_color.reshape(1, 1, 3)).flatten()
    
    for other in regions:
        if other is region or regions.index(other) in to_remove:
            continue
        
        # Check if regions are adjacent (share boundary)
        if not _are_regions_adjacent(region, other):
            continue
        
        # Compute color difference
        other_lab = rgb_to_lab(other.mean_color.reshape(1, 1, 3)).flatten()
        delta_e = delta_e_2000(region_lab, other_lab)
        
        if delta_e < best_score:
            best_score = delta_e
            best_neighbor = other
    
    return best_neighbor


def _are_regions_adjacent(region1: Region, region2: Region) -> bool:
    """Check if two regions share a boundary."""
    # Dilate both masks and check for overlap
    from scipy.ndimage import binary_dilation
    
    mask1_dilated = binary_dilation(region1.mask)
    mask2_dilated = binary_dilation(region2.mask)
    
    # Check if dilated masks overlap
    overlap = mask1_dilated & mask2_dilated
    
    return np.any(overlap)


def _merge_two_regions(region1: Region, region2: Region, image: np.ndarray):
    """Merge region2 into region1."""
    # Combine masks
    region1.mask = region1.mask | region2.mask
    
    # Recalculate centroid
    coords = np.where(region1.mask)
    if len(coords[0]) > 0:
        region1.centroid = (float(np.mean(coords[1])), float(np.mean(coords[0])))
    
    # Recalculate bbox
    if len(coords[0]) > 0:
        region1.bbox = (
            int(np.min(coords[1])),
            int(np.min(coords[0])),
            int(np.max(coords[1]) - np.min(coords[1]) + 1),
            int(np.max(coords[0]) - np.min(coords[0]) + 1)
        )
    
    # Recalculate mean color
    region1.mean_color = np.mean(image[region1.mask], axis=0)
    
    # Update label (use lower label)
    region1.label = min(region1.label, region2.label)


def _compute_neighbors(regions: List[Region], segments: np.ndarray) -> List[Region]:
    """Compute neighbor relationships between regions."""
    from scipy.ndimage import binary_dilation
    
    # Create a label map
    height, width = segments.shape
    label_map = np.zeros((height, width), dtype=int)
    
    for region in regions:
        label_map[region.mask] = region.label
    
    # Find neighbors for each region
    for region in regions:
        # Dilate the mask to find neighbors
        dilated = binary_dilation(region.mask)
        boundary = dilated & ~region.mask
        
        # Get labels of neighboring pixels
        neighbor_labels = set(label_map[boundary])
        neighbor_labels.discard(0)  # Remove background
        neighbor_labels.discard(region.label)  # Remove self
        
        region.neighbors = list(neighbor_labels)
    
    return regions


def extract_region_boundary(
    region: Region,
    image_shape: Tuple[int, int],
    region_idx: int = -1
) -> np.ndarray:
    """
    Extract boundary contour for a region.
    
    Args:
        region: Region object
        image_shape: Shape of original image (H, W)
        region_idx: Index of region for logging (optional)
        
    Returns:
        Array of boundary points (N, 2) in (x, y) format
    """
    # Check for empty mask
    if region.mask is None or not np.any(region.mask):
        logger.warning(
            f"Region {region_idx} (label={region.label}): Cannot extract boundary - empty mask"
        )
        return np.array([])
    
    # Extract boundary using skimage
    boundaries = find_boundaries(region.mask, mode='thick')
    
    # Find contours
    contours = find_contours(boundaries, level=0.5)
    
    if not contours:
        logger.warning(
            f"Region {region_idx} (label={region.label}): No contours found, "
            f"mask_area={np.sum(region.mask)}"
        )
        return np.array([])
    
    # Return the longest contour
    longest_contour = max(contours, key=len)
    
    # Log successful extraction
    logger.debug(
        f"Region {region_idx} (label={region.label}): "
        f"Extracted {len(longest_contour)} boundary points from {len(contours)} contours"
    )
    
    # skimage find_contours returns (row, col) which is (y, x)
    # We need to swap to (x, y) for SVG coordinates
    longest_contour = longest_contour[:, [1, 0]]
    
    return longest_contour
