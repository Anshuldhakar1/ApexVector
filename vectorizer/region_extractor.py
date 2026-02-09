"""Region extraction via connected components and small region merging."""
from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage
from sklearn.metrics.pairwise import pairwise_distances

from vectorizer.types import Region, ApexConfig, VectorizationError
from vectorizer.color_quantizer import rgb_to_lab


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Calculate Delta E 2000 color difference.
    
    Args:
        lab1: First LAB color
        lab2: Second LAB color
        
    Returns:
        Delta E 2000 value
    """
    # Simplified CIE76 Delta E (sufficient for region merging)
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def extract_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    image: np.ndarray,
    config: ApexConfig
) -> List[Region]:
    """
    Extract regions from quantized label map.
    
    Finds connected components per color label and merges small regions
    based on color similarity.
    
    Args:
        label_map: (H, W) array of color indices
        palette: (n_colors, 3) array of RGB colors
        image: Original image in linear RGB
        config: Configuration parameters
        
    Returns:
        List of extracted regions
        
    Raises:
        VectorizationError: If extraction fails
    """
    try:
        h, w = label_map.shape
        total_area = h * w
        regions = []
        region_id = 0
        
        # Convert palette to LAB for color comparisons
        palette_lab = np.array([rgb_to_lab(rgb.reshape(1, 1, 3)).flatten() 
                                for rgb in palette])
        
        # Process each color label
        for color_idx in range(len(palette)):
            # Create binary mask for this color
            color_mask = (label_map == color_idx)
            
            if not color_mask.any():
                continue
            
            # Find connected components
            labeled_array, num_features = ndimage.label(color_mask)
            
            # Extract each connected component as a region
            for component_id in range(1, num_features + 1):
                component_mask = (labeled_array == component_id)
                area = component_mask.sum()
                
                # Skip very tiny regions (noise)
                if area < 10:
                    continue
                
                # Get bounding box
                coords = np.where(component_mask)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
                
                # Compute mean color from original image
                region_pixels = image[component_mask]
                mean_color = region_pixels.mean(axis=0)
                
                # Create region
                region = Region(
                    mask=component_mask,
                    label=region_id,
                    centroid=None,  # Will be computed in __post_init__
                    mean_color=mean_color,
                    bbox=bbox
                )
                
                # Store color index for later merging
                region._color_idx = color_idx
                
                regions.append(region)
                region_id += 1
        
        # Merge small regions
        regions = _merge_small_regions(
            regions, 
            palette_lab, 
            total_area, 
            config.min_region_area_ratio
        )
        
        # Update neighbor information
        regions = _update_neighbors(regions)
        
        return regions
        
    except Exception as e:
        raise VectorizationError(f"Region extraction failed: {e}")


def _merge_small_regions(
    regions: List[Region],
    palette_lab: np.ndarray,
    total_area: int,
    min_area_ratio: float
) -> List[Region]:
    """
    Merge regions smaller than threshold to nearest color neighbor.
    
    Args:
        regions: List of regions
        palette_lab: LAB palette for color distance
        total_area: Total image area
        min_area_ratio: Minimum area ratio threshold
        
    Returns:
        List of regions after merging
    """
    min_area = total_area * min_area_ratio
    
    # Sort regions by area (smallest first)
    sorted_regions = sorted(enumerate(regions), 
                           key=lambda x: x[1].mask.sum())
    
    merged = set()
    
    for idx, region in sorted_regions:
        if idx in merged:
            continue
            
        area = region.mask.sum()
        if area >= min_area:
            continue
        
        # Find nearest color by Delta E
        color_idx = getattr(region, '_color_idx', 0)
        region_lab = palette_lab[color_idx]
        
        min_distance = float('inf')
        nearest_idx = None
        
        for other_idx, other_region in enumerate(regions):
            if other_idx == idx or other_idx in merged:
                continue
            
            other_color_idx = getattr(other_region, '_color_idx', 0)
            other_lab = palette_lab[other_color_idx]
            
            distance = delta_e_2000(region_lab, other_lab)
            
            if distance < min_distance:
                min_distance = distance
                nearest_idx = other_idx
        
        if nearest_idx is not None:
            # Merge into nearest region
            regions[nearest_idx].mask = regions[nearest_idx].mask | region.mask
            merged.add(idx)
    
    # Filter out merged regions
    return [r for i, r in enumerate(regions) if i not in merged]


def _update_neighbors(regions: List[Region]) -> List[Region]:
    """
    Update neighbor relationships between regions.
    
    Args:
        regions: List of regions
        
    Returns:
        Regions with updated neighbor lists
    """
    h, w = regions[0].mask.shape
    
    for i, region in enumerate(regions):
        neighbors = set()
        
        # Dilate mask to find neighbors
        dilated = ndimage.binary_dilation(region.mask, iterations=1)
        boundary = dilated & ~region.mask
        
        # Check which other regions touch the boundary
        for j, other in enumerate(regions):
            if i == j:
                continue
            if (boundary & other.mask).any():
                neighbors.add(other.label)
        
        region.neighbors = list(neighbors)
    
    return regions
