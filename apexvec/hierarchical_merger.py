"""Hierarchical region merging from coarse to fine scales."""
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, field

from apexvec.types import Region, VectorizationError


@dataclass
class HierarchicalRegion:
    """Region with hierarchical information."""
    mask: np.ndarray
    label: int
    color: np.ndarray
    scale_idx: int  # Which quantization scale this came from
    parent_idx: Optional[int] = None  # Index of parent region
    area: float = 0.0
    
    def __post_init__(self):
        if self.area == 0.0:
            self.area = np.sum(self.mask) / self.mask.size


def compute_delta_e(color1: np.ndarray, color2: np.ndarray) -> float:
    """
    Compute Delta E color difference in LAB space.
    
    Args:
        color1: First color in RGB [0, 1]
        color2: Second color in RGB [0, 1]
        
    Returns:
        Delta E value
    """
    from skimage.color import rgb2lab
    
    # Convert to LAB
    lab1 = rgb2lab(color1.reshape(1, 1, 3)).flatten()
    lab2 = rgb2lab(color2.reshape(1, 1, 3)).flatten()
    
    # Compute Delta E (CIE76)
    return np.sqrt(np.sum((lab1 - lab2) ** 2))


def contrast_with_parent(
    fine_region: HierarchicalRegion,
    image: np.ndarray,
    parent_region: HierarchicalRegion
) -> float:
    """
    Compute contrast between a fine region and its parent.
    
    Args:
        fine_region: The fine-scale region
        image: Original image
        parent_region: The parent coarse-scale region
        
    Returns:
        Contrast value (Delta E)
    """
    return compute_delta_e(fine_region.color, parent_region.color)


def extract_regions_from_labels(
    label_map: np.ndarray,
    palette: np.ndarray,
    scale_idx: int
) -> List[HierarchicalRegion]:
    """
    Extract regions from a label map.
    
    Args:
        label_map: (H, W) array of color labels
        palette: (n_colors, 3) array of RGB colors
        scale_idx: Index of the quantization scale
        
    Returns:
        List of HierarchicalRegion objects
    """
    regions = []
    n_colors = len(palette)
    
    for label in range(n_colors):
        mask = (label_map == label).astype(np.uint8)
        
        if np.sum(mask) == 0:
            continue
            
        region = HierarchicalRegion(
            mask=mask,
            label=label,
            color=palette[label],
            scale_idx=scale_idx
        )
        regions.append(region)
    
    return regions


def hierarchical_merge(
    label_maps: List[Tuple[np.ndarray, np.ndarray]],
    image: np.ndarray,
    area_threshold: float = 0.02,
    contrast_threshold: float = 10.0,
    max_regions: int = 100
) -> List[HierarchicalRegion]:
    """
    Merge regions from coarse to fine scales intelligently.
    
    Strategy:
    1. Start with coarse regions as base
    2. Add medium regions only where they differ significantly (Delta E > threshold)
    3. Add fine regions only for small high-contrast areas
    4. Prefer larger regions - reject fine segmentation if it doesn't improve perceptual loss
    
    Args:
        label_maps: List of (label_map, palette) tuples from multi-scale quantization
        image: Original image in linear RGB
        area_threshold: Maximum area ratio for fine regions (default: 0.02 = 2%)
        contrast_threshold: Minimum Delta E to accept fine region (default: 10.0)
        max_regions: Maximum number of regions to return
        
    Returns:
        List of merged HierarchicalRegion objects
    """
    if not label_maps:
        raise VectorizationError("No label maps provided for merging")
    
    # Start with coarse regions as base
    coarse_labels, coarse_palette = label_maps[0]
    regions = extract_regions_from_labels(coarse_labels, coarse_palette, scale_idx=0)
    
    print(f"  Coarse scale: {len(regions)} regions")
    
    # Process finer scales
    for scale_idx, (fine_labels, fine_palette) in enumerate(label_maps[1:], start=1):
        fine_regions = extract_regions_from_labels(fine_labels, fine_palette, scale_idx)
        print(f"  Scale {scale_idx}: {len(fine_regions)} candidate regions")
        
        added_count = 0
        for fr in fine_regions:
            # Only consider small regions
            if fr.area >= area_threshold:
                continue
            
            # Find parent region (overlapping coarse region)
            parent_idx = None
            max_overlap = 0
            
            for idx, parent in enumerate(regions):
                if parent.scale_idx >= scale_idx:
                    continue  # Only consider coarser parents
                    
                # Compute overlap
                overlap = np.sum(fr.mask * parent.mask)
                if overlap > max_overlap:
                    max_overlap = overlap
                    parent_idx = idx
            
            if parent_idx is None:
                continue
                
            fr.parent_idx = parent_idx
            parent = regions[parent_idx]
            
            # Check contrast with parent
            contrast = contrast_with_parent(fr, image, parent)
            
            # Only add if high contrast
            if contrast > contrast_threshold:
                regions.append(fr)
                added_count += 1
        
        print(f"    Added {added_count} regions from scale {scale_idx}")
    
    # Sort by scale (coarse first) then by area (larger first)
    regions.sort(key=lambda r: (r.scale_idx, -r.area))
    
    # Limit total regions
    if len(regions) > max_regions:
        print(f"  Limiting to {max_regions} regions (had {len(regions)})")
        regions = regions[:max_regions]
    
    print(f"  Final: {len(regions)} regions")
    return regions


def convert_to_standard_regions(
    hierarchical_regions: List[HierarchicalRegion]
) -> List[Region]:
    """
    Convert hierarchical regions to standard Region objects.
    
    Args:
        hierarchical_regions: List of HierarchicalRegion objects
        
    Returns:
        List of standard Region objects
    """
    regions = []
    
    for hr in hierarchical_regions:
        region = Region(
            mask=hr.mask,
            label=hr.label,
            mean_color=hr.color
        )
        regions.append(region)
    
    return regions
