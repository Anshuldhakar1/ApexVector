"""Region merging with same-color constraint."""
from typing import List
import numpy as np
from scipy.ndimage import label, binary_dilation
from apexvec.types import Region


def merge_small_regions_same_color(
    regions: List[Region],
    min_area: int = 100
) -> List[Region]:
    """
    Merge small regions only with same-color neighbors.

    Unlike the original merge_small_regions which could merge across colors,
    this only merges regions that share the same mean_color. This prevents
    color bleeding and maintains palette integrity.

    Args:
        regions: List of regions (must have mean_color and color_label attributes)
        min_area: Minimum area in pixels. Regions smaller than this get merged.

    Returns:
        List of merged regions
    """
    if not regions:
        return regions

    # Group regions by color
    color_groups = {}
    for i, region in enumerate(regions):
        # Use mean_color as key (convert to tuple for hashing)
        color_key = tuple(region.mean_color.astype(int))
        if color_key not in color_groups:
            color_groups[color_key] = []
        color_groups[color_key].append((i, region))

    merged_regions = []

    # Process each color group independently
    for color_key, group_regions in color_groups.items():
        if len(group_regions) == 1:
            # Only one region of this color, keep it
            merged_regions.append(group_regions[0][1])
            continue

        # Separate small and large regions
        small = []
        large = []
        for idx, region in group_regions:
            area = np.sum(region.mask)
            if area < min_area:
                small.append((idx, region, area))
            else:
                large.append((idx, region, area))

        if not small:
            # No small regions to merge
            for idx, region in group_regions:
                merged_regions.append(region)
            continue

        # Build spatial index for finding neighbors
        h, w = regions[0].mask.shape
        region_id_map = np.full((h, w), -1, dtype=np.int32)
        for idx, region, area in large:
            region_id_map[region.mask] = idx

        # Merge each small region into its largest same-color neighbor
        merged_small = set()
        for small_idx, small_region, small_area in small:
            if small_idx in merged_small:
                continue

            # Find neighbors by dilation
            dilated = binary_dilation(small_region.mask, iterations=1)
            neighbor_mask = dilated & ~small_region.mask
            neighbor_ids = np.unique(region_id_map[neighbor_mask])
            neighbor_ids = neighbor_ids[neighbor_ids >= 0]

            if len(neighbor_ids) == 0:
                # Isolated small region, keep it
                merged_regions.append(small_region)
                continue

            # Find largest neighbor
            largest_neighbor_idx = max(
                neighbor_ids,
                key=lambda nid: np.sum(regions[nid].mask)
            )

            # Merge into that neighbor's mask
            regions[largest_neighbor_idx].mask = (
                regions[largest_neighbor_idx].mask | small_region.mask
            )
            merged_small.add(small_idx)

        # Add all large regions (now with merged small ones)
        for idx, region, area in large:
            merged_regions.append(region)

    # Renumber labels
    for i, region in enumerate(merged_regions):
        region.label = i

    return merged_regions
