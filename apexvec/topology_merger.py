"""Topology merging to reduce region count and optimize adjacencies."""
from typing import List, Set, Tuple
import numpy as np

from apexvec.types import VectorRegion, RegionKind
from apexvec.compute_backend import rgb_to_lab, delta_e_2000


def merge_topology(
    regions: List[VectorRegion],
    merge_threshold_delta_e: float = 5.0
) -> List[VectorRegion]:
    """
    Merge adjacent regions with similar colors to reduce complexity.
    
    Args:
        regions: List of vectorized regions
        merge_threshold_delta_e: Delta E threshold for merging
        
    Returns:
        List of merged regions
    """
    if len(regions) <= 1:
        return regions
    
    # Build adjacency graph
    adjacency = _build_adjacency_graph(regions)
    
    # Find merge candidates
    merged = True
    while merged:
        merged = False
        to_remove = set()
        
        for i, region in enumerate(regions):
            if i in to_remove:
                continue
            
            # Check neighbors
            for j in adjacency.get(i, []):
                if j in to_remove or j >= len(regions):
                    continue
                
                neighbor = regions[j]
                
                # Check if both are FLAT regions
                if region.kind != RegionKind.FLAT or neighbor.kind != RegionKind.FLAT:
                    continue
                
                # Check color similarity
                if _should_merge(region, neighbor, merge_threshold_delta_e):
                    # Merge neighbor into region
                    _merge_two_vector_regions(region, neighbor)
                    to_remove.add(j)
                    merged = True
                    break
        
        # Remove merged regions
        regions = [r for i, r in enumerate(regions) if i not in to_remove]
        
        # Rebuild adjacency if we merged
        if merged:
            adjacency = _build_adjacency_graph(regions)
    
    return regions


def _build_adjacency_graph(regions: List[VectorRegion]) -> dict:
    """Build adjacency graph from regions."""
    adjacency = {i: set() for i in range(len(regions))}
    
    # Simple bounding box overlap check
    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i >= j:
                continue
            
            # Check if regions are adjacent (bounding boxes overlap or touch)
            if _regions_adjacent(region1, region2):
                adjacency[i].add(j)
                adjacency[j].add(i)
    
    return adjacency


def _regions_adjacent(region1: VectorRegion, region2: VectorRegion) -> bool:
    """Check if two regions are adjacent."""
    # Use path bounding boxes
    bbox1 = _get_path_bbox(region1.path)
    bbox2 = _get_path_bbox(region2.path)
    
    if bbox1 is None or bbox2 is None:
        return False
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Check if bounding boxes overlap or touch (with small tolerance)
    tolerance = 2.0
    
    overlap_x = not (x1 + w1 < x2 - tolerance or x2 + w2 < x1 - tolerance)
    overlap_y = not (y1 + h1 < y2 - tolerance or y2 + h2 < y1 - tolerance)
    
    return overlap_x and overlap_y


def _get_path_bbox(path) -> Tuple[float, float, float, float]:
    """Get bounding box of a path."""
    if not path:
        return None
    
    all_points = []
    for curve in path:
        all_points.extend([(curve.p0.x, curve.p0.y), (curve.p3.x, curve.p3.y)])
    
    if not all_points:
        return None
    
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _should_merge(region1: VectorRegion, region2: VectorRegion, threshold: float) -> bool:
    """Check if two regions should be merged based on color similarity."""
    if region1.fill_color is None or region2.fill_color is None:
        return False
    
    # Convert to LAB and compute Delta E
    lab1 = rgb_to_lab(region1.fill_color.reshape(1, 1, 3)).flatten()
    lab2 = rgb_to_lab(region2.fill_color.reshape(1, 1, 3)).flatten()
    
    delta_e = delta_e_2000(lab1, lab2)
    
    return delta_e < threshold


def _merge_two_vector_regions(region1: VectorRegion, region2: VectorRegion):
    """Merge region2 into region1."""
    # Average fill colors
    if region1.fill_color is not None and region2.fill_color is not None:
        region1.fill_color = (region1.fill_color + region2.fill_color) / 2.0
    
    # Merge paths (simplified - just keep region1's path)
    # In a full implementation, we'd compute the union of both paths
    # For now, we keep the larger region's path
    bbox1 = _get_path_bbox(region1.path)
    bbox2 = _get_path_bbox(region2.path)
    
    if bbox1 and bbox2:
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        
        if area2 > area1:
            # Use region2's path if it's larger
            region1.path = region2.path


def detect_duplicate_adjacent_colors(regions: List[VectorRegion]) -> List[Tuple[int, int]]:
    """
    Detect adjacent regions with very similar colors.
    
    Returns:
        List of (index1, index2) tuples of duplicate pairs
    """
    duplicates = []
    
    adjacency = _build_adjacency_graph(regions)
    
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if i < j:  # Avoid duplicates
                region1 = regions[i]
                region2 = regions[j]
                
                if region1.kind == RegionKind.FLAT and region2.kind == RegionKind.FLAT:
                    if _should_merge(region1, region2, threshold=3.0):
                        duplicates.append((i, j))
    
    return duplicates
