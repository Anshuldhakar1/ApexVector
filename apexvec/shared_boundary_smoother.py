"""Gaussian smoothing of shared boundaries between regions."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from apexvec.types import VectorRegion, RegionKind, BezierCurve, Point


def smooth_shared_boundary(
    boundary: List[Tuple[float, float]],
    sigma: float = 1.0,
    mode: str = 'wrap'
) -> List[Tuple[float, float]]:
    """
    Apply Gaussian smoothing to a shared boundary.

    Args:
        boundary: List of (x, y) points defining the boundary
        sigma: Gaussian smoothing parameter (pixels)
        mode: Edge handling mode ('wrap', 'nearest', 'reflect')

    Returns:
        Smoothed boundary points
    """
    if len(boundary) < 3:
        return boundary

    points = np.array(boundary)

    # Sort points to form continuous path
    # Use angle around centroid for closed curves
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]

    # Apply Gaussian smoothing to x and y separately
    smoothed_x = gaussian_filter1d(sorted_points[:, 0], sigma=sigma, mode=mode)
    smoothed_y = gaussian_filter1d(sorted_points[:, 1], sigma=sigma, mode=mode)

    smoothed = list(zip(smoothed_x, smoothed_y))

    return smoothed


def reconstruct_regions_from_boundaries(
    boundaries: Dict[Tuple[int, int], List[Tuple[float, float]]],
    palette: np.ndarray,
    image_shape: Tuple[int, int],
    sigma: float = 1.0
) -> List[VectorRegion]:
    """
    Reconstruct all regions from smoothed shared boundaries.

    For each region, collect all its incident smoothed boundaries and
    assemble into closed paths.

    Args:
        boundaries: Dict mapping (label_a, label_b) to boundary points
        palette: (N, 3) color palette
        image_shape: (H, W) image dimensions
        sigma: Smoothing parameter

    Returns:
        List of VectorRegion objects
    """
    # Find all unique labels
    all_labels = set()
    for label_a, label_b in boundaries.keys():
        all_labels.add(label_a)
        all_labels.add(label_b)

    vector_regions = []

    for label_id in sorted(all_labels):
        # Find all boundaries involving this label
        region_boundaries = []
        for (la, lb), boundary_points in boundaries.items():
            if la == label_id:
                # Use boundary as-is
                smoothed = smooth_shared_boundary(boundary_points, sigma=sigma)
                region_boundaries.append(smoothed)
            elif lb == label_id:
                # Use boundary in reverse direction
                smoothed = smooth_shared_boundary(boundary_points, sigma=sigma)
                region_boundaries.append(smoothed[::-1])  # Reverse

        if not region_boundaries:
            continue

        # Combine all boundaries into one or more closed paths
        # For now, concatenate them (simplification)
        all_points = []
        for boundary in region_boundaries:
            all_points.extend(boundary)

        # Close the path
        if all_points and all_points[0] != all_points[-1]:
            all_points.append(all_points[0])

        # Convert to Bezier curves
        bezier_path = points_to_bezier(all_points)

        # Create vector region
        vector_region = VectorRegion(
            kind=RegionKind.FLAT,
            path=bezier_path,
            hole_paths=[],  # TODO: Handle holes
            fill_color=palette[label_id]
        )

        vector_regions.append(vector_region)

    return vector_regions


def points_to_bezier(
    points: List[Tuple[float, float]],
    tolerance: float = 1.0
) -> List[BezierCurve]:
    """
    Convert polyline points to Bezier curves.

    Simple implementation: create line segments.
    TODO: Implement proper curve fitting for smoother results.
    """
    if len(points) < 2:
        return []

    curves = []

    for i in range(len(points) - 1):
        p0 = Point(x=points[i][0], y=points[i][1])
        p3 = Point(x=points[i+1][0], y=points[i+1][1])

        # For straight lines, control points are midpoints
        p1 = Point(
            x=p0.x + (p3.x - p0.x) * 0.33,
            y=p0.y + (p3.y - p0.y) * 0.33
        )
        p2 = Point(
            x=p0.x + (p3.x - p0.x) * 0.67,
            y=p0.y + (p3.y - p0.y) * 0.67
        )

        curves.append(BezierCurve(p0=p0, p1=p1, p2=p2, p3=p3))

    return curves
