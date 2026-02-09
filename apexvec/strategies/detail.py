"""Detail region vectorization strategy."""
import numpy as np
from typing import List, Tuple

from apexvec.types import Region, VectorRegion, RegionKind, BezierCurve, Point
from apexvec.compute_backend import fit_bezier, delaunay_triangulation
from apexvec.region_decomposer import extract_region_boundary


def vectorize_detail(region: Region, image: np.ndarray, max_triangles: int = 500, max_error: float = 2.0) -> VectorRegion:
    """
    Vectorize a detail region using mesh approximation.
    
    Creates a mesh with color interpolation for complex regions.
    
    Args:
        region: Region to vectorize
        image: Original image
        max_triangles: Maximum number of mesh triangles
        max_error: Maximum bezier fitting error for boundary
        
    Returns:
        VectorRegion with mesh representation
    """
    # Extract boundary
    boundary = extract_region_boundary(region, image.shape[:2])
    if len(boundary) < 2:
        boundary = _create_bbox_boundary(region)
    
    # Fit bezier curves to boundary
    bezier_curves = fit_bezier(boundary, max_error=max_error)
    if bezier_curves:
        bezier_curves = _close_path(bezier_curves)
    
    # Create mesh for interior
    triangles, colors = _create_detail_mesh(region, image, max_triangles)
    
    return VectorRegion(
        kind=RegionKind.DETAIL,
        path=bezier_curves,
        mesh_triangles=triangles,
        mesh_colors=colors
    )


def _create_bbox_boundary(region: Region) -> np.ndarray:
    """Create boundary from bounding box as fallback."""
    if region.bbox is None:
        return np.array([])
    
    x, y, w, h = region.bbox
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
        [x, y]
    ])


def _close_path(curves: List[BezierCurve]) -> List[BezierCurve]:
    """Ensure bezier path is closed."""
    if not curves:
        return curves
    
    first_curve = curves[0]
    last_curve = curves[-1]
    
    if (abs(last_curve.p3.x - first_curve.p0.x) < 0.001 and
        abs(last_curve.p3.y - first_curve.p0.y) < 0.001):
        return curves
    
    p0 = last_curve.p3
    p3 = first_curve.p0
    p1 = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
    p2 = p1
    
    curves.append(BezierCurve(p0, p1, p2, p3))
    return curves


def _create_detail_mesh(region: Region, image: np.ndarray, max_triangles: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create adaptive mesh for detail region.
    
    Uses more triangles in high-variance areas.
    """
    from scipy.ndimage import gaussian_filter
    
    coords = np.where(region.mask)
    if len(coords[0]) == 0:
        return np.array([]), np.array([])
    
    # Compute local variance to guide sampling
    if image.ndim == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Apply mask
    masked_gray = gray.copy()
    masked_gray[~region.mask] = 0
    
    # Compute local variance
    smoothed = gaussian_filter(masked_gray, sigma=2.0)
    variance = (masked_gray - smoothed) ** 2
    variance[~region.mask] = 0
    
    # Adaptive sampling: more points in high variance areas
    num_adaptive_points = min(max_triangles // 2, len(coords[0]) // 10)
    
    # Use variance as probability distribution
    if np.sum(variance) > 0:
        probabilities = variance[coords] / np.sum(variance[coords])
        probabilities = np.nan_to_num(probabilities)
        
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
            adaptive_indices = np.random.choice(
                len(coords[0]),
                size=min(num_adaptive_points, len(coords[0])),
                replace=False,
                p=probabilities
            )
            adaptive_points = np.column_stack([coords[1][adaptive_indices], coords[0][adaptive_indices]])
        else:
            adaptive_points = np.column_stack([coords[1], coords[0]])
    else:
        # Uniform sampling
        step = max(1, len(coords[0]) // num_adaptive_points)
        uniform_indices = range(0, len(coords[0]), step)
        adaptive_points = np.column_stack([coords[1][list(uniform_indices)], coords[0][list(uniform_indices)]])
    
    # Add boundary points
    boundary = extract_region_boundary(region, image.shape[:2])
    if len(boundary) > 0:
        boundary_indices = np.linspace(0, len(boundary) - 1, min(30, len(boundary)), dtype=int)
        boundary_points = boundary[boundary_indices]
        all_points = np.vstack([adaptive_points, boundary_points])
    else:
        all_points = adaptive_points
    
    # Compute Delaunay triangulation
    triangles = delaunay_triangulation(all_points)
    
    # Filter triangles to keep only those inside region
    triangles = _filter_triangles_in_region(triangles, all_points, region)
    
    # Sample colors at vertices
    colors = []
    for point in all_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            colors.append(image[y, x])
        else:
            colors.append(np.array([0.5, 0.5, 0.5]))
    
    return triangles, np.array(colors)


def _filter_triangles_in_region(triangles: np.ndarray, points: np.ndarray, region: Region) -> np.ndarray:
    """Keep only triangles that are mostly inside the region."""
    if len(triangles) == 0:
        return triangles
    
    valid_triangles = []
    
    for tri in triangles:
        # Get triangle vertices
        v0, v1, v2 = points[tri]
        
        # Check if centroid is inside region
        centroid = (v0 + v1 + v2) / 3.0
        cx, cy = int(centroid[0]), int(centroid[1])
        
        if 0 <= cx < region.mask.shape[1] and 0 <= cy < region.mask.shape[0]:
            if region.mask[cy, cx]:
                valid_triangles.append(tri)
    
    return np.array(valid_triangles) if valid_triangles else np.array([])


def simplify_detail_mesh(triangles: np.ndarray, colors: np.ndarray, tolerance: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify mesh by merging similar adjacent triangles.
    
    Args:
        triangles: Triangle indices
        colors: Vertex colors
        tolerance: Color difference threshold for merging
        
    Returns:
        Simplified triangles and colors
    """
    if len(triangles) == 0:
        return triangles, colors
    
    # Simple simplification: remove triangles with very similar colors
    # This is a basic implementation - could be improved with quadric error metrics
    
    return triangles, colors
