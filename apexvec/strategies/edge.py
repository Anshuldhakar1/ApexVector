"""Edge region vectorization strategy."""
import numpy as np
from typing import List

from apexvec.types import Region, VectorRegion, RegionKind, BezierCurve, Point
from apexvec.compute_backend import fit_bezier
from apexvec.region_decomposer import extract_region_boundary


def vectorize_edge(region: Region, image: np.ndarray, max_error: float = 1.0) -> VectorRegion:
    """
    Vectorize an edge region with high edge density.
    
    Uses more bezier segments to precisely capture boundaries.
    
    Args:
        region: Region to vectorize
        image: Original image
        max_error: Maximum bezier fitting error (tighter than flat)
        
    Returns:
        VectorRegion with precise edge representation
    """
    # Extract boundary with sub-pixel precision
    boundary = extract_region_boundary(region, image.shape[:2])
    
    if len(boundary) < 2:
        boundary = _create_bbox_boundary(region)
    
    # Fit bezier curves with tighter error tolerance
    # This creates more segments for better edge accuracy
    bezier_curves = fit_bezier(boundary, max_error=max_error, max_iterations=6)
    
    # Ensure path is closed
    if bezier_curves:
        bezier_curves = _close_path(bezier_curves)
    
    # Compute mean color for fill
    fill_color = np.mean(image[region.mask], axis=0)
    
    return VectorRegion(
        kind=RegionKind.EDGE,
        path=bezier_curves,
        fill_color=fill_color
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


def extract_edge_features(region: Region, image: np.ndarray) -> dict:
    """
    Extract edge features for adaptive processing.
    
    Returns dict with edge statistics.
    """
    import cv2
    
    # Extract region from image
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    # Apply mask
    masked_gray = gray.copy()
    masked_gray[~region.mask] = 0
    
    # Detect edges
    edges = cv2.Canny(masked_gray, 50, 150)
    
    # Compute edge statistics
    edge_pixels = np.sum(edges > 0)
    region_pixels = np.sum(region.mask)
    
    if region_pixels > 0:
        edge_density = edge_pixels / region_pixels
    else:
        edge_density = 0.0
    
    # Count edge orientations
    sobelx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(masked_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    orientation_masked = orientation[edges > 0]
    
    if len(orientation_masked) > 0:
        dominant_orientation = np.median(orientation_masked)
        orientation_variance = np.var(orientation_masked)
    else:
        dominant_orientation = 0.0
        orientation_variance = 0.0
    
    return {
        'edge_density': edge_density,
        'edge_count': edge_pixels,
        'dominant_orientation': dominant_orientation,
        'orientation_variance': orientation_variance
    }
