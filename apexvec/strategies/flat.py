"""Flat color region vectorization strategy."""
import numpy as np
from typing import List

from apexvec.types import Region, VectorRegion, RegionKind, BezierCurve, Point
from apexvec.compute_backend import fit_bezier
from apexvec.region_decomposer import extract_region_boundary


def vectorize_flat(region: Region, image: np.ndarray, max_error: float = 2.0) -> VectorRegion:
    """
    Vectorize a flat (uniform color) region.
    
    Args:
        region: Region to vectorize
        image: Original image
        max_error: Maximum bezier fitting error
        
    Returns:
        VectorRegion with solid fill
    """
    # Compute mean color for fill
    fill_color = np.mean(image[region.mask], axis=0)
    
    # Extract and fit boundary
    boundary = extract_region_boundary(region, image.shape[:2])
    
    if len(boundary) < 2:
        # Fallback: create rectangular boundary from bbox
        boundary = _create_bbox_boundary(region)
    
    # Fit bezier curves to boundary
    bezier_curves = fit_bezier(boundary, max_error=max_error)
    
    # Ensure path is closed
    if bezier_curves and len(bezier_curves) > 0:
        bezier_curves = _close_path(bezier_curves)
    
    return VectorRegion(
        kind=RegionKind.FLAT,
        path=bezier_curves,
        fill_color=fill_color
    )


def _create_bbox_boundary(region: Region) -> np.ndarray:
    """Create boundary from bounding box as fallback."""
    if region.bbox is None:
        return np.array([])
    
    x, y, w, h = region.bbox
    
    # Create rectangular boundary
    boundary = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
        [x, y]  # Close the loop
    ])
    
    return boundary


def _close_path(curves: List[BezierCurve]) -> List[BezierCurve]:
    """Ensure bezier path is closed by connecting end to start."""
    if not curves:
        return curves
    
    first_curve = curves[0]
    last_curve = curves[-1]
    
    # Check if already closed
    if (abs(last_curve.p3.x - first_curve.p0.x) < 0.001 and
        abs(last_curve.p3.y - first_curve.p0.y) < 0.001):
        return curves
    
    # Add closing segment
    p0 = last_curve.p3
    p3 = first_curve.p0
    
    # Create smooth closing curve
    p1 = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
    p2 = p1
    
    closing_curve = BezierCurve(p0, p1, p2, p3)
    curves.append(closing_curve)
    
    return curves
