"""Flat color region vectorization strategy."""
import logging
import numpy as np
from typing import List

from apexvec.types import Region, VectorRegion, RegionKind, BezierCurve, Point
from apexvec.region_decomposer import extract_region_boundary
from apexvec.boundary_smoother import smooth_boundary_infallible
from apexvec.debug_utils import log_boundary_extraction

logger = logging.getLogger(__name__)


def vectorize_flat(
    region: Region,
    image: np.ndarray,
    max_error: float = 2.0,
    region_idx: int = -1
) -> VectorRegion:
    """
    Vectorize a flat (uniform color) region.
    
    Uses infallible boundary smoothing with 3-tier fallback to ensure
    every region gets a valid path.
    
    Args:
        region: Region to vectorize
        image: Original image
        max_error: Maximum bezier fitting error (unused, kept for API compatibility)
        region_idx: Index for logging
        
    Returns:
        VectorRegion with solid fill
    """
    # Compute mean color for fill (input is linear RGB, convert to sRGB uint8)
    fill_color_linear = np.mean(image[region.mask], axis=0)
    fill_color = _linear_to_srgb_uint8(fill_color_linear)
    
    # Extract boundary
    boundary = extract_region_boundary(region, image.shape[:2], region_idx=region_idx)
    
    if len(boundary) < 2:
        logger.warning(
            f"Region {region_idx} (label={region.label}): Boundary too short ({len(boundary)} points), "
            f"using bbox fallback"
        )
        # Fallback: create rectangular boundary from bbox
        boundary = _create_bbox_boundary(region)
    
    # Use infallible boundary smoothing (3-tier fallback)
    bezier_curves = smooth_boundary_infallible(
        boundary,
        smoothness_factor=0.5,
        min_points=3
    )
    
    # Log boundary extraction results
    log_boundary_extraction(
        region_idx=region_idx,
        region_label=region.label,
        input_points=len(boundary),
        output_curves=len(bezier_curves),
        success=len(bezier_curves) > 0
    )
    
    # Ensure path is closed
    if bezier_curves and len(bezier_curves) > 0:
        bezier_curves = _close_path(bezier_curves)
    else:
        # This should never happen with infallible smoother, but just in case
        logger.error(f"Region {region_idx}: Infallible smoother returned empty path!")
        bezier_curves = _create_bbox_curves(region)
    
    return VectorRegion(
        kind=RegionKind.FLAT,
        path=bezier_curves,
        fill_color=fill_color
    )


def _linear_to_srgb_uint8(linear_rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0, 1] to sRGB uint8 [0, 255]."""
    # Apply gamma correction
    mask = linear_rgb > 0.0031308
    srgb = np.where(
        mask,
        1.055 * (linear_rgb ** (1.0 / 2.4)) - 0.055,
        12.92 * linear_rgb
    )
    # Convert to uint8 with clipping
    return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def _create_bbox_curves(region: Region) -> List[BezierCurve]:
    """Create bezier curves from region bounding box."""
    if region.bbox is None:
        p = Point(0, 0)
        return [BezierCurve(p, p, p, p)]
    
    x, y, w, h = region.bbox
    
    p0 = Point(x, y)
    p1 = Point(x + w, y)
    p2 = Point(x + w, y + h)
    p3 = Point(x, y + h)
    
    curves = []
    for start, end in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
        c1 = Point((2*start.x + end.x)/3, (2*start.y + end.y)/3)
        c2 = Point((start.x + 2*end.x)/3, (start.y + 2*end.y)/3)
        curves.append(BezierCurve(start, c1, c2, end))
    
    return curves


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
