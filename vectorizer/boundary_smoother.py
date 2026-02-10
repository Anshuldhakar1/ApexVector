"""Infallible boundary smoothing with multi-tier fallback chain."""
import logging
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours

from vectorizer.types import Point, BezierCurve

logger = logging.getLogger(__name__)


def smooth_boundary_infallible(
    points: np.ndarray,
    smoothness_factor: float = 0.5,
    min_points: int = 3
) -> List[BezierCurve]:
    """
    Smooth a boundary with infallible 3-tier fallback chain.
    
    Tier 1: Fit periodic cubic B-spline with Chaikin preprocessing
    Tier 2: Use Chaikin-smoothed polygon as polyline path
    Tier 3: Use raw contours as polygon (guaranteed to work)
    
    Args:
        points: Array of boundary points (N, 2) in (x, y) format
        smoothness_factor: Controls smoothing amount (0.1-2.0)
        min_points: Minimum points required for spline fitting
        
    Returns:
        List of BezierCurve objects (never empty for valid input)
    """
    # Validate input
    if points is None or len(points) < 2:
        logger.warning("smooth_boundary_infallible: Empty or insufficient points")
        return []
    
    if len(points) < min_points:
        logger.debug(f"Too few points ({len(points)}), using direct polyline")
        return _points_to_polyline_curves(points)
    
    # === TIER 1: B-spline with Chaikin preprocessing ===
    try:
        curves = _fit_bspline_with_chaikin(points, smoothness_factor)
        if curves and len(curves) > 0:
            logger.debug(f"Tier 1 succeeded: {len(points)} points -> {len(curves)} bezier curves")
            return curves
    except Exception as e:
        logger.warning(f"Tier 1 (B-spline) failed: {e}")
    
    # === TIER 2: Chaikin-smoothed polygon ===
    try:
        smoothed_points = chaikin_smooth(points, iterations=2)
        curves = _points_to_polyline_curves(smoothed_points)
        if curves and len(curves) > 0:
            logger.debug(f"Tier 2 succeeded: Chaikin smoothing -> {len(curves)} curves")
            return curves
    except Exception as e:
        logger.warning(f"Tier 2 (Chaikin) failed: {e}")
    
    # === TIER 3: Raw polygon (guaranteed) ===
    logger.warning(f"Using Tier 3 fallback: raw polygon for {len(points)} points")
    curves = _points_to_polyline_curves(points)
    
    # Final safety check - should never happen but just in case
    if not curves:
        logger.error("CRITICAL: All fallback tiers failed, creating minimal boundary")
        curves = _create_minimal_boundary(points)
    
    return curves


def _fit_bspline_with_chaikin(
    points: np.ndarray,
    smoothness_factor: float
) -> List[BezierCurve]:
    """
    Fit periodic cubic B-spline with Chaikin preprocessing.
    
    Args:
        points: Boundary points (N, 2)
        smoothness_factor: Smoothing parameter
        
    Returns:
        List of BezierCurve objects
    """
    # Step 1: Resample to uniform arc-length spacing
    uniform_points = resample_uniform_spacing(points, spacing=3.0)
    
    if len(uniform_points) < 4:
        raise ValueError(f"Too few points after resampling: {len(uniform_points)}")
    
    # Step 2: Apply Chaikin corner-cutting (2 iterations)
    chaikin_points = chaikin_smooth(uniform_points, iterations=2)
    
    if len(chaikin_points) < 4:
        raise ValueError(f"Too few points after Chaikin: {len(chaikin_points)}")
    
    # Step 3: Fit periodic cubic B-spline
    # Separate x and y coordinates
    x = chaikin_points[:, 0]
    y = chaikin_points[:, 1]
    
    # Fit spline with periodic boundary conditions
    s = smoothness_factor * len(chaikin_points)
    
    try:
        tck, u = splprep([x, y], s=s, per=1, k=3, quiet=2)
    except Exception as e:
        raise ValueError(f"spline fitting failed: {e}")
    
    # Step 4: Convert spline to cubic Bézier segments
    curves = _spline_to_bezier(tck, num_segments=max(4, len(chaikin_points) // 4))
    
    return curves


def resample_uniform_spacing(
    points: np.ndarray,
    spacing: float = 3.0
) -> np.ndarray:
    """
    Resample points to uniform arc-length spacing.
    
    Args:
        points: Input points (N, 2)
        spacing: Desired spacing between points
        
    Returns:
        Resampled points (M, 2)
    """
    if len(points) < 2:
        return points
    
    # Ensure points form a closed loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Compute cumulative arc length
    diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative_length[-1]
    
    if total_length < spacing:
        # Too short, return original
        return points
    
    # Determine number of samples
    num_samples = max(3, int(total_length / spacing))
    
    # Create evenly spaced parameter values
    new_params = np.linspace(0, total_length, num_samples)
    
    # Interpolate x and y separately
    resampled = np.zeros((num_samples, 2))
    resampled[:, 0] = np.interp(new_params, cumulative_length, points[:, 0])
    resampled[:, 1] = np.interp(new_params, cumulative_length, points[:, 1])
    
    return resampled


def chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Apply Chaikin corner-cutting algorithm.
    
    Each iteration replaces each edge with two new points at 25% and 75%,
    naturally rounding corners without overshooting.
    
    Args:
        points: Input points (N, 2)
        iterations: Number of iterations (2 for sharp, 3 for organic)
        
    Returns:
        Smoothed points
    """
    if len(points) < 3:
        return points
    
    result = points.copy()
    
    for _ in range(iterations):
        if len(result) < 3:
            break
        
        new_points = []
        n = len(result)
        
        for i in range(n):
            # Get current and next point (wrap around for closed loop)
            p0 = result[i]
            p1 = result[(i + 1) % n]
            
            # Create two new points at 25% and 75% along the edge
            q = 0.25 * p0 + 0.75 * p1
            r = 0.75 * p0 + 0.25 * p1
            
            new_points.append(r)
            new_points.append(q)
        
        result = np.array(new_points)
    
    return result


def _spline_to_bezier(tck, num_segments: int = 10) -> List[BezierCurve]:
    """
    Convert B-spline to cubic Bézier curves.
    
    Args:
        tck: Spline tuple (t, c, k) from splprep
        num_segments: Number of Bézier segments to generate
        
    Returns:
        List of BezierCurve objects
    """
    t, c, k = tck
    
    # Generate parameter values
    u_new = np.linspace(0, 1, num_segments + 1)
    
    # Evaluate spline at these points
    x, y = splev(u_new, (t, c, k))
    
    curves = []
    
    # For each segment, approximate as cubic bezier
    for i in range(num_segments):
        p0 = Point(x[i], y[i])
        p3 = Point(x[i + 1], y[i + 1])
        
        # Estimate control points using tangent vectors
        # Evaluate derivatives at endpoints
        dx, dy = splev([u_new[i], u_new[i + 1]], (t, c, k), der=1)
        
        # Scale factor for control points (1/3 for cubic bezier)
        dt = u_new[i + 1] - u_new[i]
        scale = dt / 3.0
        
        p1 = Point(p0.x + scale * dx[0], p0.y + scale * dy[0])
        p2 = Point(p3.x - scale * dx[1], p3.y - scale * dy[1])
        
        curves.append(BezierCurve(p0, p1, p2, p3))
    
    return curves


def _points_to_polyline_curves(points: np.ndarray) -> List[BezierCurve]:
    """
    Convert points to polyline represented as bezier curves.
    
    Creates a series of line segments (degenerate bezier curves).
    
    Args:
        points: Points (N, 2)
        
    Returns:
        List of BezierCurve objects
    """
    if len(points) < 2:
        return []
    
    curves = []
    
    for i in range(len(points) - 1):
        p0 = Point(points[i, 0], points[i, 1])
        p3 = Point(points[i + 1, 0], points[i + 1, 1])
        
        # For a line segment, control points are at 1/3 and 2/3
        p1 = Point((2 * p0.x + p3.x) / 3, (2 * p0.y + p3.y) / 3)
        p2 = Point((p0.x + 2 * p3.x) / 3, (p0.y + 2 * p3.y) / 3)
        
        curves.append(BezierCurve(p0, p1, p2, p3))
    
    return curves


def _create_minimal_boundary(points: np.ndarray) -> List[BezierCurve]:
    """
    Create minimal boundary as absolute fallback.
    
    Creates a simple rectangle or line based on point extents.
    
    Args:
        points: Any points (N, 2)
        
    Returns:
        List with at least one BezierCurve
    """
    if len(points) == 0:
        # Absolute fallback - single point at origin
        p = Point(0, 0)
        return [BezierCurve(p, p, p, p)]
    
    if len(points) == 1:
        p = Point(points[0, 0], points[0, 1])
        return [BezierCurve(p, p, p, p)]
    
    # Create bounding box from points
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    
    # Create rectangular boundary
    p0 = Point(min_x, min_y)
    p1 = Point(max_x, min_y)
    p2 = Point(max_x, max_y)
    p3 = Point(min_x, max_y)
    
    curves = [
        BezierCurve(p0, Point((2*p0.x+p1.x)/3, (2*p0.y+p1.y)/3), 
                   Point((p0.x+2*p1.x)/3, (p0.y+2*p1.y)/3), p1),
        BezierCurve(p1, Point((2*p1.x+p2.x)/3, (2*p1.y+p2.y)/3),
                   Point((p1.x+2*p2.x)/3, (p1.y+2*p2.y)/3), p2),
        BezierCurve(p2, Point((2*p2.x+p3.x)/3, (2*p2.y+p3.y)/3),
                   Point((p2.x+2*p3.x)/3, (p2.y+2*p3.y)/3), p3),
        BezierCurve(p3, Point((2*p3.x+p0.x)/3, (2*p3.y+p0.y)/3),
                   Point((p3.x+2*p0.x)/3, (p3.y+2*p0.y)/3), p0),
    ]
    
    return curves


def extract_contours_subpixel(
    mask: np.ndarray,
    level: float = 0.5
) -> List[np.ndarray]:
    """
    Extract contours at sub-pixel precision using skimage.
    
    Args:
        mask: Binary mask
        level: Contour level (0.5 for boundary between True/False)
        
    Returns:
        List of contour arrays, each (N, 2) in (row, col) format
    """
    contours = find_contours(mask, level=level)
    return contours
