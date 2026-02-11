"""Boundary smoothing using cubic splines."""
import logging
from typing import List, Tuple, Optional
import numpy as np
from scipy import interpolate
from skimage import measure

from apexvec.types import Region, BezierPath, BezierCurve, Point, VectorizationError, ApexConfig

logger = logging.getLogger(__name__)


def extract_contours(mask: np.ndarray) -> List[np.ndarray]:
    """
    Extract contours from a binary mask.
    
    Args:
        mask: Binary mask array
        
    Returns:
        List of contour arrays, each (N, 2) with (x, y) coordinates
    """
    # Find contours using marching squares
    contours = measure.find_contours(mask, 0.5)
    
    # Convert from (row, col) to (x, y) format
    contours = [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]
    
    return contours


def fit_periodic_spline(
    points: np.ndarray,
    smoothness: float = 0.5,
    n_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a periodic cubic spline to a set of points.
    
    Args:
        points: (N, 2) array of (x, y) coordinates
        smoothness: Smoothing factor (0 = no smoothing, 1 = maximum)
        n_samples: Number of points to sample on spline (default: auto)
        
    Returns:
        Tuple of (t_values, spline_points)
    """
    n_points = len(points)
    
    if n_points < 4:
        # Too few points for cubic spline, return as-is
        return np.linspace(0, 1, n_points), points
    
    # Compute chord-length parameterization
    diffs = np.diff(points, axis=0)
    chord_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    t = np.concatenate([[0], np.cumsum(chord_lengths)])
    t = t / t[-1]  # Normalize to [0, 1]
    
    # Determine number of output samples
    if n_samples is None:
        n_samples = max(4, n_points // 10)
    
    # Fit periodic spline with smoothing
    # For periodic boundary: bc_type='periodic' requires first and last points to match
    if not np.allclose(points[0], points[-1]):
        # Close the loop
        points = np.vstack([points, points[0]])
        t = np.concatenate([t, [1.0]])
    
    # Calculate smoothing factor based on smoothness parameter
    # Higher smoothness = more smoothing
    s = smoothness * n_points * 0.1
    
    try:
        # Fit splines for x and y
        cs_x = interpolate.UnivariateSpline(t, points[:, 0], s=s)
        cs_y = interpolate.UnivariateSpline(t, points[:, 1], s=s)
        
        # Sample spline
        t_fine = np.linspace(0, 1, n_samples)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        spline_points = np.column_stack([x_fine, y_fine])
        
        return t_fine, spline_points
        
    except Exception:
        # Fallback: simple linear interpolation
        t_fine = np.linspace(0, 1, n_samples)
        x_fine = np.interp(t_fine, t, points[:, 0])
        y_fine = np.interp(t_fine, t, points[:, 1])
        return t_fine, np.column_stack([x_fine, y_fine])


def spline_to_bezier(
    t_values: np.ndarray,
    spline_points: np.ndarray
) -> List[BezierCurve]:
    """
    Convert spline points to cubic Bezier curves.
    
    Uses chord-length parameterization to create smooth Bezier segments.
    
    Args:
        t_values: Parameter values along spline
        spline_points: (N, 2) array of spline points
        
    Returns:
        List of BezierCurve segments
    """
    curves = []
    n_points = len(spline_points)
    
    if n_points < 4:
        # Create single curve from points
        p0 = Point(*spline_points[0])
        p3 = Point(*spline_points[-1])
        # Simple linear control points
        p1 = Point(
            p0.x + (p3.x - p0.x) / 3,
            p0.y + (p3.y - p0.y) / 3
        )
        p2 = Point(
            p0.x + 2 * (p3.x - p0.x) / 3,
            p0.y + 2 * (p3.y - p0.y) / 3
        )
        curves.append(BezierCurve(p0, p1, p2, p3))
        return curves
    
    # Create Bezier curves between consecutive points
    for i in range(n_points - 1):
        p0 = Point(*spline_points[i])
        p3 = Point(*spline_points[i + 1])
        
        # Calculate control points based on tangent
        if i == 0:
            # First segment
            prev_point = spline_points[-2] if len(spline_points) > 2 else spline_points[0]
        else:
            prev_point = spline_points[i - 1]
            
        if i == n_points - 2:
            # Last segment
            next_point = spline_points[1] if len(spline_points) > 2 else spline_points[-1]
        else:
            next_point = spline_points[i + 2]
        
        # Calculate tangent at p0 and p3
        tangent0 = spline_points[i + 1] - prev_point
        tangent3 = next_point - spline_points[i]
        
        # Scale tangent for control points (1/3 of segment length)
        seg_length = np.sqrt(np.sum((spline_points[i + 1] - spline_points[i]) ** 2))
        tangent_scale = seg_length / 3
        
        tangent0_norm = np.linalg.norm(tangent0)
        tangent3_norm = np.linalg.norm(tangent3)
        
        if tangent0_norm > 0:
            tangent0 = tangent0 / tangent0_norm * tangent_scale
        if tangent3_norm > 0:
            tangent3 = tangent3 / tangent3_norm * tangent_scale
        
        p1 = Point(p0.x + tangent0[0], p0.y + tangent0[1])
        p2 = Point(p3.x - tangent3[0], p3.y - tangent3[1])
        
        curves.append(BezierCurve(p0, p1, p2, p3))
    
    return curves


def smooth_region_boundaries(
    regions: List[Region],
    config: ApexConfig
) -> List[BezierPath]:
    """
    Smooth boundaries of all regions using cubic splines.
    
    Args:
        regions: List of regions
        config: Configuration parameters
        
    Returns:
        List of BezierPath objects
        
    Raises:
        VectorizationError: If smoothing fails
    """
    try:
        paths = []
        
        for region in regions:
            # Extract contours from mask
            contours = extract_contours(region.mask)
            
            if not contours:
                # Empty region, skip
                paths.append(BezierPath(curves=[], is_closed=True))
                continue
            
            # Use the longest contour (main boundary)
            main_contour = max(contours, key=len)
            
            # Fit periodic spline
            t_values, spline_points = fit_periodic_spline(
                main_contour,
                smoothness=config.spline_smoothness
            )
            
            # Convert to Bezier curves
            curves = spline_to_bezier(t_values, spline_points)
            
            # Create path
            path = BezierPath(curves=curves, is_closed=True)
            paths.append(path)
        
        return paths
        
    except Exception as e:
        raise VectorizationError(f"Boundary smoothing failed: {e}")


def extract_contours_subpixel(mask: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract contours from a binary mask with subpixel precision.
    
    Returns both outer boundary contours and hole contours.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Tuple of (outer_contours, hole_contours)
        Each is a list of (N, 2) arrays with (x, y) coordinates
    """
    from skimage import measure
    
    # Find contours at sub-pixel level
    contours = measure.find_contours(mask, 0.5)
    
    if not contours:
        return [], []
    
    # Convert from (row, col) to (x, y) format and ensure closed loops
    outer_contours = []
    hole_contours = []
    
    for contour in contours:
        # Convert coordinates
        points = np.column_stack([contour[:, 1], contour[:, 0]])
        
        # Ensure closed loop
        if len(points) > 2 and not np.allclose(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        
        # Determine if outer or hole based on area sign
        # Outer contours are counter-clockwise (positive area)
        # Hole contours are clockwise (negative area)
        area = 0
        n = len(points)
        for i in range(n - 1):
            area += (points[i + 1, 0] - points[i, 0]) * (points[i + 1, 1] + points[i, 1])
        
        if area < 0:  # Clockwise = hole
            hole_contours.append(points)
        else:  # Counter-clockwise = outer
            outer_contours.append(points)
    
    return outer_contours, hole_contours


def smooth_boundary_infallible(
    points: np.ndarray,
    smoothness_factor: float = 0.5,
    min_points: int = 3
) -> List[BezierCurve]:
    """
    Smooth a boundary using periodic spline fitting (infallible version).
    
    Args:
        points: Array of boundary points (N, 2) in (x, y) format
        smoothness_factor: Controls smoothing amount (0.1-2.0)
        min_points: Minimum points required
        
    Returns:
        List of BezierCurve objects (never empty for valid input)
    """
    if points is None or len(points) < min_points:
        logger.warning("smooth_boundary_infallible: Insufficient points")
        return []
    
    try:
        # Fit periodic spline
        t_values, spline_points = fit_periodic_spline(
            points,
            smoothness=smoothness_factor
        )
        
        # Convert to Bezier curves
        curves = spline_to_bezier(t_values, spline_points)
        
        return curves
        
    except Exception as e:
        logger.warning(f"Smoothing failed: {e}, returning polyline")
        # Fallback: return as polyline
        return _points_to_polyline(points)


def _points_to_polyline(points: np.ndarray) -> List[BezierCurve]:
    """Convert points to polyline represented as Bezier curves."""
    curves = []
    n = len(points)
    
    for i in range(n - 1):
        p0 = Point(float(points[i, 0]), float(points[i, 1]))
        p3 = Point(float(points[i + 1, 0]), float(points[i + 1, 1]))
        
        # For straight lines, control points are at 1/3 and 2/3
        p1 = Point(
            (2 * p0.x + p3.x) / 3,
            (2 * p0.y + p3.y) / 3
        )
        p2 = Point(
            (p0.x + 2 * p3.x) / 3,
            (p0.y + 2 * p3.y) / 3
        )
        
        curves.append(BezierCurve(p0, p1, p2, p3))
    
    return curves
