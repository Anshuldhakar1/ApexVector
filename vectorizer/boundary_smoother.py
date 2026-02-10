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
        p0 = Point(float(x[i]), float(y[i]))
        p3 = Point(float(x[i + 1]), float(y[i + 1]))
        
        # Estimate control points using tangent vectors
        # Evaluate derivatives at endpoints
        dx_vals, dy_vals = splev([u_new[i], u_new[i + 1]], (t, c, k), der=1)
        
        # Ensure we have scalar values
        dx_start = float(dx_vals[0]) if hasattr(dx_vals, '__len__') else float(dx_vals)
        dy_start = float(dy_vals[0]) if hasattr(dy_vals, '__len__') else float(dy_vals)
        dx_end = float(dx_vals[1]) if hasattr(dx_vals, '__len__') else float(dx_vals)
        dy_end = float(dy_vals[1]) if hasattr(dy_vals, '__len__') else float(dy_vals)
        
        # Scale factor for control points (1/3 for cubic bezier)
        dt = u_new[i + 1] - u_new[i]
        scale = dt / 3.0
        
        p1 = Point(float(p0.x + scale * dx_start), float(p0.y + scale * dy_start))
        p2 = Point(float(p3.x - scale * dx_end), float(p3.y - scale * dy_end))
        
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
    p0 = Point(float(min_x), float(min_y))
    p1 = Point(float(max_x), float(min_y))
    p2 = Point(float(max_x), float(max_y))
    p3 = Point(float(min_x), float(max_y))
    
    curves = [
        BezierCurve(p0, Point(float((2*p0.x+p1.x)/3), float((2*p0.y+p1.y)/3)), 
                   Point(float((p0.x+2*p1.x)/3), float((p0.y+2*p1.y)/3)), p1),
        BezierCurve(p1, Point(float((2*p1.x+p2.x)/3), float((2*p1.y+p2.y)/3)),
                   Point(float((p1.x+2*p2.x)/3), float((p1.y+2*p2.y)/3)), p2),
        BezierCurve(p2, Point(float((2*p2.x+p3.x)/3), float((2*p2.y+p3.y)/3)),
                   Point(float((p2.x+2*p3.x)/3), float((p2.y+2*p3.y)/3)), p3),
        BezierCurve(p3, Point(float((2*p3.x+p0.x)/3), float((2*p3.y+p0.y)/3)),
                   Point(float((p3.x+2*p0.x)/3), float((p3.y+2*p0.y)/3)), p0),
    ]
    
    return curves


def extract_contours_subpixel(
    mask: np.ndarray,
    level: float = 0.5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract contours at sub-pixel precision using skimage.
    
    Detects outer boundaries and inner boundaries (holes) using winding order.
    In skimage, contours with positive signed area are typically outer boundaries,
    while those with negative signed area are holes.
    
    Args:
        mask: Binary mask
        level: Contour level (0.5 for boundary between True/False)
        
    Returns:
        Tuple of (outer_contours, hole_contours), each a list of (N, 2) arrays
                  in (col, row) format (x, y coordinates)
    """
    from skimage.measure import find_contours
    
    contours = find_contours(mask, level=level)
    
    if not contours:
        return [], []
    
    outer_contours = []
    hole_contours = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Calculate signed area using shoelace formula
        # Positive area = counter-clockwise = outer boundary
        # Negative area = clockwise = hole
        x = contour[:, 1]  # col = x
        y = contour[:, 0]  # row = y
        
        signed_area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
        
        # Convert to (x, y) format for output
        contour_xy = contour[:, [1, 0]]
        
        if signed_area > 0:
            outer_contours.append(contour_xy)
        else:
            hole_contours.append(contour_xy)
    
    # If no outer contours found but we have contours, assume the longest is outer
    if not outer_contours and contours:
        longest_idx = max(range(len(contours)), key=lambda i: len(contours[i]))
        outer_contours = [contours[longest_idx][:, [1, 0]]]
        hole_contours = [c[:, [1, 0]] for i, c in enumerate(contours) if i != longest_idx]
    
    return outer_contours, hole_contours


def smooth_region_with_holes(
    mask: np.ndarray,
    smoothness_factor: float = 0.5
) -> Tuple[List[BezierCurve], List[List[BezierCurve]]]:
    """
    Smooth a region boundary including holes.
    
    Args:
        mask: Binary mask of region
        smoothness_factor: Controls smoothing amount
        
    Returns:
        Tuple of (outer_curves, list_of_hole_curves)
    """
    outer_contours, hole_contours = extract_contours_subpixel(mask)
    
    # Smooth outer boundary
    outer_curves = []
    if outer_contours:
        # Use the largest outer contour
        longest_outer = max(outer_contours, key=len)
        outer_curves = smooth_boundary_infallible(
            longest_outer,
            smoothness_factor=smoothness_factor
        )
    
    # Smooth holes
    hole_curves_list = []
    for hole_contour in hole_contours:
        hole_curves = smooth_boundary_infallible(
            hole_contour,
            smoothness_factor=smoothness_factor
        )
        if hole_curves:
            hole_curves_list.append(hole_curves)
    
    return outer_curves, hole_curves_list


def simplify_bezier_curves(
    curves: List[BezierCurve],
    tolerance: float = 1.0,
    min_angle_deg: float = 15.0
) -> List[BezierCurve]:
    """
    Simplify a chain of Bezier curves by merging collinear segments.
    
    Args:
        curves: List of BezierCurve objects forming a continuous path
        tolerance: Maximum deviation threshold for simplification
        min_angle_deg: Minimum angle change to keep a segment (in degrees)
        
    Returns:
        Simplified list of BezierCurve objects
    """
    if len(curves) < 3:
        return curves
    
    simplified = []
    
    # Calculate angles at each joint
    angles = []
    for i in range(len(curves) - 1):
        p3_prev = curves[i].p3
        p0_next = curves[i + 1].p0
        
        # Skip if points don't match (discontinuous path)
        if not _points_equal(p3_prev, p0_next, tolerance):
            angles.append(180.0)  # Large angle = don't simplify
            continue
        
        # Calculate angle at joint using tangent vectors
        v1 = (curves[i].p3.x - curves[i].p2.x, curves[i].p3.y - curves[i].p2.y)
        v2 = (curves[i + 1].p1.x - curves[i + 1].p0.x, curves[i + 1].p1.y - curves[i + 1].p0.y)
        
        angle = _angle_between(v1, v2)
        angles.append(angle)
    
    # First pass: remove very small segments (noise)
    filtered_curves = []
    for i, curve in enumerate(curves):
        length = _bezier_length(curve)
        if length < tolerance:
            continue
        filtered_curves.append(curve)
    
    if len(filtered_curves) < 3:
        return filtered_curves
    
    # Second pass: merge collinear segments
    merged = [filtered_curves[0]]
    
    for i in range(1, len(filtered_curves)):
        current = filtered_curves[i]
        previous = merged[-1]
        
        # Check if we can merge with previous
        if _can_merge_curves(previous, current, tolerance, min_angle_deg):
            # Merge by extending previous curve's control points
            merged[-1] = _merge_two_curves(previous, current)
        else:
            merged.append(current)
    
    # Third pass: reduce degree of curves (convert cubic to quadratic if close)
    final = []
    for curve in merged:
        # Check if control points are close to the line between endpoints
        if _is_nearly_linear(curve, tolerance):
            # Convert to quadratic-like representation
            p1_linear = Point(
                (curve.p0.x + 2 * curve.p1.x + curve.p3.x) / 4,
                (curve.p0.y + 2 * curve.p1.y + curve.p3.y) / 4
            )
            p2_linear = Point(
                (curve.p0.x + 2 * curve.p2.x + curve.p3.x) / 4,
                (curve.p0.y + 2 * curve.p2.y + curve.p3.y) / 4
            )
            # Keep as cubic but with collinear control points
            curve = BezierCurve(curve.p0, p1_linear, p2_linear, curve.p3)
        final.append(curve)
    
    return final


def _points_equal(p1: Point, p2: Point, tolerance: float) -> bool:
    """Check if two points are equal within tolerance."""
    return abs(p1.x - p2.x) < tolerance and abs(p1.y - p2.y) < tolerance


def _angle_between(v1: tuple, v2: tuple) -> float:
    """Calculate angle between two vectors in degrees."""
    import math
    
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 < 1e-10 or mag2 < 1e-10:
        return 180.0
    
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.degrees(math.acos(cos_angle))


def _bezier_length(curve: BezierCurve) -> float:
    """Calculate approximate length of a Bezier curve."""
    # Use chord length as approximation
    dx = curve.p3.x - curve.p0.x
    dy = curve.p3.y - curve.p0.y
    return (dx**2 + dy**2) ** 0.5


def _can_merge_curves(
    curve1: BezierCurve,
    curve2: BezierCurve,
    tolerance: float,
    min_angle_deg: float
) -> bool:
    """Check if two curves can be merged."""
    # Check continuity at joint
    if not _points_equal(curve1.p3, curve2.p0, tolerance):
        return False
    
    # Calculate angle at joint
    v1 = (curve1.p3.x - curve1.p2.x, curve1.p3.y - curve1.p2.y)
    v2 = (curve2.p1.x - curve2.p0.x, curve2.p1.y - curve2.p0.y)
    
    angle = _angle_between(v1, v2)
    
    # Merge if angle is close to 180 (collinear)
    return angle > (180 - min_angle_deg)


def _merge_two_curves(curve1: BezierCurve, curve2: BezierCurve) -> BezierCurve:
    """Merge two consecutive Bezier curves into one."""
    # Calculate new control points using subdivision approximation
    # This creates a curve that approximates both original curves
    
    # New p1: weighted average of curve1's p1 and the midpoint
    p1_new = Point(
        (curve1.p1.x + curve1.p3.x) / 2,
        (curve1.p1.y + curve1.p3.y) / 2
    )
    
    # New p2: weighted average of curve2's p2 and the midpoint
    p2_new = Point(
        (curve2.p2.x + curve2.p0.x) / 2,
        (curve2.p2.y + curve2.p0.y) / 2
    )
    
    return BezierCurve(curve1.p0, p1_new, p2_new, curve2.p3)


def _is_nearly_linear(curve: BezierCurve, tolerance: float) -> bool:
    """Check if a Bezier curve is nearly linear."""
    # Check if control points lie near the line between endpoints
    p0, p1, p2, p3 = curve.p0, curve.p1, curve.p2, curve.p3
    
    # Distance from p1 to line p0-p3
    d1 = _point_line_distance(p1, p0, p3)
    
    # Distance from p2 to line p0-p3
    d2 = _point_line_distance(p2, p0, p3)
    
    return d1 < tolerance and d2 < tolerance


def _point_line_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Calculate perpendicular distance from point to line."""
    import math
    
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    
    if abs(dx) < 1e-10 and abs(dy) < 1e-10:
        return math.sqrt((point.x - line_start.x)**2 + (point.y - line_start.y)**2)
    
    # Projection parameter
    t = ((point.x - line_start.x) * dx + (point.y - line_start.y) * dy) / (dx**2 + dy**2)
    t = max(0.0, min(1.0, t))
    
    # Closest point on line
    closest_x = line_start.x + t * dx
    closest_y = line_start.y + t * dy
    
    return math.sqrt((point.x - closest_x)**2 + (point.y - closest_y)**2)
