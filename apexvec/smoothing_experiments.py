"""Boundary smoothing experiments with multiple techniques."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev, CubicSpline
from typing import List, Tuple, Optional
from apexvec.types import Point, BezierCurve


def smooth_gaussian(points: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Smooth boundary using Gaussian filter.
    
    Args:
        points: Input points (N, 2)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed points
    """
    if len(points) < 3:
        return points
    
    # Ensure closed loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Apply Gaussian filter to x and y separately
    x_smooth = gaussian_filter1d(points[:, 0], sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(points[:, 1], sigma=sigma, mode='wrap')
    
    smoothed = np.column_stack([x_smooth, y_smooth])
    
    # Ensure still closed
    smoothed[-1] = smoothed[0]
    
    return smoothed


def smooth_moving_average(points: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth boundary using moving average (box filter).
    
    Args:
        points: Input points (N, 2)
        window_size: Size of averaging window
        
    Returns:
        Smoothed points
    """
    if len(points) < window_size:
        return points
    
    # Ensure closed loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    n = len(points)
    smoothed = np.zeros_like(points)
    
    half_window = window_size // 2
    
    for i in range(n):
        # Get window indices (with wraparound)
        indices = [(i + j) % (n - 1) for j in range(-half_window, half_window + 1)]
        window = points[indices]
        smoothed[i] = np.mean(window, axis=0)
    
    # Ensure still closed
    smoothed[-1] = smoothed[0]
    
    return smoothed


def smooth_savgol(points: np.ndarray, window_length: int = 7, polyorder: int = 3) -> np.ndarray:
    """
    Smooth boundary using Savitzky-Golay filter.
    Preserves peak shapes better than moving average.
    
    Args:
        points: Input points (N, 2)
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial to fit
        
    Returns:
        Smoothed points
    """
    if len(points) < window_length:
        return points
    
    # Ensure window_length is odd
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure closed loop
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    
    # Pad the signal to handle circular data
    pad = window_length // 2
    x_padded = np.concatenate([points[-pad-1:-1, 0], points[:, 0], points[1:pad+1, 0]])
    y_padded = np.concatenate([points[-pad-1:-1, 1], points[:, 1], points[1:pad+1, 1]])
    
    # Apply Savitzky-Golay filter
    x_smooth = savgol_filter(x_padded, window_length, polyorder)
    y_smooth = savgol_filter(y_padded, window_length, polyorder)
    
    # Extract center portion
    smoothed = np.column_stack([
        x_smooth[pad:pad+len(points)],
        y_smooth[pad:pad+len(points)]
    ])
    
    # Ensure still closed
    smoothed[-1] = smoothed[0]
    
    return smoothed


def smooth_cubic_spline(points: np.ndarray, num_points: Optional[int] = None) -> np.ndarray:
    """
    Smooth boundary using cubic spline interpolation.
    
    Args:
        points: Input points (N, 2)
        num_points: Number of output points (default: same as input)
        
    Returns:
        Smoothed points
    """
    if len(points) < 4:
        return points
    
    if num_points is None:
        num_points = len(points)
    
    # Remove duplicate consecutive points
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.concatenate([[True], dists > 1e-10])
    points_clean = points[mask]
    
    if len(points_clean) < 4:
        return points
    
    # Parameterize by cumulative arc length
    diffs = np.diff(points_clean, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    t = np.concatenate([[0], np.cumsum(dists)])
    
    if t[-1] == 0:
        return points
    
    t = t / t[-1]  # Normalize to [0, 1]
    
    try:
        # Fit cubic splines
        cs_x = CubicSpline(t, points_clean[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, points_clean[:, 1], bc_type='periodic')
        
        # Evaluate at new points
        t_new = np.linspace(0, 1, num_points)
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)
        
        smoothed = np.column_stack([x_new, y_new])
        
        # Ensure closed
        smoothed[-1] = smoothed[0]
        
        return smoothed
    except Exception:
        return points


def smooth_bspline_scipy(points: np.ndarray, smoothness: float = 1.0) -> np.ndarray:
    """
    Smooth boundary using B-spline with scipy (more robust than current implementation).
    
    Args:
        points: Input points (N, 2)
        smoothness: Smoothing factor (higher = smoother)
        
    Returns:
        Smoothed points
    """
    if len(points) < 4:
        return points
    
    # Clean duplicate points
    diffs = np.diff(points, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    mask = np.concatenate([[True], dists > 1e-10])
    points_clean = points[mask]
    
    if len(points_clean) < 4:
        return points
    
    try:
        # Fit smoothing spline with periodic boundary
        tck, u = splprep(
            [points_clean[:, 0], points_clean[:, 1]],
            s=smoothness * len(points_clean),
            per=1,
            k=3,
            quiet=2
        )
        
        # Evaluate at uniform parameter values
        u_new = np.linspace(0, 1, len(points))
        x_new, y_new = splev(u_new, tck)
        
        smoothed = np.column_stack([x_new, y_new])
        
        # Ensure closed
        smoothed[-1] = smoothed[0]
        
        return smoothed
    except Exception:
        return points


def smooth_douglas_peucker(points: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Simplify boundary using Douglas-Peucker algorithm.
    Reduces points while preserving shape.
    
    Args:
        points: Input points (N, 2)
        epsilon: Distance threshold for simplification
        
    Returns:
        Simplified points
    """
    if len(points) <= 3:
        return points
    
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line."""
        if np.all(line_start == line_end):
            return np.sqrt(np.sum((point - line_start)**2))
        
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len_sq = np.sum(line_vec**2)
        
        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        projection = line_start + t * line_vec
        
        return np.sqrt(np.sum((point - projection)**2))
    
    def douglas_peucker_recursive(pts, start, end, eps, keep):
        """Recursive Douglas-Peucker implementation."""
        if end <= start + 1:
            return
        
        # Find point with maximum distance
        max_dist = 0
        max_idx = start
        
        for i in range(start + 1, end):
            dist = perpendicular_distance(pts[i], pts[start], pts[end])
            if dist > max_dist:
                max_dist = dist
                max_idx = i
        
        # If max distance is greater than epsilon, keep the point
        if max_dist > eps:
            keep[max_idx] = True
            douglas_peucker_recursive(pts, start, max_idx, eps, keep)
            douglas_peucker_recursive(pts, max_idx, end, eps, keep)
    
    # Ensure closed loop
    was_closed = np.allclose(points[0], points[-1])
    if was_closed:
        points_work = points[:-1]  # Remove duplicate last point
    else:
        points_work = points
    
    n = len(points_work)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True
    
    douglas_peucker_recursive(points_work, 0, n - 1, epsilon, keep)
    
    simplified = points_work[keep]
    
    # Close the loop
    if was_closed:
        simplified = np.vstack([simplified, simplified[0]])
    
    return simplified


def smooth_combined(
    points: np.ndarray,
    technique: str = 'gaussian',
    **kwargs
) -> np.ndarray:
    """
    Apply smoothing using specified technique.
    
    Args:
        points: Input points (N, 2)
        technique: Smoothing technique to use
        **kwargs: Technique-specific parameters
        
    Returns:
        Smoothed points
    """
    smoothers = {
        'gaussian': smooth_gaussian,
        'moving_average': smooth_moving_average,
        'savgol': smooth_savgol,
        'cubic_spline': smooth_cubic_spline,
        'bspline': smooth_bspline_scipy,
        'douglas_peucker': smooth_douglas_peucker,
    }
    
    if technique not in smoothers:
        raise ValueError(f"Unknown technique: {technique}. Available: {list(smoothers.keys())}")
    
    return smoothers[technique](points, **kwargs)


def points_to_bezier_curves(points: np.ndarray) -> List[BezierCurve]:
    """
    Convert smoothed points to Bezier curves.
    
    For smooth curves, we create a series of cubic Bezier segments
    that approximate the smoothed points.
    
    Args:
        points: Smoothed boundary points (N, 2)
        
    Returns:
        List of BezierCurve objects
    """
    if len(points) < 2:
        return []
    
    curves = []
    n = len(points)
    
    for i in range(n - 1):
        p0 = Point(float(points[i, 0]), float(points[i, 1]))
        p3 = Point(float(points[i + 1, 0]), float(points[i + 1, 1]))
        
        # For smooth curves, estimate control points from neighboring points
        if i == 0:
            # First segment: use forward difference
            if n > 2:
                dx = points[i + 2, 0] - points[i, 0]
                dy = points[i + 2, 1] - points[i, 1]
            else:
                dx = points[i + 1, 0] - points[i, 0]
                dy = points[i + 1, 1] - points[i, 1]
        elif i == n - 2:
            # Last segment: use backward difference
            dx = points[i + 1, 0] - points[i - 1, 0]
            dy = points[i + 1, 1] - points[i - 1, 1]
        else:
            # Middle segments: use central difference
            dx = points[i + 2, 0] - points[i, 0]
            dy = points[i + 2, 1] - points[i, 1]
        
        # Scale control points for smoothness
        scale = 0.3  # Adjust for more/less tension
        
        p1 = Point(
            float(p0.x + scale * dx / 2),
            float(p0.y + scale * dy / 2)
        )
        p2 = Point(
            float(p3.x - scale * dx / 2),
            float(p3.y - scale * dy / 2)
        )
        
        curves.append(BezierCurve(p0, p1, p2, p3))
    
    return curves
