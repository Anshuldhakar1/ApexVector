"""Curve smoothing module using Bézier and spline fitting."""

from typing import List, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

from .types import Contour


def smooth_contour_bspline(contour: Contour, smoothness: float = 3.0) -> Contour:
    """Smooth contour using B-spline interpolation.
    
    Fits smooth curves through the simplified points using
    B-spline interpolation for smooth polynomial curves.
    
    Args:
        contour: Array of (x, y) points (N, 2)
        smoothness: Smoothing factor, higher = smoother
        
    Returns:
        Smoothed contour points
    """
    if len(contour) < 4:
        return contour
    
    try:
        # Prepare points - need to close the loop
        points = contour.T  # Shape becomes (2, N)
        
        # Create B-spline representation
        # s is smoothing factor (s=0 means interpolation through all points)
        tck, u = splprep(points, u=None, s=smoothness, per=1)
        
        # Evaluate at more points for smooth curve
        u_new = np.linspace(u.min(), u.max(), len(contour) * 2)
        x_new, y_new = splev(u_new, tck, der=0)
        
        return np.column_stack([x_new, y_new])
        
    except Exception:
        # Fall back to original if smoothing fails
        return contour


def fit_bezier_curve(points: np.ndarray, num_control_points: int = 4) -> np.ndarray:
    """Fit a Bézier curve to points.
    
    Args:
        points: Array of (x, y) points
        num_control_points: Number of control points for Bézier
        
    Returns:
        Control points for Bézier curve
    """
    if len(points) < num_control_points:
        return points
    
    # Use least squares to fit Bézier control points
    n = len(points) - 1
    t = np.linspace(0, 1, len(points))
    
    # Bernstein polynomial basis
    def bernstein(i, n, t):
        from scipy.special import comb
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
    
    # Build basis matrix
    A = np.array([bernstein(i, num_control_points - 1, t) 
                  for i in range(num_control_points)]).T
    
    # Solve for control points
    control_points_x = np.linalg.lstsq(A, points[:, 0], rcond=None)[0]
    control_points_y = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
    
    return np.column_stack([control_points_x, control_points_y])


def smooth_contour_catmull_rom(contour: Contour, num_points: int = None) -> Contour:
    """Smooth contour using Catmull-Rom splines.
    
    Catmull-Rom splines pass through all control points.
    
    Args:
        contour: Array of (x, y) points
        num_points: Number of output points (default: len(contour) * 2)
        
    Returns:
        Smoothed contour
    """
    if len(contour) < 4:
        return contour
    
    if num_points is None:
        num_points = len(contour) * 2
    
    def catmull_rom_segment(p0, p1, p2, p3, num_seg_points=10):
        """Generate points on a Catmull-Rom segment."""
        t = np.linspace(0, 1, num_seg_points)
        
        # Catmull-Rom basis matrix
        t2 = t ** 2
        t3 = t ** 3
        
        points = []
        for i in range(len(t)):
            ti = t[i]
            ti2 = t2[i]
            ti3 = t3[i]
            
            point = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * ti +
                (2*p0 - 5*p1 + 4*p2 - p3) * ti2 +
                (-p0 + 3*p1 - 3*p2 + p3) * ti3
            )
            points.append(point)
        
        return np.array(points)
    
    # Close the contour by repeating first few points
    closed = np.vstack([contour[-1], contour, contour[0], contour[1]])
    
    # Generate segments
    all_points = []
    points_per_segment = max(2, num_points // len(contour))
    
    for i in range(1, len(closed) - 2):
        segment = catmull_rom_segment(
            closed[i-1], closed[i], closed[i+1], closed[i+2],
            points_per_segment
        )
        all_points.extend(segment[:-1])  # Skip last point to avoid duplicates
    
    result = np.array(all_points)
    
    # Ensure we have exactly num_points
    if len(result) > num_points:
        indices = np.linspace(0, len(result) - 1, num_points, dtype=int)
        result = result[indices]
    
    return result


def gaussian_smooth_contour(contour: Contour, sigma: float = 1.0) -> Contour:
    """Smooth contour using Gaussian filter on coordinates.
    
    Simple but effective smoothing that doesn't change point count.
    
    Args:
        contour: Array of (x, y) points
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed contour
    """
    from scipy.ndimage import gaussian_filter1d
    
    if len(contour) < 3:
        return contour
    
    # Close the contour for circular smoothing
    extended = np.vstack([contour[-2:], contour, contour[:2]])
    
    # Apply Gaussian filter
    x_smooth = gaussian_filter1d(extended[:, 0], sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(extended[:, 1], sigma=sigma, mode='wrap')
    
    # Return original length (remove padding)
    return np.column_stack([x_smooth[2:-2], y_smooth[2:-2]])
