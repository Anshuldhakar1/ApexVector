"""Computational backend with SLIC, color metrics, bezier fitting, and triangulation."""
import numpy as np
from typing import List, Tuple, Optional
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy.spatial import Delaunay
import cv2

from apexvec.types import Point, BezierCurve


def slic_superpixels(
    image: np.ndarray,
    n_segments: int = 400,
    compactness: float = 10.0,
    sigma: float = 1.0,
    channel_axis: int = -1
) -> np.ndarray:
    """
    Compute SLIC superpixels for an image.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        n_segments: Approximate number of segments
        compactness: Balance between color and spatial proximity
        sigma: Gaussian smoothing sigma
        channel_axis: Axis of the channel dimension (-1 for last axis)
        
    Returns:
        Array of segment labels (H, W)
    """
    image_float = img_as_float(image)
    
    segments = slic(
        image_float,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        channel_axis=channel_axis,
        start_label=0
    )
    
    return segments


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIELAB color space.
    
    Args:
        rgb: RGB values in range [0, 1] or [0, 255]
        
    Returns:
        LAB values
    """
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    
    # RGB to XYZ
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    
    # D65 illuminant
    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    xyz = np.dot(rgb_linear, xyz_matrix.T)
    
    # XYZ to Lab
    xyz_ref = np.array([0.95047, 1.0, 1.08883])
    xyz_normalized = xyz / xyz_ref
    
    mask = xyz_normalized > 0.008856
    f_xyz = np.where(mask, xyz_normalized ** (1/3), 7.787 * xyz_normalized + 16/116)
    
    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    
    return np.stack([L, a, b], axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Calculate CIEDE2000 color difference between two LAB colors.
    
    Args:
        lab1: First LAB color [L, a, b]
        lab2: Second LAB color [L, a, b]
        
    Returns:
        Delta E 2000 value
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    # CIEDE2000 implementation (simplified for speed)
    dL = L2 - L1
    Lbar = (L1 + L2) / 2
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    Cbar = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    dC = C2_prime - C1_prime
    
    if C1_prime * C2_prime == 0:
        dh = 0
    else:
        if abs(h2_prime - h1_prime) <= 180:
            dh = h2_prime - h1_prime
        elif h2_prime - h1_prime > 180:
            dh = h2_prime - h1_prime - 360
        else:
            dh = h2_prime - h1_prime + 360
    
    dH = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh / 2))
    
    Lbar_prime = (L1 + L2) / 2
    Cbar_prime = (C1_prime + C2_prime) / 2
    
    if C1_prime * C2_prime == 0:
        hbar_prime = h1_prime + h2_prime
    else:
        if abs(h1_prime - h2_prime) <= 180:
            hbar_prime = (h1_prime + h2_prime) / 2
        elif h1_prime + h2_prime < 360:
            hbar_prime = (h1_prime + h2_prime + 360) / 2
        else:
            hbar_prime = (h1_prime + h2_prime - 360) / 2
    
    T = (1 - 0.17 * np.cos(np.radians(hbar_prime - 30)) +
         0.24 * np.cos(np.radians(2 * hbar_prime)) +
         0.32 * np.cos(np.radians(3 * hbar_prime + 6)) -
         0.20 * np.cos(np.radians(4 * hbar_prime - 63)))
    
    SL = 1 + 0.015 * (Lbar_prime - 50)**2 / np.sqrt(20 + (Lbar_prime - 50)**2)
    SC = 1 + 0.045 * Cbar_prime
    SH = 1 + 0.015 * Cbar_prime * T
    
    RT = -2 * np.sqrt(Cbar_prime**7 / (Cbar_prime**7 + 25**7)) * np.sin(np.radians(60 * np.exp(-((hbar_prime - 275) / 25)**2)))
    
    dE = np.sqrt((dL / SL)**2 + (dC / SC)**2 + (dH / SH)**2 + RT * (dC / SC) * (dH / SH))
    
    return float(dE)


def fit_bezier(
    points: np.ndarray,
    max_error: float = 2.0,
    max_iterations: int = 4
) -> List[BezierCurve]:
    """
    Fit cubic bezier curves to a set of points using Schneider's algorithm.
    
    Args:
        points: Array of points (N, 2)
        max_error: Maximum fitting error
        max_iterations: Maximum refinement iterations
        
    Returns:
        List of BezierCurve segments
    """
    if len(points) < 2:
        return []
    
    if len(points) == 2:
        # Line segment
        p0 = Point(points[0, 0], points[0, 1])
        p3 = Point(points[1, 0], points[1, 1])
        p1 = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
        p2 = p1
        return [BezierCurve(p0, p1, p2, p3)]
    
    curves = []
    
    # Split at corners (points with high curvature)
    corners = _find_corners(points)
    
    start_idx = 0
    for end_idx in corners + [len(points)]:
        segment = points[start_idx:end_idx]
        if len(segment) >= 2:
            segment_curves = _fit_bezier_segment_recursive(segment, max_error, max_iterations)
            curves.extend(segment_curves)
        start_idx = end_idx
    
    # Enforce minimum 3 segments per contour to prevent jagged output
    min_segments = 3
    if len(curves) < min_segments and len(points) >= 4:
        # Redistribute points and force split into minimum segments
        curves = _force_minimum_segments(points, max_error, max_iterations, min_segments)
    
    return curves


def _force_minimum_segments(
    points: np.ndarray,
    max_error: float,
    max_iterations: int,
    min_segments: int
) -> List[BezierCurve]:
    """Force split points into at least min_segments bezier curves."""
    curves = []
    
    # Split points into min_segments roughly equal parts
    segment_size = len(points) // min_segments
    
    for i in range(min_segments):
        start_idx = i * segment_size
        if i == min_segments - 1:
            # Last segment gets remaining points
            end_idx = len(points)
        else:
            end_idx = (i + 1) * segment_size + 1  # +1 for overlap
        
        segment = points[start_idx:end_idx]
        if len(segment) >= 2:
            segment_curves = _fit_bezier_segment_recursive(segment, max_error, max_iterations)
            curves.extend(segment_curves)
    
    return curves


def _find_corners(points: np.ndarray, angle_threshold: float = 60.0) -> List[int]:
    """Find corner points based on angle change."""
    if len(points) < 3:
        return []
    
    corners = []
    
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            v1_norm = v1 / np.linalg.norm(v1)
            v2_norm = v2 / np.linalg.norm(v2)
            
            angle = np.degrees(np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1, 1)))
            
            if angle > angle_threshold:
                corners.append(i)
    
    return corners


def _fit_bezier_segment_recursive(
    points: np.ndarray,
    max_error: float,
    max_iterations: int
) -> List[BezierCurve]:
    """Recursively fit cubic bezier curves to a segment of points."""
    if len(points) < 2:
        return []
    
    if len(points) == 2:
        p0 = Point(points[0, 0], points[0, 1])
        p3 = Point(points[1, 0], points[1, 1])
        p1 = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
        p2 = p1
        return [BezierCurve(p0, p1, p2, p3)]
    
    # Fit using least squares approximation
    p0 = Point(points[0, 0], points[0, 1])
    p3 = Point(points[-1, 0], points[-1, 1])
    
    # Estimate control points
    chord = np.array([p3.x - p0.x, p3.y - p0.y])
    chord_length = np.linalg.norm(chord)
    
    if chord_length > 0:
        alpha = chord_length * 0.3
        p1 = Point(p0.x + alpha * chord[0] / chord_length, p0.y + alpha * chord[1] / chord_length)
        p2 = Point(p3.x - alpha * chord[0] / chord_length, p3.y - alpha * chord[1] / chord_length)
    else:
        mid = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
        p1 = p2 = mid
    
    curve = BezierCurve(p0, p1, p2, p3)
    
    # Check error and split if needed
    error = _bezier_error(curve, points)
    
    if error > max_error and max_iterations > 0 and len(points) > 3:
        # Split at midpoint and fit recursively
        mid_idx = len(points) // 2
        curves1 = _fit_bezier_segment_recursive(points[:mid_idx + 1], max_error, max_iterations - 1)
        curves2 = _fit_bezier_segment_recursive(points[mid_idx:], max_error, max_iterations - 1)
        
        return curves1 + curves2
    
    return [curve]


def _bezier_error(curve: BezierCurve, points: np.ndarray) -> float:
    """Calculate maximum distance from points to bezier curve."""
    max_dist = 0.0
    
    for point in points:
        dist = _point_to_bezier_distance(Point(point[0], point[1]), curve)
        max_dist = max(max_dist, dist)
    
    return max_dist


def _point_to_bezier_distance(point: Point, curve: BezierCurve) -> float:
    """Approximate distance from point to bezier curve."""
    # Sample curve and find minimum distance
    min_dist = float('inf')
    
    for t in np.linspace(0, 1, 20):
        curve_point = _eval_bezier(curve, t)
        dist = np.sqrt((point.x - curve_point.x)**2 + (point.y - curve_point.y)**2)
        min_dist = min(min_dist, dist)
    
    return min_dist


def _eval_bezier(curve: BezierCurve, t: float) -> Point:
    """Evaluate bezier curve at parameter t."""
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    
    x = mt3 * curve.p0.x + 3 * mt2 * t * curve.p1.x + 3 * mt * t2 * curve.p2.x + t3 * curve.p3.x
    y = mt3 * curve.p0.y + 3 * mt2 * t * curve.p1.y + 3 * mt * t2 * curve.p2.y + t3 * curve.p3.y
    
    return Point(x, y)


def delaunay_triangulation(points: np.ndarray) -> np.ndarray:
    """
    Compute Delaunay triangulation of points.
    
    Args:
        points: Array of points (N, 2)
        
    Returns:
        Array of triangle indices (M, 3)
    """
    if len(points) < 3:
        return np.array([])
    
    tri = Delaunay(points)
    return tri.simplices


def compute_edge_density(mask: np.ndarray, image: np.ndarray) -> float:
    """
    Compute edge density within a masked region.
    
    Args:
        mask: Binary mask of region
        image: Input image (H, W, C)
        
    Returns:
        Edge density ratio (0.0 to 1.0)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    edges = cv2.Canny(gray, 50, 150)
    
    masked_edges = edges * mask.astype(np.uint8)
    
    region_pixels = np.sum(mask)
    if region_pixels == 0:
        return 0.0
    
    edge_pixels = np.sum(masked_edges > 0)
    
    return float(edge_pixels) / float(region_pixels)


def compute_gradient_direction(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute gradient direction consistency within a region.
    
    Args:
        image: Input image (H, W, C)
        mask: Binary mask
        
    Returns:
        Tuple of (mean gradient direction, consistency score)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (image * 255).astype(np.uint8)
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # Apply mask
    masked_mag = magnitude * mask.astype(np.float64)
    masked_dir = direction * mask.astype(np.float64)
    
    # Compute consistency
    region_pixels = np.sum(mask)
    if region_pixels == 0:
        return np.array([0.0, 0.0]), 0.0
    
    # Mean direction (using circular statistics)
    sin_sum = np.sum(np.sin(masked_dir))
    cos_sum = np.sum(np.cos(masked_dir))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    
    # Consistency (1 = perfectly consistent, 0 = random)
    r = np.sqrt(sin_sum**2 + cos_sum**2) / region_pixels
    
    return np.array([np.cos(mean_angle), np.sin(mean_angle)]), float(r)
