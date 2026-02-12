"""Curve simplification module using Douglas-Peucker algorithm."""

from typing import List

import cv2
import numpy as np

from .types import Contour


def simplify_contour(contour: Contour, epsilon_factor: float = 0.01) -> Contour:
    """Simplify contour using Douglas-Peucker algorithm.
    
    Raw contours have too many points (jagged, follows every pixel).
    This applies Ramer-Douglas-Peucker to reduce points while
    maintaining shape, removing redundant points on straight sections.
    
    Args:
        contour: Array of (x, y) points
        epsilon_factor: Approximation accuracy as fraction of perimeter.
                       Smaller = more points, larger = more aggressive simplification.
                       Typical values: 0.001 to 0.05
                       
    Returns:
        Simplified contour with fewer points
    """
    if len(contour) < 3:
        return contour
    
    # Calculate epsilon based on contour perimeter
    perimeter = cv2.arcLength(contour.astype(np.float32), closed=True)
    epsilon = epsilon_factor * perimeter
    
    # Apply Douglas-Peucker
    simplified = cv2.approxPolyDP(
        contour.astype(np.float32),
        epsilon,
        closed=True
    )
    
    return simplified.reshape(-1, 2)


def simplify_contours(contours: List[Contour], epsilon_factor: float = 0.01) -> List[Contour]:
    """Simplify multiple contours.
    
    Args:
        contours: List of contours
        epsilon_factor: Approximation accuracy
        
    Returns:
        List of simplified contours
    """
    return [simplify_contour(c, epsilon_factor) for c in contours if len(c) >= 3]


def adaptive_simplify(contour: Contour, target_points: int = 50) -> Contour:
    """Simplify contour to target number of points.
    
    Uses binary search to find epsilon that produces approximately
    the target number of points.
    
    Args:
        contour: Array of (x, y) points
        target_points: Desired number of output points
        
    Returns:
        Simplified contour
    """
    if len(contour) <= target_points:
        return contour
    
    low, high = 0.0001, 0.5
    best_result = contour
    
    for _ in range(20):  # Max iterations
        mid = (low + high) / 2
        result = simplify_contour(contour, mid)
        
        if len(result) == target_points:
            return result
        elif len(result) < target_points:
            high = mid
        else:
            low = mid
            best_result = result
    
    return best_result
