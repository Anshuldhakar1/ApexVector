"""Contour detection module for boundary tracing."""

from typing import List, Tuple

import cv2
import numpy as np

from .types import ImageArray, Contour, ContourError


def find_contours(mask: np.ndarray, method: str = "simple") -> List[Contour]:
    """Find contours in a binary mask.
    
    Uses edge detection algorithms or boundary tracing on the mask
    to find boundary points around connected regions.
    
    Args:
        mask: Binary mask (H, W) with True/255 for object pixels
        method: Contour method - "simple" or "tc89_l1" or "tc89_kcos"
        
    Returns:
        List of contours, each contour is array of (x, y) points
        
    Raises:
        ContourError: If contour detection fails
    """
    try:
        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8)) * 255
        
        # Map method string to OpenCV constant
        method_map = {
            "simple": cv2.CHAIN_APPROX_SIMPLE,
            "tc89_l1": cv2.CHAIN_APPROX_TC89_L1,
            "tc89_kcos": cv2.CHAIN_APPROX_TC89_KCOS,
            "none": cv2.CHAIN_APPROX_NONE
        }
        cv_method = method_map.get(method, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL,  # External contours only
            cv_method
        )
        
        # Convert to numpy arrays of shape (N, 2)
        result = []
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points
                contour = contour.reshape(-1, 2)
                result.append(contour)
        
        return result
        
    except Exception as e:
        raise ContourError(f"Contour detection failed: {e}") from e


def find_contours_hierarchical(mask: np.ndarray) -> Tuple[List[Contour], List[int]]:
    """Find contours with hierarchy information (for holes).
    
    Args:
        mask: Binary mask
        
    Returns:
        Tuple of (contours, hierarchy) where hierarchy indicates
        parent-child relationships for holes
    """
    if mask.dtype != np.uint8:
        mask = (mask.astype(np.uint8)) * 255
    
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_TREE,  # Retrieve all contours with hierarchy
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Reshape contours
    result = []
    for contour in contours:
        if len(contour) >= 3:
            result.append(contour.reshape(-1, 2))
    
    return result, hierarchy


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Dilate mask to close small gaps.
    
    Args:
        mask: Binary mask
        iterations: Number of dilation iterations
        
    Returns:
        Dilated mask
    """
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0
