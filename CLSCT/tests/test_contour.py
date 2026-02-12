"""Tests for contour detection module."""

import numpy as np
import pytest

from apx_clsct.contour import find_contours, dilate_mask


class TestFindContours:
    """Test cases for find_contours function."""
    
    def test_find_square_contour(self):
        """Test finding contour of a square."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        # Draw a square
        mask[10:40, 10:40] = 255
        
        contours = find_contours(mask)
        
        assert len(contours) == 1
        assert len(contours[0]) >= 4  # At least 4 corner points
    
    def test_empty_mask(self):
        """Test finding contours in empty mask."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        
        contours = find_contours(mask)
        
        assert len(contours) == 0
    
    def test_multiple_contours(self):
        """Test finding multiple separate contours."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Two separate squares
        mask[10:30, 10:30] = 255
        mask[60:80, 60:80] = 255
        
        contours = find_contours(mask)
        
        assert len(contours) == 2


class TestDilateMask:
    """Test cases for dilate_mask function."""
    
    def test_dilate_expands_mask(self):
        """Test that dilation expands the mask."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True
        
        dilated = dilate_mask(mask, iterations=1)
        
        # Dilated mask should be larger
        original_area = np.sum(mask)
        dilated_area = np.sum(dilated)
        assert dilated_area > original_area
