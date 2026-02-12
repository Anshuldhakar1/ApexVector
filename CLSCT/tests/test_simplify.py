"""Tests for curve simplification module."""

import numpy as np

from apx_clsct.simplify import simplify_contour, simplify_contours, adaptive_simplify


class TestSimplifyContour:
    """Test cases for simplify_contour function."""
    
    def test_simplify_straight_line(self):
        """Test simplifying a straight line (should reduce points)."""
        # Create a straight horizontal line with many points
        contour = np.array([[i, 10] for i in range(100)])
        
        simplified = simplify_contour(contour, epsilon_factor=0.01)
        
        # Should be reduced to just endpoints
        assert len(simplified) < len(contour)
        assert len(simplified) == 2
    
    def test_simplify_square(self):
        """Test simplifying a square contour."""
        # Create a square with many points per side
        top = [[i, 0] for i in range(100)]
        right = [[100, i] for i in range(100)]
        bottom = [[i, 100] for i in range(100, 0, -1)]
        left = [[0, i] for i in range(100, 0, -1)]
        
        contour = np.array(top + right + bottom + left)
        
        simplified = simplify_contour(contour, epsilon_factor=0.01)
        
        # Should reduce to roughly 4 corners (possibly with some extra)
        assert len(simplified) <= 8  # Allow some flexibility
        assert len(simplified) >= 4
    
    def test_small_contour_unchanged(self):
        """Test that small contours are returned as-is."""
        contour = np.array([[0, 0], [10, 0], [10, 10]])
        
        simplified = simplify_contour(contour)
        
        np.testing.assert_array_equal(simplified, contour)


class TestAdaptiveSimplify:
    """Test cases for adaptive_simplify function."""
    
    def test_adaptive_simplify(self):
        """Test adaptive simplification to target point count."""
        # Create a circle with many points
        angles = np.linspace(0, 2*np.pi, 100)
        contour = np.column_stack([
            50 + 30 * np.cos(angles),
            50 + 30 * np.sin(angles)
        ])
        
        simplified = adaptive_simplify(contour, target_points=20)
        
        # Should be close to target (allow some flexibility)
        assert 10 <= len(simplified) <= 50
