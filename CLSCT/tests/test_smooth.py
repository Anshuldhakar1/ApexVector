"""Tests for curve smoothing module."""

import numpy as np

from apx_clsct.smooth import (
    gaussian_smooth_contour,
    smooth_contour_bspline,
    smooth_contour_catmull_rom
)


class TestGaussianSmooth:
    """Test cases for gaussian_smooth_contour function."""
    
    def test_preserves_shape(self):
        """Test that smoothing preserves point count."""
        angles = np.linspace(0, 2*np.pi, 50)
        contour = np.column_stack([
            50 + 30 * np.cos(angles),
            50 + 30 * np.sin(angles)
        ])
        
        smoothed = gaussian_smooth_contour(contour, sigma=1.0)
        
        assert smoothed.shape == contour.shape
    
    def test_reduces_noise(self):
        """Test that smoothing reduces high-frequency noise."""
        # Create a circle with noise
        angles = np.linspace(0, 2*np.pi, 100)
        noise = np.random.normal(0, 2, 100)
        contour = np.column_stack([
            50 + (30 + noise) * np.cos(angles),
            50 + (30 + noise) * np.sin(angles)
        ])
        
        smoothed = gaussian_smooth_contour(contour, sigma=2.0)
        
        # Smoothed should have less variance
        original_variance = np.var(np.linalg.norm(contour - np.mean(contour, axis=0), axis=1))
        smoothed_variance = np.var(np.linalg.norm(smoothed - np.mean(smoothed, axis=0), axis=1))
        
        assert smoothed_variance < original_variance


class TestBsplineSmooth:
    """Test cases for smooth_contour_bspline function."""
    
    def test_bspline_smooth(self):
        """Test B-spline smoothing."""
        angles = np.linspace(0, 2*np.pi, 20)
        contour = np.column_stack([
            50 + 30 * np.cos(angles),
            50 + 30 * np.sin(angles)
        ])
        
        smoothed = smooth_contour_bspline(contour, smoothness=3.0)
        
        # Should produce more points
        assert len(smoothed) >= len(contour)


class TestCatmullRom:
    """Test cases for smooth_contour_catmull_rom function."""
    
    def test_catmull_rom_smooth(self):
        """Test Catmull-Rom spline smoothing."""
        angles = np.linspace(0, 2*np.pi, 20)
        contour = np.column_stack([
            50 + 30 * np.cos(angles),
            50 + 30 * np.sin(angles)
        ])
        
        smoothed = smooth_contour_catmull_rom(contour, num_points=50)
        
        # Should produce at least as many points as input
        assert len(smoothed) >= len(contour)
