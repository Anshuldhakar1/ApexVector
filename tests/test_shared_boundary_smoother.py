"""Tests for shared boundary smoothing."""
import pytest
import numpy as np
from apexvec.shared_boundary_smoother import (
    smooth_shared_boundary,
    reconstruct_regions_from_boundaries,
    points_to_bezier
)


class TestSharedBoundarySmoothing:
    """Test shared boundary smoothing and reconstruction."""

    def test_gaussian_smooth_shared_boundary(self):
        """Test Gaussian smoothing of shared boundary."""
        # Create simple boundary
        boundary = [(i, 5.0) for i in range(10)]  # Horizontal line

        # Add noise
        np.random.seed(42)
        noisy_boundary = [(x, y + np.random.normal(0, 0.5)) for x, y in boundary]

        # Smooth
        smoothed = smooth_shared_boundary(noisy_boundary, sigma=1.0)

        # Should have same number of points
        assert len(smoothed) == len(noisy_boundary)

        # Should be smoother (less variance in y)
        original_variance = np.var([p[1] for p in noisy_boundary])
        smoothed_variance = np.var([p[1] for p in smoothed])
        assert smoothed_variance < original_variance

    def test_boundary_used_by_both_regions(self):
        """Test that smoothed boundary is used by both adjacent regions."""
        # Simple two-region setup
        boundaries = {
            (0, 1): [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]  # Vertical boundary
        }

        # Mock palette
        palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)

        regions = reconstruct_regions_from_boundaries(
            boundaries,
            palette,
            image_shape=(10, 10)
        )

        # Should have 2 regions
        assert len(regions) == 2

        # Both regions should have paths
        for region in regions:
            assert len(region.path) > 0

    def test_points_to_bezier(self):
        """Test conversion of points to Bezier curves."""
        points = [(0, 0), (10, 0), (10, 10), (0, 10)]

        curves = points_to_bezier(points)

        # Should have n-1 curves for n points
        assert len(curves) == len(points) - 1

        # Check first curve has correct endpoints
        assert curves[0].p0.x == points[0][0]
        assert curves[0].p0.y == points[0][1]
        assert curves[0].p3.x == points[1][0]
        assert curves[0].p3.y == points[1][1]

    def test_small_boundary_no_crash(self):
        """Test that very small boundaries don't crash."""
        # Boundary with < 3 points
        boundary = [(0, 0), (1, 1)]

        smoothed = smooth_shared_boundary(boundary, sigma=1.0)

        # Should return original for small boundaries
        assert smoothed == boundary

    def test_reconstruct_empty_boundaries(self):
        """Test reconstruction with empty boundaries."""
        palette = np.array([[255, 0, 0]], dtype=np.uint8)

        regions = reconstruct_regions_from_boundaries(
            {},
            palette,
            image_shape=(10, 10)
        )

        # Should return empty list
        assert len(regions) == 0
