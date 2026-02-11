"""Tests for Marching Squares boundary extraction."""
import pytest
import numpy as np
from apexvec.boundary_extraction import extract_shared_boundaries, build_adjacency_graph


class TestMarchingSquares:
    """Test Marching Squares boundary extraction."""

    def test_marching_squares_basic(self):
        """Test Marching Squares extracts boundaries between two regions."""
        # Simple two-region image
        label_map = np.zeros((10, 10), dtype=np.int32)
        label_map[:, :5] = 0  # Left
        label_map[:, 5:] = 1  # Right

        boundaries = extract_shared_boundaries(label_map)

        # Should have one boundary between labels 0 and 1
        assert (0, 1) in boundaries or (1, 0) in boundaries

        # Boundary should be roughly vertical around x=4.5
        boundary = boundaries.get((0, 1)) or boundaries.get((1, 0))
        assert boundary is not None

        # Check boundary points are at sub-pixel positions
        xs = [p[0] for p in boundary]
        assert all(4.0 <= x <= 6.0 for x in xs), "Boundary should be near x=5"

    def test_marching_squares_handles_holes(self):
        """Test Marching Squares correctly extracts hole boundaries."""
        # Create region with hole
        label_map = np.ones((20, 20), dtype=np.int32)
        label_map[5:15, 5:15] = 0  # Outer region 0
        label_map[8:12, 8:12] = 1  # Hole region 1 inside

        boundaries = extract_shared_boundaries(label_map)

        # Should have boundary between 0 and 1
        assert (0, 1) in boundaries or (1, 0) in boundaries

        # Boundary should form a closed loop
        boundary = boundaries.get((0, 1)) or boundaries.get((1, 0))
        # First and last points should be close (closed loop)
        first = np.array(boundary[0])
        last = np.array(boundary[-1])
        distance = np.linalg.norm(first - last)
        assert distance < 2.0, "Boundary should form closed loop"

    def test_three_region_boundaries(self):
        """Test extraction with three regions."""
        label_map = np.zeros((15, 15), dtype=np.int32)
        label_map[:5, :] = 0  # Top
        label_map[5:10, :] = 1  # Middle
        label_map[10:, :] = 2  # Bottom

        boundaries = extract_shared_boundaries(label_map)

        # Should have boundaries between 0-1 and 1-2
        assert len(boundaries) >= 2

    def test_adjacency_graph(self):
        """Test adjacency graph construction."""
        label_map = np.zeros((10, 10), dtype=np.int32)
        label_map[:, :5] = 0
        label_map[:, 5:] = 1

        adjacency = build_adjacency_graph(label_map)

        # Should have one adjacency between 0 and 1
        assert (0, 1) in adjacency or (1, 0) in adjacency

    def test_empty_label_map(self):
        """Test handling of empty/single label maps."""
        # Single label
        label_map = np.zeros((10, 10), dtype=np.int32)
        boundaries = extract_shared_boundaries(label_map)
        assert len(boundaries) == 0

        # Two labels with no shared boundary (completely separate)
        label_map = np.zeros((10, 10), dtype=np.int32)
        label_map[0:5, 0:5] = 0
        label_map[5:10, 5:10] = 1
        boundaries = extract_shared_boundaries(label_map)
        # May or may not have boundary depending on diagonal adjacency
