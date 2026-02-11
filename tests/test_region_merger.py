"""Tests for same-color region merging."""
import pytest
import numpy as np
from apexvec.types import Region
from apexvec.region_merger import merge_small_regions_same_color


class TestSameColorRegionMerging:
    """Test same-color-only region merging."""

    def test_merge_same_color_only(self):
        """Test that merging only happens within same color."""
        # Create label map with two colors
        label_map = np.zeros((20, 20), dtype=np.int32)
        label_map[:, :10] = 0  # Color 0 left
        label_map[:, 10:] = 1  # Color 1 right

        # Add small region of color 0 inside color 1 area
        label_map[5:7, 12:14] = 0  # Small red island in blue

        # Create regions
        regions = []
        for label_id in [0, 1]:
            # Find connected components for this label
            from scipy.ndimage import label
            mask = (label_map == label_id)
            labeled, num = label(mask)
            for i in range(1, num + 1):
                comp_mask = (labeled == i)
                region = Region(
                    mask=comp_mask,
                    label=len(regions),
                    mean_color=np.array([255, 0, 0] if label_id == 0 else [0, 0, 255])
                )
                region.color_label = label_id  # Track original color
                regions.append(region)

        # Merge with threshold of 10 pixels
        merged = merge_small_regions_same_color(regions, min_area=10)

        # Small red region should merge with larger red region
        # NOT with blue regions
        red_regions = [r for r in merged if np.array_equal(r.mean_color, [255, 0, 0])]
        blue_regions = [r for r in merged if np.array_equal(r.mean_color, [0, 0, 255])]

        assert len(red_regions) == 1, "Red regions should merge into one"
        assert len(blue_regions) == 1, "Blue region should remain"

    def test_respect_area_threshold(self):
        """Test that regions above threshold are not merged."""
        # Create two separate red regions, both large
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[10:20, 10:20] = 0  # Region 1: 100 pixels
        label_map[30:40, 30:40] = 0  # Region 2: 100 pixels

        regions = []
        from scipy.ndimage import label
        mask = (label_map == 0)
        labeled, num = label(mask)
        for i in range(1, num + 1):
            comp_mask = (labeled == i)
            region = Region(mask=comp_mask, label=i-1, mean_color=np.array([255, 0, 0]))
            region.color_label = 0
            regions.append(region)

        # Merge with threshold of 50
        merged = merge_small_regions_same_color(regions, min_area=50)

        # Both regions should remain (both > 50 pixels)
        assert len(merged) == 2, "Both large regions should be preserved"

    def test_empty_regions(self):
        """Test handling of empty region list."""
        merged = merge_small_regions_same_color([], min_area=100)
        assert merged == []

    def test_single_region(self):
        """Test handling of single region."""
        region = Region(
            mask=np.ones((10, 10), dtype=bool),
            label=0,
            mean_color=np.array([255, 0, 0])
        )
        region.color_label = 0

        merged = merge_small_regions_same_color([region], min_area=100)
        assert len(merged) == 1
        assert merged[0].label == 0

    def test_isolated_small_region(self):
        """Test that isolated small regions without neighbors are kept."""
        # Create isolated small region
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:7, 5:7] = True  # 4 pixels

        region = Region(mask=mask, label=0, mean_color=np.array([255, 0, 0]))
        region.color_label = 0

        # Merge with threshold of 10
        merged = merge_small_regions_same_color([region], min_area=10)

        # Should keep the isolated region
        assert len(merged) == 1
