"""Tests for color quantization module."""

import numpy as np
import pytest
from PIL import Image

from apx_clsct.quantize import quantize_colors, get_dominant_colors
from apx_clsct.types import QuantizationError


class TestQuantizeColors:
    """Test cases for quantize_colors function."""
    
    def test_basic_quantization(self):
        """Test basic color quantization."""
        # Create a simple test image with 4 distinct colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :50] = [255, 0, 0]  # Red
        image[:50, 50:] = [0, 255, 0]  # Green
        image[50:, :50] = [0, 0, 255]  # Blue
        image[50:, 50:] = [255, 255, 0]  # Yellow
        
        quantized, palette = quantize_colors(image, 4)
        
        assert quantized.shape == image.shape
        assert len(palette) == 4
        assert palette.shape[1] == 3  # RGB channels
    
    def test_n_colors_validation(self):
        """Test that n_colors < 2 raises ValueError."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="n_colors must be >= 2"):
            quantize_colors(image, 1)
    
    def test_empty_image(self):
        """Test that empty image raises QuantizationError."""
        image = np.array([])
        
        with pytest.raises(QuantizationError):
            quantize_colors(image, 4)
    
    def test_reproducibility(self):
        """Test that quantization is reproducible with same random_state."""
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        quantized1, palette1 = quantize_colors(image, 8, random_state=42)
        quantized2, palette2 = quantize_colors(image, 8, random_state=42)
        
        np.testing.assert_array_equal(quantized1, quantized2)
        np.testing.assert_array_equal(palette1, palette2)


class TestDominantColors:
    """Test cases for get_dominant_colors function."""
    
    def test_dominant_colors(self):
        """Test extracting dominant colors."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[:25, :] = [100, 150, 200]
        image[25:, :] = [200, 100, 50]
        
        colors = get_dominant_colors(image, 2)
        
        assert len(colors) == 2
        assert colors.shape[1] == 3
    
    def test_dominant_colors_validation(self):
        """Test that n_colors < 2 raises ValueError."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="n_colors must be >= 2"):
            get_dominant_colors(image, 1)
