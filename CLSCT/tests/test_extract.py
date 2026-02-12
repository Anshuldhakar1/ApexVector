"""Tests for layer extraction module."""

import numpy as np

from apx_clsct.extract import extract_color_layers, create_color_mask, clean_mask


class TestExtractColorLayers:
    """Test cases for extract_color_layers function."""
    
    def test_extract_layers(self):
        """Test extracting layers from quantized image."""
        # Create image with 2 colors
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[:, :25] = [255, 0, 0]  # Red left half
        image[:, 25:] = [0, 255, 0]  # Green right half
        
        palette = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        
        layers = extract_color_layers(image, palette)
        
        assert len(layers) == 2
        
        # Check first layer (red)
        color1, mask1 = layers[0]
        assert color1 == (255, 0, 0) or color1 == (0, 255, 0)
        assert mask1.shape == (50, 50)
        assert mask1.dtype == bool


class TestCreateColorMask:
    """Test cases for create_color_mask function."""
    
    def test_create_mask(self):
        """Test creating a color mask."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:5, :] = [255, 0, 0]
        image[5:, :] = [0, 255, 0]
        
        mask = create_color_mask(image, (255, 0, 0))
        
        assert mask.shape == (10, 10)
        assert mask[:5, :].all()
        assert not mask[5:, :].any()


class TestCleanMask:
    """Test cases for clean_mask function."""
    
    def test_clean_small_regions(self):
        """Test removing small regions from mask."""
        mask = np.zeros((50, 50), dtype=bool)
        # Large region
        mask[10:40, 10:40] = True
        # Small noise
        mask[0, 0] = True
        mask[1, 1] = True
        
        cleaned = clean_mask(mask, min_area=10)
        
        assert not cleaned[0, 0]
        assert not cleaned[1, 1]
        assert cleaned[10:40, 10:40].all()
