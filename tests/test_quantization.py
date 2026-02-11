"""Tests for SLIC superpixel quantization module."""
import pytest
import numpy as np
from pathlib import Path

# Import the module that doesn't exist yet
from apexvec.quantization import slic_quantization


class TestSLICQuantization:
    """Test SLIC superpixel quantization."""
    
    def test_slic_quantization_basic(self):
        """Test that SLIC creates spatially coherent regions without salt-and-pepper."""
        # Create test image with smooth gradients
        h, w = 64, 64
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Create smooth gradient from red to blue
        for i in range(h):
            for j in range(w):
                image[i, j] = [
                    int(255 * (1 - i/h)),  # Red gradient
                    0,                     # No green
                    int(255 * (i/h))      # Blue gradient
                ]
        
        # Apply SLIC quantization
        label_map, palette = slic_quantization(
            image, 
            n_segments=16, 
            compactness=10.0,
            sigma=1.0
        )
        
        # Check output shapes
        assert label_map.shape == (h, w)
        assert palette.shape == (16, 3)
        assert palette.dtype == np.uint8
        
        # Check that labels are in valid range
        assert np.all(label_map >= 0)
        assert np.all(label_map < 16)
        
        # Check spatial coherence: neighboring pixels should often have same label
        # This should be much higher than random assignment (1/16 = 6.25%)
        same_label_neighbors = 0
        total_neighbors = 0
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                current_label = label_map[i, j]
                neighbors = [
                    label_map[i-1, j], label_map[i+1, j],
                    label_map[i, j-1], label_map[i, j+1]
                ]
                same_label_neighbors += sum(1 for n in neighbors if n == current_label)
                total_neighbors += 4
        
        coherence_ratio = same_label_neighbors / total_neighbors
        assert coherence_ratio > 0.3, f"Coherence ratio {coherence_ratio:.3f} too low"
        
        # Check no salt-and-pepper: isolated pixels should be rare
        isolated_pixels = 0
        for i in range(1, h-1):
            for j in range(1, w-1):
                current_label = label_map[i, j]
                neighbors = [
                    label_map[i-1, j], label_map[i+1, j],
                    label_map[i, j-1], label_map[i, j+1]
                ]
                if all(n != current_label for n in neighbors):
                    isolated_pixels += 1
        
        isolated_ratio = isolated_pixels / (h * w)
        assert isolated_ratio < 0.05, f"Isolated pixel ratio {isolated_ratio:.3f} too high"
    
    def test_slic_preserves_palette(self):
        """Test that SLIC preserves palette size and all labels are used."""
        # Create test image with distinct color regions
        h, w = 32, 32
        image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Four quadrants with different colors
        image[:h//2, :w//2] = [255, 0, 0]    # Red
        image[:h//2, w//2:] = [0, 255, 0]    # Green  
        image[h//2:, :w//2] = [0, 0, 255]    # Blue
        image[h//2:, w//2:] = [255, 255, 0]  # Yellow
        
        # Apply SLIC with 8 segments
        label_map, palette = slic_quantization(
            image,
            n_segments=8,
            compactness=5.0,
            sigma=1.0
        )
        
        # Check palette size
        assert palette.shape == (8, 3)
        assert palette.dtype == np.uint8
        
        # Check that all labels are in range
        assert np.all(label_map >= 0)
        assert np.all(label_map < 8)
        
        # Check that most labels are actually used (allow some flexibility)
        unique_labels = np.unique(label_map)
        assert len(unique_labels) >= 4, f"Only {len(unique_labels)} labels used, expected >=4"
        assert len(unique_labels) <= 8, f"{len(unique_labels)} labels used, expected <=8"
        
        # Check that palette colors are reasonable (not all black or white)
        assert np.any(palette > 50), "Palette colors too dark"
        assert np.any(palette < 200), "Palette colors too light"
        
        # Check that each used label has a corresponding palette color
        for label in unique_labels:
            color = palette[label]
            assert np.any(color > 0), f"Label {label} has black color"
            assert np.any(color < 255), f"Label {label} has white color"
    
    def test_slic_different_parameters(self):
        """Test SLIC with different parameter combinations."""
        # Create simple test image
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # Test different numbers of segments
        for n_segments in [4, 8, 16]:
            label_map, palette = slic_quantization(
                image, 
                n_segments=n_segments,
                compactness=10.0,
                sigma=1.0
            )
            
            assert label_map.shape == (32, 32)
            assert palette.shape == (n_segments, 3)
            assert np.all(label_map >= 0)
            assert np.all(label_map < n_segments)
        
        # Test different compactness values
        for compactness in [1.0, 10.0, 20.0]:
            label_map, palette = slic_quantization(
                image,
                n_segments=8,
                compactness=compactness,
                sigma=1.0
            )
            
            assert label_map.shape == (32, 32)
            assert palette.shape == (8, 3)
            # Higher compactness should create more coherent regions
            coherence = self._measure_coherence(label_map)
            # This is a rough check - exact behavior depends on image content
            assert coherence > 0.1, f"Coherence {coherence:.3f} too low for compactness {compactness}"
    
    def _measure_coherence(self, label_map):
        """Helper to measure spatial coherence of label map."""
        h, w = label_map.shape
        same_label_neighbors = 0
        total_neighbors = 0
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                current_label = label_map[i, j]
                neighbors = [
                    label_map[i-1, j], label_map[i+1, j],
                    label_map[i, j-1], label_map[i, j+1]
                ]
                same_label_neighbors += sum(1 for n in neighbors if n == current_label)
                total_neighbors += 4
        
        return same_label_neighbors / total_neighbors if total_neighbors > 0 else 0