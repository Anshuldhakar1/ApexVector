"""Tests for SVG generation module."""

import numpy as np
import pytest
from xml.etree import ElementTree as ET

from apx_clsct.svg import (
    color_to_hex,
    contour_to_line_path,
    contour_to_bezier_path,
    contours_to_svg
)


class TestColorToHex:
    """Test cases for color_to_hex function."""
    
    def test_rgb_to_hex(self):
        """Test RGB to hex conversion."""
        assert color_to_hex((255, 0, 0)) == "#FF0000"
        assert color_to_hex((0, 255, 0)) == "#00FF00"
        assert color_to_hex((0, 0, 255)) == "#0000FF"
        assert color_to_hex((255, 255, 255)) == "#FFFFFF"
    
    def test_rgba_to_hex(self):
        """Test RGBA to hex conversion (alpha ignored)."""
        assert color_to_hex((255, 0, 0, 128)) == "#FF0000"


class TestContourToPath:
    """Test cases for path conversion functions."""
    
    def test_line_path(self):
        """Test converting contour to line path."""
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        path = contour_to_line_path(contour)
        
        assert path.startswith("M")
        assert "Z" in path
        assert "L" in path
    
    def test_bezier_path(self):
        """Test converting contour to bezier path."""
        contour = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        path = contour_to_bezier_path(contour)
        
        assert path.startswith("M")
        assert "Z" in path
        assert "C" in path  # Cubic bezier command


class TestContoursToSvg:
    """Test cases for contours_to_svg function."""
    
    def test_basic_svg(self):
        """Test generating basic SVG."""
        layers = [
            ((255, 0, 0), [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]),
        ]
        
        svg = contours_to_svg(layers, 100, 100)
        
        assert svg.startswith("<svg")
        assert "</svg>" in svg
        assert "path" in svg
        assert '#FF0000' in svg
    
    def test_valid_xml(self):
        """Test that generated SVG is valid XML."""
        layers = [
            ((255, 0, 0), [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]),
        ]
        
        svg = contours_to_svg(layers, 100, 100)
        
        # Should parse without error
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")
