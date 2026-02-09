"""Tests for poster-style vectorization modules."""
import pytest
import numpy as np
from pathlib import Path

from apexvec.types import ApexConfig, BezierPath, BezierCurve, Point, Region
from apexvec.color_quantizer import quantize_colors, rgb_to_lab, lab_to_rgb
from apexvec.region_extractor import extract_regions, delta_e_2000
from apexvec.boundary_smoother import (
    extract_contours, fit_periodic_spline, spline_to_bezier, smooth_region_boundaries
)
from apexvec.svg_export import format_color, format_number, generate_svg
from apexvec.poster_pipeline import PosterPipeline


class TestColorQuantizer:
    """Test color quantization."""
    
    def test_rgb_lab_conversion(self):
        """Test RGB to LAB conversion."""
        rgb = np.array([[[1.0, 0.0, 0.0]]])  # Red
        lab = rgb_to_lab(rgb)
        rgb_back = lab_to_rgb(lab)
        
        assert rgb_back.shape == rgb.shape
        assert np.allclose(rgb, rgb_back, atol=0.1)
    
    def test_quantize_colors(self):
        """Test color quantization."""
        # Create test image with 3 distinct colors
        image = np.zeros((64, 64, 3))
        image[:21, :] = [1.0, 0.0, 0.0]  # Red
        image[21:42, :] = [0.0, 1.0, 0.0]  # Green
        image[42:, :] = [0.0, 0.0, 1.0]  # Blue
        
        label_map, palette = quantize_colors(image, n_colors=3, n_init=3)
        
        assert label_map.shape == (64, 64)
        assert len(palette) == 3
        assert palette.shape == (3, 3)
        assert np.all((palette >= 0) & (palette <= 1))


class TestRegionExtractor:
    """Test region extraction."""
    
    def test_delta_e_2000(self):
        """Test Delta E calculation."""
        lab1 = np.array([50.0, 0.0, 0.0])
        lab2 = np.array([50.0, 10.0, 0.0])
        
        de = delta_e_2000(lab1, lab2)
        assert de > 0
        assert de < 100
    
    def test_extract_regions(self):
        """Test region extraction."""
        # Create simple test case
        label_map = np.zeros((32, 32), dtype=int)
        label_map[:16, :] = 0
        label_map[16:, :] = 1
        
        palette = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        image = np.zeros((32, 32, 3))
        
        config = ApexConfig(n_colors=2, min_region_area_ratio=0.01)
        
        regions = extract_regions(label_map, palette, image, config)
        
        assert len(regions) >= 1
        assert all(isinstance(r, Region) for r in regions)


class TestBoundarySmoother:
    """Test boundary smoothing."""
    
    def test_extract_contours(self):
        """Test contour extraction."""
        # Create simple circle mask
        mask = np.zeros((64, 64), dtype=bool)
        y, x = np.ogrid[:64, :64]
        mask[(x - 32)**2 + (y - 32)**2 <= 400] = True
        
        contours = extract_contours(mask)
        
        assert len(contours) >= 1
        assert all(isinstance(c, np.ndarray) for c in contours)
        assert all(c.shape[1] == 2 for c in contours)  # (N, 2) format
    
    def test_fit_periodic_spline(self):
        """Test spline fitting."""
        # Create circular points
        theta = np.linspace(0, 2*np.pi, 50)
        points = np.column_stack([
            32 + 20 * np.cos(theta),
            32 + 20 * np.sin(theta)
        ])
        
        t, spline = fit_periodic_spline(points, smoothness=0.5, n_samples=20)
        
        assert len(t) == 20
        assert spline.shape == (20, 2)
    
    def test_spline_to_bezier(self):
        """Test spline to Bezier conversion."""
        spline = np.array([
            [0, 0],
            [10, 0],
            [10, 10],
            [0, 10]
        ])
        t = np.linspace(0, 1, 4)
        
        curves = spline_to_bezier(t, spline)
        
        assert len(curves) == 3
        assert all(isinstance(c, BezierCurve) for c in curves)
    
    def test_smooth_region_boundaries(self):
        """Test boundary smoothing."""
        # Create simple region
        mask = np.zeros((64, 64), dtype=bool)
        y, x = np.ogrid[:64, :64]
        mask[(x - 32)**2 + (y - 32)**2 <= 400] = True
        
        region = Region(mask=mask, label=0)
        config = ApexConfig(spline_smoothness=0.5)
        
        paths = smooth_region_boundaries([region], config)
        
        assert len(paths) == 1
        assert isinstance(paths[0], BezierPath)


class TestSvgExport:
    """Test SVG export."""
    
    def test_format_color(self):
        """Test color formatting."""
        # Test with primary red
        color = np.array([1.0, 0.0, 0.0])
        result = format_color(color)
        assert result.startswith('#')
        assert len(result) in [4, 7]  # #RGB or #RRGGBB
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1.5, 2) == '1.5'
        assert format_number(1.50, 2) == '1.5'
        assert format_number(2.0, 2) == '2'
    
    def test_generate_svg(self):
        """Test SVG generation."""
        # Create simple Bezier path
        curves = [
            BezierCurve(
                Point(0, 0), Point(10, 0), Point(20, 10), Point(30, 10)
            )
        ]
        path = BezierPath(curves=curves, is_closed=False)
        
        # Create region
        mask = np.zeros((64, 64), dtype=bool)
        mask[:32, :32] = True
        region = Region(
            mask=mask,
            label=0,
            mean_color=np.array([1.0, 0.0, 0.0])
        )
        
        config = ApexConfig(precision=2)
        
        svg = generate_svg([path], [region], 64, 64, config)
        
        assert svg.startswith('<?xml version="1.0"')
        assert '<svg' in svg
        assert '</svg>' in svg
        assert '<path' in svg


class TestPosterPipeline:
    """Test poster pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        config = ApexConfig(n_colors=8)
        pipeline = PosterPipeline(config)
        
        assert pipeline.config.n_colors == 8
    
    def test_pipeline_validate(self, tmp_path):
        """Test pipeline validation."""
        # Create test image
        from PIL import Image
        
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (32, 32), color='red')
        img.save(img_path)
        
        # Create simple SVG
        svg = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <path d="M0,0 L32,0 L32,32 L0,32 Z" fill="#f00"/>
</svg>'''
        
        pipeline = PosterPipeline()
        results = pipeline.validate(img_path, svg, max_regions=20)
        
        assert 'region_count' in results
        assert 'overall_pass' in results
