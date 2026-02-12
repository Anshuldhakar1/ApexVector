"""Tests for SVG to PNG rasterizer utility.

Note: This is a testing utility, not part of the user-facing pipeline.
These tests are skipped if matplotlib/cairosvg are not installed.
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image

from apx_clsct.pipeline import Pipeline, PipelineConfig


# Check if rendering backends are available
has_cairosvg = False
has_matplotlib = False

try:
    import cairosvg
    has_cairosvg = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    pass

skip_if_no_renderer = pytest.mark.skipif(
    not (has_cairosvg or has_matplotlib),
    reason="Requires cairosvg or matplotlib for SVG rendering"
)


class TestSVGRasterizer:
    """Test SVG to PNG rasterization for visualization."""
    
    @skip_if_no_renderer
    def test_rasterize_simple_svg(self, output_dir, svg_rasterizer):
        """Test rasterizing a simple SVG file."""
        # Create a simple test SVG
        svg_content = '''<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <rect x="10" y="10" width="80" height="80" fill="red"/>
</svg>'''
        svg_path = output_dir / "test_rect.svg"
        svg_path.write_text(svg_content)
        
        # Convert to PNG
        svg_to_png = svg_rasterizer["svg_to_png"]
        png_path = svg_to_png(svg_path)
        
        assert png_path.exists()
        assert png_path.suffix == ".png"
        
        # Verify it's a valid image
        img = Image.open(png_path)
        assert img.size[0] > 0
        assert img.size[1] > 0
    
    @skip_if_no_renderer
    def test_rasterize_pipeline_output(self, test_img0, output_dir, svg_rasterizer):
        """Test rasterizing actual pipeline output for visual comparison."""
        svg_path = output_dir / "rasterizer_test.svg"
        png_path = output_dir / "rasterizer_test.png"
        
        # Generate SVG
        config = PipelineConfig(n_colors=6)
        pipeline = Pipeline(config)
        pipeline.process(str(test_img0), str(svg_path))
        
        # Convert to PNG
        svg_to_png = svg_rasterizer["svg_to_png"]
        result = svg_to_png(svg_path, png_path)
        
        assert result == png_path
        assert png_path.exists()
        print(f"\nGenerated PNG: {png_path}")
    
    @skip_if_no_renderer
    def test_convert_multiple_svgs(self, test_img0, test_img1, output_dir, svg_rasterizer):
        """Test converting multiple SVGs for comparison testing."""
        convert_folder_svgs = svg_rasterizer["convert_folder_svgs"]
        
        # Generate SVGs with different configs
        configs = [
            ("colors_4", PipelineConfig(n_colors=4)),
            ("colors_8", PipelineConfig(n_colors=8)),
            ("smooth_bspline", PipelineConfig(n_colors=6, smooth_method="bspline")),
        ]
        
        for name, config in configs:
            svg_path = output_dir / f"comparison_{name}.svg"
            pipeline = Pipeline(config)
            pipeline.process(str(test_img0), str(svg_path))
        
        # Convert all to PNG
        png_files = convert_folder_svgs(output_dir, "comparison_*.svg")
        
        assert len(png_files) == len(configs)
        for png_file in png_files:
            assert png_file.exists()
            print(f"\nConverted: {png_file}")
    
    @skip_if_no_renderer
    def test_rasterize_with_custom_size(self, output_dir, svg_rasterizer):
        """Test rasterizing with custom output size."""
        # Create test SVG
        svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <circle cx="50" cy="50" r="40" fill="blue"/>
</svg>'''
        svg_path = output_dir / "test_circle.svg"
        svg_path.write_text(svg_content)
        
        svg_to_png = svg_rasterizer["svg_to_png"]
        png_path = output_dir / "test_circle_200x200.png"
        
        # Convert with custom size
        svg_to_png(svg_path, png_path, size=(200, 200))
        
        img = Image.open(png_path)
        # Allow some tolerance due to matplotlib padding
        assert img.size[0] >= 180
        assert img.size[1] >= 180


def visualize_outputs(image_path: Path, output_dir: Path, svg_rasterizer: dict):
    """Helper function to visualize pipeline outputs.
    
    This is a development utility for comparing different configurations.
    Not part of the test suite.
    
    Usage:
        visualize_outputs(Path("input.jpg"), Path("output"), svg_rasterizer)
    """
    svg_to_png = svg_rasterizer["svg_to_png"]
    
    # Test different color counts
    for n_colors in [4, 8, 12]:
        config = PipelineConfig(n_colors=n_colors)
        pipeline = Pipeline(config)
        
        svg_path = output_dir / f"viz_colors_{n_colors}.svg"
        png_path = output_dir / f"viz_colors_{n_colors}.png"
        
        pipeline.process(str(image_path), str(svg_path))
        svg_to_png(svg_path, png_path)
        print(f"Created: {png_path}")
    
    # Test different smoothing methods
    for method in ["gaussian", "bspline", "none"]:
        config = PipelineConfig(n_colors=8, smooth_method=method)
        pipeline = Pipeline(config)
        
        svg_path = output_dir / f"viz_smooth_{method}.svg"
        png_path = output_dir / f"viz_smooth_{method}.png"
        
        pipeline.process(str(image_path), str(svg_path))
        svg_to_png(svg_path, png_path)
        print(f"Created: {png_path}")
