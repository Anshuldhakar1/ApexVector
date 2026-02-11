"""Integration tests for poster pipeline."""
import pytest
import numpy as np
from pathlib import Path
import tempfile


class TestPosterPipelineIntegration:
    """Full integration tests for poster pipeline."""

    def test_poster_pipeline_end_to_end(self):
        """Full integration test of poster pipeline."""
        from PIL import Image
        from apexvec.poster_pipeline import PosterPipeline
        from apexvec.types import AdaptiveConfig

        # Create test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :] = [100, 150, 200]  # Teal-ish
        img[50:, :] = [250, 240, 230]  # Cream

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.png"
            output_path = Path(tmpdir) / "test.svg"

            Image.fromarray(img).save(input_path)

            # Run pipeline
            config = AdaptiveConfig()
            pipeline = PosterPipeline(
                config=config,
                num_colors=2,
                save_stages=True,
                stages_dir=Path(tmpdir) / "stages"
            )

            svg = pipeline.process(input_path, output_path)

            # Assertions
            assert output_path.exists(), "SVG output not created"
            assert len(svg) > 0, "Empty SVG"

            # Check for paths
            path_count = svg.count("<path")
            assert path_count >= 2, f"Expected 2+ paths, got {path_count}"

    def test_color_fidelity(self):
        """Test that output colors match input palette."""
        from PIL import Image
        from apexvec.poster_pipeline import PosterPipeline
        import tempfile
        from pathlib import Path
        import re

        # Create solid color image
        color = np.array([100, 150, 200], dtype=np.uint8)
        img = np.full((50, 50, 3), color, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "solid.png"
            output_path = Path(tmpdir) / "solid.svg"

            Image.fromarray(img).save(input_path)

            pipeline = PosterPipeline(num_colors=1)
            svg = pipeline.process(input_path, output_path)

            # Extract fill color from SVG
            match = re.search(r'fill="#([0-9a-fA-F]{6})"', svg)
            assert match, "No fill color found in SVG"

            hex_color = match.group(1)
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            # Should be close to input color (within tolerance)
            assert abs(r - color[0]) < 10
            assert abs(g - color[1]) < 10
            assert abs(b - color[2]) < 10

    def test_slic_quantization_integration(self):
        """Test that SLIC quantization is used."""
        from apexvec.poster_pipeline import quantize_colors

        # Create test image
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :25] = [255, 0, 0]  # Red
        img[:, 25:] = [0, 0, 255]  # Blue

        label_map, palette = quantize_colors(img, num_colors=2)

        # Should have 2 colors in palette
        assert len(palette) == 2
        assert palette.shape == (2, 3)

        # Label map should have 2 unique values
        unique_labels = np.unique(label_map)
        assert len(unique_labels) == 2

    def test_same_color_merging_integration(self):
        """Test same-color merging in pipeline."""
        from apexvec.region_merger import merge_small_regions_same_color
        from apexvec.types import Region

        # Create regions with same color
        regions = []
        for i in range(5):
            mask = np.zeros((20, 20), dtype=bool)
            mask[5 + i, 5 + i] = True  # Single pixel regions
            region = Region(
                mask=mask,
                label=i,
                mean_color=np.array([255, 0, 0])
            )
            regions.append(region)

        # Add one large region of same color
        large_mask = np.zeros((20, 20), dtype=bool)
        large_mask[10:15, 10:15] = True
        large_region = Region(
            mask=large_mask,
            label=5,
            mean_color=np.array([255, 0, 0])
        )
        regions.append(large_region)

        # Merge with threshold of 10 pixels
        merged = merge_small_regions_same_color(regions, min_area=10)

        # Should have merged small regions into large one
        assert len(merged) < len(regions)

    def test_transparent_background(self):
        """Test that output has transparent background."""
        from PIL import Image
        from apexvec.poster_pipeline import PosterPipeline
        import tempfile
        from pathlib import Path

        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:] = [255, 0, 0]  # Red

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.png"
            output_path = Path(tmpdir) / "test.svg"

            Image.fromarray(img).save(input_path)

            pipeline = PosterPipeline(num_colors=1)
            svg = pipeline.process(input_path, output_path)

            # Should not have background rect
            assert 'fill="#ffffff"' not in svg or 'fill="white"' not in svg

    def test_no_gaps_basic(self):
        """Basic test that regions cover the image."""
        from PIL import Image
        from apexvec.poster_pipeline import PosterPipeline
        import tempfile
        from pathlib import Path

        # Create two-color image
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :25] = [255, 0, 0]  # Red left
        img[:, 25:] = [0, 0, 255]  # Blue right

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "test.png"
            output_path = Path(tmpdir) / "test.svg"

            Image.fromarray(img).save(input_path)

            pipeline = PosterPipeline(num_colors=2)
            svg = pipeline.process(input_path, output_path)

            # Should have at least 2 paths (one per color)
            path_count = svg.count("<path")
            assert path_count >= 2
