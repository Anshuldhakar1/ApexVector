"""Integration tests for the full pipeline."""

import pytest
from pathlib import Path
from xml.etree import ElementTree as ET

from apx_clsct.pipeline import Pipeline, PipelineConfig, process_image
from apx_clsct.types import VectorizationError


class TestPipelineIntegration:
    """Integration tests using actual images."""

    def test_pipeline_img0(self, test_img0, output_dir):
        """Test pipeline with img0.jpg."""
        output_file = output_dir / "img0_test_output.svg"

        config = PipelineConfig(n_colors=8)
        pipeline = Pipeline(config)

        svg = pipeline.process(str(test_img0), str(output_file))

        # Verify output
        assert output_file.exists()
        assert svg.startswith("<svg")

        # Validate XML
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

        # Check that paths were generated
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) > 0

    def test_pipeline_img1(self, test_img1, output_dir):
        """Test pipeline with img1.jpg."""
        output_file = output_dir / "img1_test_output.svg"

        config = PipelineConfig(n_colors=10)
        pipeline = Pipeline(config)

        svg = pipeline.process(str(test_img1), str(output_file))

        # Verify output
        assert output_file.exists()
        assert svg.startswith("<svg")

        # Validate XML
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_pipeline_different_color_counts(self, test_img0, output_dir):
        """Test pipeline with different color counts."""
        for n_colors in [4, 8, 12]:
            output_file = output_dir / f"img0_colors_{n_colors}.svg"

            config = PipelineConfig(n_colors=n_colors)
            pipeline = Pipeline(config)

            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()
            assert len(svg) > 0

    def test_pipeline_smooth_methods(self, test_img0, output_dir):
        """Test pipeline with different smoothing methods."""
        for method in ["gaussian", "bspline", "none"]:
            output_file = output_dir / f"img0_smooth_{method}.svg"

            config = PipelineConfig(n_colors=8, smooth_method=method)
            pipeline = Pipeline(config)

            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()
            assert len(svg) > 0

    def test_pipeline_debug_mode(self, test_img0, output_dir):
        """Test pipeline with debug mode enabled."""
        output_file = output_dir / "img0_debug.svg"

        config = PipelineConfig(n_colors=8)
        pipeline = Pipeline(config)

        svg = pipeline.process(str(test_img0), str(output_file), debug=True)

        # Check that debug stages were collected
        assert len(pipeline.debug_stages) > 0
        assert pipeline.debug_stages[0][0] == "1_original"


class TestProcessImage:
    """Test cases for process_image convenience function."""

    def test_process_image(self, test_img0, output_dir):
        """Test convenience function."""
        output_file = output_dir / "convenience_test.svg"

        config = PipelineConfig(n_colors=8)
        svg = process_image(str(test_img0), str(output_file), config=config)

        assert output_file.exists()
        assert svg.startswith("<svg")

    def test_process_image_no_output(self, test_img0):
        """Test convenience function without output path."""
        config = PipelineConfig(n_colors=8)
        svg = process_image(str(test_img0), config=config)

        assert svg.startswith("<svg")


class TestPipelineErrors:
    """Test error handling."""

    def test_missing_file(self):
        """Test handling of missing input file."""
        pipeline = Pipeline()

        with pytest.raises(FileNotFoundError):
            pipeline.process("nonexistent.jpg")
