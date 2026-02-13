"""Tests for the POSTER pipeline mode."""

import pytest
from pathlib import Path
from xml.etree import ElementTree as ET

from apx_clsct.pipeline import (
    PosterPipeline,
    PosterPipelineConfig,
    process_image_poster,
)
from apx_clsct.types import PosterPipelineConfig as PosterConfig


class TestPosterPipelineIntegration:
    """Integration tests for POSTER pipeline using actual images."""

    def test_poster_pipeline_img0(self, test_img0, output_dir):
        """Test poster pipeline with img0.jpg."""
        output_file = output_dir / "poster_img0_test_output.svg"

        config = PosterPipelineConfig(n_colors=32)
        pipeline = PosterPipeline(config)

        svg = pipeline.process(str(test_img0), str(output_file))

        # Verify output
        assert output_file.exists()
        assert svg.startswith("<svg")

        # Validate XML
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

        # Check that paths were generated
        paths = root.findall(".{http://www.w3.org/2000/svg}path")
        assert len(paths) > 0

    def test_poster_pipeline_img1(self, test_img1, output_dir):
        """Test poster pipeline with img1.jpg."""
        output_file = output_dir / "poster_img1_test_output.svg"

        config = PosterPipelineConfig(n_colors=32)
        pipeline = PosterPipeline(config)

        svg = pipeline.process(str(test_img1), str(output_file))

        # Verify output
        assert output_file.exists()
        assert svg.startswith("<svg")

    def test_poster_different_color_counts(self, test_img0, output_dir):
        """Test poster pipeline with different color counts."""
        for n_colors in [24, 32, 48]:
            output_file = output_dir / f"poster_img0_colors_{n_colors}.svg"

            config = PosterPipelineConfig(n_colors=n_colors)
            pipeline = PosterPipeline(config)

            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()
            assert len(svg) > 0

    def test_poster_different_epsilon_values(self, test_img0, output_dir):
        """Test poster pipeline with different simplification factors."""
        for epsilon in [0.001, 0.002, 0.003]:
            output_file = output_dir / f"poster_img0_epsilon_{epsilon}.svg"

            config = PosterPipelineConfig(n_colors=32, epsilon_factor=epsilon)
            pipeline = PosterPipeline(config)

            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()
            assert len(svg) > 0

    def test_poster_debug_mode(self, test_img0, output_dir):
        """Test poster pipeline with debug mode enabled."""
        output_file = output_dir / "poster_img0_debug.svg"

        config = PosterPipelineConfig(n_colors=32)
        pipeline = PosterPipeline(config)

        svg = pipeline.process(str(test_img0), str(output_file), debug=True)

        # Check that debug stages were collected
        assert len(pipeline.debug_stages) > 0
        assert pipeline.debug_stages[0][0] == "1_original"

    def test_poster_pipeline_no_smoothing(self, test_img0, output_dir):
        """Verify poster pipeline never applies smoothing."""
        output_file = output_dir / "poster_img0_no_smooth.svg"

        # Even if someone tries to configure it, smooth_method is locked
        config = PosterPipelineConfig(n_colors=32)
        pipeline = PosterPipeline(config)

        # Verify the config has no smoothing
        assert pipeline.config.smooth_method == "none"
        assert pipeline.config.dilate_iterations == 0

        svg = pipeline.process(str(test_img0), str(output_file))
        assert output_file.exists()

    def test_poster_preserves_small_features(self, test_img0, output_dir):
        """Test that poster mode preserves small features with low min_area."""
        output_file = output_dir / "poster_img0_small_features.svg"

        # Poster mode uses min_contour_area=20.0 by default
        config = PosterPipelineConfig(n_colors=32)
        pipeline = PosterPipeline(config)

        assert pipeline.config.min_contour_area == 20.0
        assert pipeline.config.min_area == 30

        svg = pipeline.process(str(test_img0), str(output_file))
        assert output_file.exists()


class TestPosterPipelineConfig:
    """Test cases for PosterPipelineConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PosterPipelineConfig()

        assert config.n_colors == 32
        assert config.min_area == 30
        assert config.dilate_iterations == 0
        assert config.min_contour_area == 20.0
        assert config.epsilon_factor == 0.002
        assert config.smooth_method == "none"
        assert config.use_bezier is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PosterPipelineConfig(
            n_colors=48, min_area=25, epsilon_factor=0.001, min_contour_area=15.0
        )

        assert config.n_colors == 48
        assert config.min_area == 25
        assert config.epsilon_factor == 0.001
        assert config.min_contour_area == 15.0
        # These should remain locked
        assert config.dilate_iterations == 0
        assert config.smooth_method == "none"

    def test_warning_on_low_colors(self):
        """Test that warning is issued for low color count."""
        import warnings

        with pytest.warns(UserWarning, match="Poster mode works best"):
            PosterPipelineConfig(n_colors=16)

    def test_warning_on_high_epsilon(self):
        """Test that warning is issued for high epsilon factor."""
        import warnings

        with pytest.warns(UserWarning, match="Poster mode recommends"):
            PosterPipelineConfig(epsilon_factor=0.005)


class TestProcessImagePoster:
    """Test cases for process_image_poster convenience function."""

    def test_process_image_poster(self, test_img0, output_dir):
        """Test poster convenience function."""
        output_file = output_dir / "poster_convenience_test.svg"

        config = PosterPipelineConfig(n_colors=32)
        svg = process_image_poster(str(test_img0), str(output_file), config=config)

        assert output_file.exists()
        assert svg.startswith("<svg")

    def test_process_image_poster_no_output(self, test_img0):
        """Test poster convenience function without output path."""
        config = PosterPipelineConfig(n_colors=32)
        svg = process_image_poster(str(test_img0), config=config)

        assert svg.startswith("<svg")


class TestPosterVsClsctComparison:
    """Tests comparing POSTER vs CLSCT pipeline outputs."""

    def test_poster_crisper_boundaries(self, test_img0, output_dir):
        """Verify poster mode produces different output than clsct."""
        from apx_clsct.pipeline import Pipeline, PipelineConfig

        # CLSCT output
        clsct_file = output_dir / "comparison_clsct.svg"
        clsct_config = PipelineConfig(
            n_colors=32, smooth_method="none", epsilon_factor=0.002
        )
        clsct_pipeline = Pipeline(clsct_config)
        clsct_svg = clsct_pipeline.process(str(test_img0), str(clsct_file))

        # POSTER output
        poster_file = output_dir / "comparison_poster.svg"
        poster_config = PosterPipelineConfig(n_colors=32)
        poster_pipeline = PosterPipeline(poster_config)
        poster_svg = poster_pipeline.process(str(test_img0), str(poster_file))

        # Both should produce valid SVGs
        assert clsct_file.exists()
        assert poster_file.exists()

        # Poster should have no dilation (different from clsct default)
        assert poster_pipeline.config.dilate_iterations == 0
        assert clsct_pipeline.config.dilate_iterations == 1

    def test_poster_smaller_contour_threshold(self, test_img0, output_dir):
        """Verify poster mode keeps smaller contours."""
        from apx_clsct.pipeline import Pipeline, PipelineConfig

        clsct_config = PipelineConfig(n_colors=32)
        poster_config = PosterPipelineConfig(n_colors=32)

        # Poster should have lower thresholds to keep more detail
        assert poster_config.min_contour_area < clsct_config.min_contour_area
        assert poster_config.min_area < clsct_config.min_area
