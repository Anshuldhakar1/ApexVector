"""Real image integration tests with organized output folders."""

import pytest
from pathlib import Path
from xml.etree import ElementTree as ET

from apx_clsct.pipeline import Pipeline, PipelineConfig


class TestRealImages:
    """Test pipeline with actual images from ../test_images/."""

    def test_img0_vectorization(self, test_img0, output_dir):
        """Test vectorization of img0.jpg with multiple color counts."""
        img0_dir = output_dir / "img0"
        img0_dir.mkdir(exist_ok=True)

        results = []
        for n_colors in [6, 8, 12]:
            output_file = img0_dir / f"img0_{n_colors}colors.svg"

            config = PipelineConfig(n_colors=n_colors)
            pipeline = Pipeline(config)
            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists(), (
                f"Output file not created for {n_colors} colors"
            )
            assert svg.startswith("<svg"), f"Invalid SVG for {n_colors} colors"

            # Validate XML structure
            root = ET.fromstring(svg)
            assert root.tag.endswith("svg")

            # Count paths
            paths = root.findall(".//{http://www.w3.org/2000/svg}path")
            results.append(
                {"colors": n_colors, "paths": len(paths), "file": output_file}
            )

        # More colors should generally create more paths
        print(f"\nimg0.jpg results:")
        for r in results:
            print(f"  {r['colors']} colors -> {r['paths']} paths")

    def test_img1_vectorization(self, test_img1, output_dir):
        """Test vectorization of img1.jpg."""
        img1_dir = output_dir / "img1"
        img1_dir.mkdir(exist_ok=True)

        output_file = img1_dir / "img1_vectorized.svg"

        config = PipelineConfig(n_colors=8, smooth_method="gaussian")
        pipeline = Pipeline(config)
        svg = pipeline.process(str(test_img1), str(output_file), debug=True)

        assert output_file.exists()
        assert svg.startswith("<svg")

        # Validate structure
        root = ET.fromstring(svg)
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")
        assert len(paths) > 0, "No paths generated for img1.jpg"

        # Check debug stages were collected
        assert len(pipeline.debug_stages) >= 3, "Debug stages not collected"

        print(f"\nimg1.jpg: Generated {len(paths)} paths")

    def test_multiple_real_images(self, output_dir):
        """Test pipeline with multiple real images from test_images."""
        from pathlib import Path

        TEST_IMAGES_DIR = Path(__file__).parent.parent.parent / "test_images"

        # Test with 2 different images as specified in plan.md
        test_images = ["img2.jpg", "img3.jpg"]

        results = []
        for img_name in test_images:
            img_path = TEST_IMAGES_DIR / img_name
            if not img_path.exists():
                pytest.skip(f"Test image {img_name} not found")

            # Create separate folder for each image
            img_folder = output_dir / Path(img_name).stem
            img_folder.mkdir(exist_ok=True)

            output_file = img_folder / f"{Path(img_name).stem}_output.svg"

            config = PipelineConfig(n_colors=8, epsilon_factor=0.005)
            pipeline = Pipeline(config)

            try:
                svg = pipeline.process(str(img_path), str(output_file))

                # Validate output
                assert output_file.exists()
                root = ET.fromstring(svg)
                paths = root.findall(".//{http://www.w3.org/2000/svg}path")

                results.append(
                    {
                        "image": img_name,
                        "folder": img_folder,
                        "paths": len(paths),
                        "success": True,
                    }
                )
            except Exception as e:
                results.append({"image": img_name, "error": str(e), "success": False})

        # All tests should succeed
        failures = [r for r in results if not r["success"]]
        assert len(failures) == 0, f"Failed to process: {failures}"

        print(f"\nMultiple images test results:")
        for r in results:
            print(f"  {r['image']}: {r['paths']} paths -> {r['folder']}")

    def test_smooth_methods_on_real_image(self, test_img0, output_dir):
        """Test different smoothing methods on real image."""
        smooth_dir = output_dir / "smooth_comparison"
        smooth_dir.mkdir(exist_ok=True)

        methods = ["none", "gaussian", "bspline"]
        results = []

        for method in methods:
            output_file = smooth_dir / f"img0_smooth_{method}.svg"

            config = PipelineConfig(
                n_colors=8,
                smooth_method=method,
                smooth_sigma=1.5 if method == "gaussian" else 1.0,
                smoothness=3.0 if method == "bspline" else 3.0,
            )
            pipeline = Pipeline(config)
            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()

            # Get path count
            root = ET.fromstring(svg)
            paths = root.findall(".//{http://www.w3.org/2000/svg}path")

            results.append({"method": method, "paths": len(paths), "file": output_file})

        print(f"\nSmoothing methods comparison:")
        for r in results:
            print(f"  {r['method']}: {r['paths']} paths")

    def test_different_color_counts_real_image(self, test_img0, output_dir):
        """Test various color counts on real image."""
        colors_dir = output_dir / "color_comparison"
        colors_dir.mkdir(exist_ok=True)

        color_counts = [4, 6, 8, 10, 12]
        results = []

        for n_colors in color_counts:
            output_file = colors_dir / f"img0_{n_colors}colors.svg"

            config = PipelineConfig(n_colors=n_colors)
            pipeline = Pipeline(config)
            svg = pipeline.process(str(test_img0), str(output_file))

            assert output_file.exists()

            root = ET.fromstring(svg)
            paths = root.findall(".//{http://www.w3.org/2000/svg}path")

            results.append(
                {
                    "colors": n_colors,
                    "paths": len(paths),
                    "size": len(svg),
                    "file": output_file,
                }
            )

        print(f"\nColor count comparison:")
        for r in results:
            print(f"  {r['colors']} colors: {r['paths']} paths, {r['size']} bytes")

    def test_debug_mode_outputs(self, test_img0, output_dir):
        """Test that debug mode saves intermediate stages."""
        debug_dir = output_dir / "debug_test"
        debug_dir.mkdir(exist_ok=True)

        output_file = debug_dir / "debug_output.svg"

        config = PipelineConfig(n_colors=8)
        pipeline = Pipeline(config)
        svg = pipeline.process(str(test_img0), str(output_file), debug=True)

        # Check debug stages
        assert len(pipeline.debug_stages) >= 3, "Expected at least 3 debug stages"

        expected_stages = ["1_original", "2_quantized", "3_contours"]
        stage_names = [name for name, _ in pipeline.debug_stages]

        for expected in expected_stages:
            assert any(expected in name for name in stage_names), (
                f"Missing debug stage: {expected}"
            )

        print(f"\nDebug stages: {stage_names}")
