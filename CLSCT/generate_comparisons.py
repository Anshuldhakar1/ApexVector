"""Generate visual comparison outputs for both pipeline modes."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apx_clsct.pipeline import (
    Pipeline,
    PipelineConfig,
    PosterPipeline,
    PosterPipelineConfig,
)
from tests.utils.svg_to_png import svg_to_png


def generate_comparison_outputs():
    """Generate side-by-side comparison of CLSCT vs POSTER modes."""

    test_images_dir = Path("../test_images")
    output_base = Path("output")

    test_images = [
        ("img0.jpg", "img0"),
        ("img1.jpg", "img1"),
    ]

    for img_file, img_name in test_images:
        img_path = test_images_dir / img_file

        if not img_path.exists():
            print(f"Skipping {img_file} - not found")
            continue

        # Create output directories
        clsct_dir = output_base / img_name / "clsct"
        poster_dir = output_base / img_name / "poster"
        clsct_dir.mkdir(parents=True, exist_ok=True)
        poster_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing: {img_file}")
        print(f"{'=' * 60}")

        # CLSCT Mode
        print("\n[CLSCT Mode]")
        clsct_svg_path = clsct_dir / "output.svg"
        clsct_config = PipelineConfig(
            n_colors=24,
            smooth_method="none",
            epsilon_factor=0.005,
            min_area=50,
            min_contour_area=50.0,
            dilate_iterations=1,
        )
        clsct_pipeline = Pipeline(clsct_config)
        clsct_svg = clsct_pipeline.process(str(img_path), str(clsct_svg_path))
        print(f"  SVG saved: {clsct_svg_path}")

        # Convert to PNG
        clsct_png_path = clsct_dir / "output.png"
        try:
            svg_to_png(str(clsct_svg_path), str(clsct_png_path))
            print(f"  PNG saved: {clsct_png_path}")
        except Exception as e:
            print(f"  PNG conversion failed: {e}")

        # POSTER Mode
        print("\n[POSTER Mode]")
        poster_svg_path = poster_dir / "output.svg"
        poster_config = PosterPipelineConfig(
            n_colors=32, epsilon_factor=0.002, min_area=30, min_contour_area=20.0
        )
        poster_pipeline = PosterPipeline(poster_config)
        poster_svg = poster_pipeline.process(str(img_path), str(poster_svg_path))
        print(f"  SVG saved: {poster_svg_path}")

        # Convert to PNG
        poster_png_path = poster_dir / "output.png"
        try:
            svg_to_png(str(poster_svg_path), str(poster_png_path))
            print(f"  PNG saved: {poster_png_path}")
        except Exception as e:
            print(f"  PNG conversion failed: {e}")

        # File size comparison
        clsct_size = clsct_svg_path.stat().st_size / 1024
        poster_size = poster_svg_path.stat().st_size / 1024

        print(f"\n[File Size Comparison]")
        print(f"  CLSCT SVG: {clsct_size:.1f} KB")
        print(f"  POSTER SVG: {poster_size:.1f} KB")
        print(f"  Difference: {poster_size - clsct_size:+.1f} KB")

    print(f"\n{'=' * 60}")
    print("Visual comparison complete!")
    print(f"Output folders:")
    for _, img_name in test_images:
        print(f"  - output/{img_name}/clsct/")
        print(f"  - output/{img_name}/poster/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    generate_comparison_outputs()
