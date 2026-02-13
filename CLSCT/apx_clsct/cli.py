"""Command-line interface for apx-clsct."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import Pipeline, PipelineConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="apx-clsct",
        description="Color Layer Separation + Contour Tracing Vectorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic vectorization with 8 colors
  apx-clsct -i input.jpg -o output.svg

  # Vectorization with 12 colors
  apx-clsct -i input.jpg -o output.svg --colors 12

  # With debug output
  apx-clsct -i input.jpg -o output.svg --debug

  # Custom smoothing
  apx-clsct -i input.jpg -o output.svg --smooth gaussian --sigma 1.5
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Input image file path")

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output SVG file path (default: input name with .svg extension)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Save intermediate stage visualizations"
    )

    parser.add_argument(
        "--colors",
        type=int,
        default=8,
        help="Number of colors for quantization (default: 8)",
    )

    parser.add_argument(
        "--smooth",
        choices=["none", "gaussian", "bspline"],
        default="gaussian",
        help="Smoothing method (default: gaussian)",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma (default: 1.0)",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Douglas-Peucker epsilon factor (default: 0.01)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum contour area in pixels (default: 50)",
    )

    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=50.0,
        help="Minimum contour area after detection (default: 50.0)",
    )

    return parser


def save_debug_stages(pipeline, output_path: str) -> None:
    """Save debug stage images.

    Args:
        pipeline: Pipeline instance with debug_stages
        output_path: Base output path for debug images
    """
    from PIL import Image

    base_path = Path(output_path)
    debug_dir = base_path.parent / f"{base_path.stem}_debug"
    debug_dir.mkdir(exist_ok=True)

    for stage_name, stage_image in pipeline.debug_stages:
        debug_file = debug_dir / f"{stage_name}.png"

        # Ensure image is uint8
        if stage_image.dtype != np.uint8:
            if stage_image.max() <= 1.0:
                stage_image = (stage_image * 255).astype(np.uint8)
            else:
                stage_image = stage_image.astype(np.uint8)

        Image.fromarray(stage_image).save(debug_file)
        print(f"  Saved debug stage: {debug_file}")


def main(args: Optional[list] = None) -> int:
    """Main entry point for CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Determine output path
    input_path = Path(parsed.input)
    if parsed.output:
        output_path = parsed.output
    else:
        output_path = str(input_path.with_suffix(".svg"))

    # Create configuration
    config = PipelineConfig(
        n_colors=parsed.colors,
        smooth_method=parsed.smooth,
        smooth_sigma=parsed.sigma,
        epsilon_factor=parsed.epsilon,
        min_area=parsed.min_area,
        min_contour_area=getattr(parsed, "min_contour_area", 50.0),
    )

    # Process image
    try:
        print(f"Processing: {parsed.input}")
        print(f"  Colors: {parsed.colors}")
        print(f"  Smoothing: {parsed.smooth}")

        pipeline = Pipeline(config)
        svg = pipeline.process(parsed.input, output_path, debug=parsed.debug)

        print(f"  Output saved: {output_path}")

        # Save debug stages if requested
        if parsed.debug and hasattr(pipeline, "debug_stages"):
            import numpy as np

            save_debug_stages(pipeline, output_path)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
