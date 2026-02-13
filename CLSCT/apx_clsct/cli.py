"""Command-line interface for apx-clsct."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .pipeline import Pipeline, PipelineConfig, PosterPipeline, PosterPipelineConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="apx-clsct",
        description="Color Layer Separation + Contour Tracing Vectorization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLSCT Mode (Default - General Purpose)
  apx-clsct -i input.jpg -o output.svg
  apx-clsct -i input.jpg -o output.svg --colors 32 --smooth gaussian

  # POSTER Mode (Sharp, Geometric Vector Art)
  apx-clsct -i input.jpg -o output.svg --mode poster
  apx-clsct -i input.jpg -o output.svg --mode poster --colors 48 --epsilon 0.001

  # Debug output
  apx-clsct -i input.jpg -o output.svg --mode poster --debug
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
        "-m",
        "--mode",
        choices=["clsct", "poster"],
        default="clsct",
        help="Pipeline mode: clsct (default, general-purpose) or poster (sharp geometric art)",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Save intermediate stage visualizations"
    )

    parser.add_argument(
        "--colors",
        "-c",
        type=int,
        default=None,  # Will be set based on mode
        help="Number of colors for quantization (clsct default: 24, poster default: 32)",
    )

    parser.add_argument(
        "--smooth",
        choices=["none", "gaussian", "bspline"],
        default="none",
        help="Smoothing method - NOT AVAILABLE in poster mode (default: none)",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian smoothing sigma (default: 1.0)",
    )

    parser.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=None,  # Will be set based on mode
        help="Simplification epsilon factor (clsct default: 0.005, poster default: 0.002)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=None,  # Will be set based on mode
        help="Minimum region area in pixels (clsct default: 50, poster default: 30)",
    )

    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=None,  # Will be set based on mode
        help="Minimum contour area in pixels (clsct default: 50.0, poster default: 20.0)",
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

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate mode-specific constraints
    if parsed.mode == "poster":
        if parsed.smooth != "none":
            print(
                f"Warning: Smoothing method '{parsed.smooth}' is not available in poster mode. "
                "Setting to 'none'.",
                file=sys.stderr,
            )

    # Set defaults based on mode
    if parsed.mode == "clsct":
        n_colors = parsed.colors if parsed.colors is not None else 24
        epsilon_factor = parsed.epsilon if parsed.epsilon is not None else 0.005
        min_area = parsed.min_area if parsed.min_area is not None else 50
        min_contour_area = (
            parsed.min_contour_area if parsed.min_contour_area is not None else 50.0
        )
        smooth_method = parsed.smooth
    else:  # poster mode
        n_colors = parsed.colors if parsed.colors is not None else 32
        epsilon_factor = parsed.epsilon if parsed.epsilon is not None else 0.002
        min_area = parsed.min_area if parsed.min_area is not None else 30
        min_contour_area = (
            parsed.min_contour_area if parsed.min_contour_area is not None else 20.0
        )
        smooth_method = "none"  # Force no smoothing in poster mode

    # Create configuration and pipeline based on mode
    try:
        print(f"Processing: {parsed.input}")
        print(f"  Mode: {parsed.mode}")
        print(f"  Colors: {n_colors}")
        print(f"  Epsilon: {epsilon_factor}")

        if parsed.mode == "clsct":
            config = PipelineConfig(
                n_colors=n_colors,
                smooth_method=smooth_method,
                smooth_sigma=parsed.sigma,
                epsilon_factor=epsilon_factor,
                min_area=min_area,
                min_contour_area=min_contour_area,
            )
            pipeline = Pipeline(config)
            print(f"  Smoothing: {smooth_method}")
        else:  # poster mode
            config = PosterPipelineConfig(
                n_colors=n_colors,
                epsilon_factor=epsilon_factor,
                min_area=min_area,
                min_contour_area=min_contour_area,
            )
            pipeline = PosterPipeline(config)
            print(f"  Smoothing: none (forced for poster mode)")
            print(f"  Dilation: 0 (forced for poster mode)")

        svg = pipeline.process(parsed.input, output_path, debug=parsed.debug)

        print(f"  Output saved: {output_path}")

        # Save debug stages if requested
        if parsed.debug and hasattr(pipeline, "debug_stages"):
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
