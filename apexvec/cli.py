"""Command line interface for vectorizer."""
import argparse
import sys
from pathlib import Path

from apexvec.poster_pipeline import PosterPipeline
from apexvec.types import AdaptiveConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='apexvec',
        description='Convert raster images to poster-style SVG'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input image path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output SVG path (default: input.svg)'
    )
    
    parser.add_argument(
        '--colors',
        type=int,
        default=24,
        help='Number of colors (default: 24)'
    )
    
    parser.add_argument(
        '--save-stages',
        action='store_true',
        help='Save intermediate stage results for debugging'
    )
    
    parser.add_argument(
        '--stages-dir',
        type=str,
        default='./stages',
        help='Directory to save stage results (default: ./stages)'
    )
    
    return parser


def main(args=None):
    """Main entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    
    # Resolve input path
    input_path = Path(parsed_args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    # Determine output path
    if parsed_args.output:
        output_path = Path(parsed_args.output)
    else:
        output_path = input_path.with_suffix('.svg')
    
    # Create configuration
    config = AdaptiveConfig()
    config.transparent_background = True
    config.boundary_smoothing_passes = 2
    config.boundary_smoothing_strength = 0.5
    
    # Create pipeline and process
    try:
        print(f"Converting with {parsed_args.colors} colors...")
        
        pipeline = PosterPipeline(
            config=config,
            num_colors=parsed_args.colors,
            save_stages=parsed_args.save_stages,
            stages_dir=Path(parsed_args.stages_dir) if parsed_args.save_stages else None
        )
        svg_string = pipeline.process(input_path, output_path)
        
        print(f"\nSuccess! Output saved to: {output_path}")
        print(f"PNG preview: {output_path.with_suffix('.png')}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
