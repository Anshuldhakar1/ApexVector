#!/usr/bin/env python
"""Run poster pipeline with save stages."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vectorizer.poster_pipeline import PosterPipeline
from vectorizer.types import AdaptiveConfig


def main():
    parser = argparse.ArgumentParser(
        prog='poster-vectorizer',
        description='Convert raster images to poster-style SVG with save stages'
    )
    
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('-o', '--output', type=str, help='Output SVG path')
    parser.add_argument('--colors', type=int, default=16, help='Number of colors (default: 16)')
    parser.add_argument('--save-stages', action='store_true', help='Save intermediate stages')
    parser.add_argument('--stages-dir', type=str, default='./stages', help='Directory for stages')
    parser.add_argument('--transparent', action='store_true', default=True, help='Transparent background')
    parser.add_argument('--no-transparent', dest='transparent', action='store_false', help='White background')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.svg')
    
    # Create config
    config = AdaptiveConfig()
    config.transparent_background = args.transparent
    config.boundary_smoothing_passes = 2
    config.boundary_smoothing_strength = 0.5
    
    # Create pipeline
    pipeline = PosterPipeline(
        config=config,
        num_colors=args.colors,
        save_stages=args.save_stages,
        stages_dir=Path(args.stages_dir)
    )
    
    try:
        svg = pipeline.process(input_path, output_path)
        print(f"\nSuccess! Output saved to: {output_path}")
        print(f"PNG preview: {output_path.with_suffix('.png')}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
