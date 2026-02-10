"""Command line interface for vectorizer."""
import argparse
import sys
from pathlib import Path

from apexvec.pipeline import UnifiedPipeline
from apexvec.poster_pipeline import PosterPipeline
from apexvec.types import AdaptiveConfig


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog='apexvec',
        description='Convert raster images to optimized SVG'
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
        '--speed',
        action='store_true',
        help='Fast mode - lower quality but faster'
    )
    
    parser.add_argument(
        '--quality',
        action='store_true',
        help='Quality mode - higher quality but slower'
    )
    
    parser.add_argument(
        '--segments',
        type=int,
        default=400,
        help='Number of SLIC segments (default: 400)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output quality'
    )
    
    parser.add_argument(
        '--debug-regions',
        action='store_true',
        help='Enable region debugging visualizations'
    )
    
    parser.add_argument(
        '--debug-output',
        type=str,
        default='debug_output',
        help='Directory for debug output (default: debug_output)'
    )
    
    # Poster pipeline options
    parser.add_argument(
        '--poster',
        action='store_true',
        help='Use poster-style pipeline with color quantization'
    )
    
    parser.add_argument(
        '--colors',
        type=int,
        default=16,
        help='Number of colors for poster mode (default: 16)'
    )
    
    parser.add_argument(
        '--save-stages',
        action='store_true',
        help='Save intermediate stage images'
    )
    
    parser.add_argument(
        '--stages-dir',
        type=str,
        default='./stages',
        help='Directory for stage images (default: ./stages)'
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
    
    if parsed_args.speed:
        # Fast mode: fewer segments, looser tolerances
        config.slic_segments = 100
        config.max_bezier_error = 5.0
        config.merge_threshold_delta_e = 10.0
        print("Mode: Speed (fast, lower quality)")
    elif parsed_args.quality:
        # Quality mode: more segments, tighter tolerances
        config.slic_segments = 800
        config.max_bezier_error = 1.0
        config.merge_threshold_delta_e = 3.0
        print("Mode: Quality (slow, higher quality)")
    else:
        # Default mode
        config.slic_segments = parsed_args.segments
        print("Mode: Balanced")
    
    # Configure debug options
    config.debug_regions = parsed_args.debug_regions
    config.debug_output_dir = parsed_args.debug_output
    
    # Create pipeline and process
    try:
        if parsed_args.poster:
            # Use poster pipeline
            print(f"Mode: Poster (color quantization with {parsed_args.colors} colors)")
            config.transparent_background = True
            config.boundary_smoothing_passes = 2
            config.boundary_smoothing_strength = 0.5
            
            pipeline = PosterPipeline(
                config=config,
                num_colors=parsed_args.colors,
                save_stages=parsed_args.save_stages,
                stages_dir=Path(parsed_args.stages_dir)
            )
            svg_string = pipeline.process(input_path, output_path)
            print(f"\nSuccess! Output saved to: {output_path}")
            print(f"PNG preview: {output_path.with_suffix('.png')}")
        else:
            # Use standard pipeline
            pipeline = UnifiedPipeline(config)
            svg_string = pipeline.process(input_path, output_path)
            
            # Validate if requested
            if parsed_args.validate:
                print("\nValidating output...")
                results = pipeline.validate(input_path, svg_string)
                
                print(f"\nValidation Results:")
                print(f"  SSIM: {results['ssim']:.3f} {'✓' if results['ssim_pass'] else '✗'}")
                print(f"  Delta E: {results['delta_e']:.2f} {'✓' if results['delta_e_pass'] else '✗'}")
                print(f"  SVG size: {results['svg_size_bytes']:,} bytes")
                print(f"  Size reduction: {results['size_reduction']:.1f}%")
                print(f"  Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
                
                return 0 if results['overall_pass'] else 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
