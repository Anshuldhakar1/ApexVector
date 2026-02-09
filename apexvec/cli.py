"""Command line interface for ApexVector."""
import argparse
import sys
from pathlib import Path

from apexvec.pipeline import UnifiedPipeline
from apexvec.poster_pipeline import PosterPipeline
from apexvec.contrast_pipeline import ContrastPipeline
from apexvec.types import AdaptiveConfig, ApexConfig


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
        '--save-stages',
        type=str,
        default=None,
        help='Directory to save pipeline stage debug images'
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['lossless', 'standard', 'compact', 'thumbnail'],
        default='standard',
        help='Optimization preset: lossless (archival), standard (balanced), compact (smaller), thumbnail (aggressive)'
    )

    parser.add_argument(
        '--poster',
        action='store_true',
        help='Poster-style mode - flat colors with smooth boundaries'
    )

    parser.add_argument(
        '--colors',
        type=int,
        default=12,
        help='Number of colors for poster mode (default: 12)'
    )

    parser.add_argument(
        '--contrast',
        action='store_true',
        help='Contrast-aware mode - multi-scale quantization for better detail preservation'
    )

    parser.add_argument(
        '--scales',
        type=str,
        default='6,10,18',
        help='Color scales for contrast mode: comma-separated list (default: 6,10,18)'
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

    # Handle save_stages if provided
    if parsed_args.save_stages:
        config.save_stages = Path(parsed_args.save_stages)
        config.save_stages.mkdir(parents=True, exist_ok=True)
        print(f"Debug stages will be saved to: {config.save_stages}")

    if parsed_args.contrast:
        # Contrast-aware mode with multi-scale quantization
        print("Mode: Contrast-aware (multi-scale quantization)")
        
        # Parse scales
        try:
            scales = [int(s.strip()) for s in parsed_args.scales.split(',')]
        except ValueError:
            print(f"Error: Invalid scales format: {parsed_args.scales}", file=sys.stderr)
            return 1
        
        print(f"  Scales: {scales}")
        
        try:
            pipeline = ContrastPipeline(scales=scales)
            svg_string = pipeline.process(input_path, output_path)
            
            # Validate if requested
            if parsed_args.validate:
                print("\nValidating output...")
                results = pipeline.validate(input_path, svg_string)
                
                print(f"\nValidation Results:")
                print(f"  Regions: {results['region_count']} {'✓' if results['region_count_pass'] else '✗'}")
                print(f"  SVG size: {results['svg_size_bytes']:,} bytes")
                print(f"  Size reduction: {results['size_reduction']:.1f}%")
                print(f"  Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
                
                return 0 if results['overall_pass'] else 1
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if parsed_args.poster:
        # Poster-style mode
        print("Mode: Poster-style (flat colors, smooth boundaries)")
        
        config = ApexConfig(n_colors=parsed_args.colors)
        
        try:
            pipeline = PosterPipeline(config)
            svg_string = pipeline.process(input_path, output_path)
            
            # Validate if requested
            if parsed_args.validate:
                print("\nValidating output...")
                results = pipeline.validate(input_path, svg_string)
                
                print(f"\nValidation Results:")
                print(f"  Regions: {results['region_count']} {'✓' if results['region_count_pass'] else '✗'}")
                print(f"  SVG size: {results['svg_size_bytes']:,} bytes")
                print(f"  Size reduction: {results['size_reduction']:.1f}%")
                print(f"  Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
                
                return 0 if results['overall_pass'] else 1
            
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Standard (unified) pipeline mode
    config = AdaptiveConfig()
    
    # Handle save_stages if provided
    if parsed_args.save_stages:
        config.save_stages = Path(parsed_args.save_stages)
        config.save_stages.mkdir(parents=True, exist_ok=True)
        print(f"Debug stages will be saved to: {config.save_stages}")

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
    
    # Create pipeline and process
    try:
        pipeline = UnifiedPipeline(config)
        
        # Map preset to optimize parameter
        optimize = parsed_args.preset if parsed_args.preset != 'lossless' else False
        svg_string = pipeline.process(input_path, output_path, optimize=optimize)
        
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
        return 1


if __name__ == '__main__':
    sys.exit(main())
