"""Poster-style vectorization pipeline."""
from pathlib import Path
from typing import Union, Optional
import time

from apexvec.types import ApexConfig, VectorizationError
from apexvec.raster_ingest import ingest
from apexvec.color_quantizer import quantize_colors
from apexvec.region_extractor import extract_regions
from apexvec.boundary_smoother import smooth_region_boundaries
from apexvec.svg_export import generate_svg, save_svg
from apexvec import viz_utils


class PosterPipeline:
    """
    Poster-style vectorization pipeline.
    
    Pipeline stages:
    1. EXIF fix and linear RGB conversion
    2. Color quantization (K-means, 8-16 colors)
    3. Connected components extraction
    4. Small region merging
    5. Spline boundary smoothing
    6. SVG export
    """
    
    def __init__(self, config: Optional[ApexConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Apex configuration. Uses defaults if None.
        """
        self.config = config or ApexConfig()
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Process an image through the poster-style vectorization pipeline.

        Args:
            input_path: Path to input image
            output_path: Optional path for output SVG

        Returns:
            SVG string

        Raises:
            FileNotFoundError: If input file doesn't exist
            VectorizationError: If processing fails
        """
        start_time = time.time()
        input_path = Path(input_path)

        print(f"Processing: {input_path}")

        # Stage 1: Ingest image
        print("Stage 1/6: Ingesting image...")
        ingest_result = ingest(input_path)
        print(f"  Loaded {ingest_result.width}x{ingest_result.height} image")

        # Save stage 1: ingest visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "01_ingest.png"
            viz_utils.visualize_ingest(ingest_result.image_srgb, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 1 to: {viz_path}")

        # Stage 2: Color quantization
        print(f"Stage 2/6: Quantizing to {self.config.n_colors} colors...")
        label_map, palette = quantize_colors(
            ingest_result.image_linear,
            n_colors=self.config.n_colors
        )
        print(f"  Quantized to {len(palette)} colors")

        # Save stage 2: quantization visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "02_quantization.png"
            viz_utils.visualize_poster_quantization(
                ingest_result.image_linear,
                label_map,
                palette,
                viz_path,
                self.config.stage_dpi
            )
            print(f"  Saved stage 2 to: {viz_path}")

        # Stage 3: Region extraction
        print("Stage 3/6: Extracting regions...")
        regions = extract_regions(
            label_map,
            palette,
            ingest_result.image_linear,
            self.config
        )
        print(f"  Extracted {len(regions)} regions")

        if len(regions) > self.config.max_regions:
            print(f"  Warning: {len(regions)} regions exceeds max ({self.config.max_regions})")

        # Save stage 3: regions visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "03_regions.png"
            viz_utils.visualize_poster_regions(
                ingest_result.image_linear,
                regions,
                viz_path,
                self.config.stage_dpi
            )
            print(f"  Saved stage 3 to: {viz_path}")

        # Stage 4: Boundary smoothing
        print("Stage 4/6: Smoothing boundaries...")
        paths = smooth_region_boundaries(regions, self.config)
        print(f"  Smoothed {len(paths)} boundaries")

        # Save stage 4: boundaries visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "04_boundaries.png"
            viz_utils.visualize_poster_boundaries(
                ingest_result.image_linear,
                paths,
                viz_path,
                self.config.stage_dpi
            )
            print(f"  Saved stage 4 to: {viz_path}")

        # Stage 5: Generate SVG
        print("Stage 5/6: Generating SVG...")
        svg_string = generate_svg(
            paths,
            regions,
            ingest_result.width,
            ingest_result.height,
            self.config
        )

        # Stage 6: Save output
        print("Stage 6/6: Saving output...")
        if output_path:
            output_path = Path(output_path)
            save_svg(svg_string, str(output_path))
            print(f"  Saved to: {output_path}")

        # Save stage 6: final visualization with comparison
        if self.config.save_stages:
            # Save the SVG
            svg_path = self.config.save_stages / "05_final.svg"
            save_svg(svg_string, str(svg_path))
            print(f"  Saved stage 5 SVG to: {svg_path}")

            # Save comparison visualization
            elapsed = time.time() - start_time
            svg_size = len(svg_string.encode('utf-8'))
            original_size = input_path.stat().st_size

            viz_path = self.config.save_stages / "06_comparison.png"
            viz_utils.visualize_final(
                ingest_result.image_linear,
                svg_string,
                viz_path,
                ingest_result.width,
                ingest_result.height,
                self.config.stage_dpi,
                metrics={
                    'regions': len(regions),
                    'svg_size': svg_size,
                    'original_size': original_size,
                    'time': elapsed
                }
            )
            print(f"  Saved stage 6 comparison to: {viz_path}")

        # Report statistics
        svg_size = len(svg_string.encode('utf-8'))
        original_size = input_path.stat().st_size
        reduction = (1 - svg_size / original_size) * 100 if original_size > 0 else 0

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Regions: {len(regions)}")
        print(f"Original size: {original_size:,} bytes")
        print(f"SVG size: {svg_size:,} bytes")
        print(f"Size reduction: {reduction:.1f}%")

        return svg_string
    
    def validate(
        self,
        input_path: Union[str, Path],
        svg_string: str,
        max_regions: int = 20
    ) -> dict:
        """
        Validate poster-style vectorization.
        
        Args:
            input_path: Path to original image
            svg_string: Generated SVG string
            max_regions: Maximum allowed regions
            
        Returns:
            Dictionary with validation results
        """
        # Parse SVG to count regions
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(svg_string)
            # Count path elements
            path_count = len(root.findall('.//{http://www.w3.org/2000/svg}path'))
        except Exception:
            path_count = svg_string.count('<path')
        
        # Check file size
        svg_size = len(svg_string.encode('utf-8'))
        original_size = Path(input_path).stat().st_size
        
        results = {
            'region_count': path_count,
            'region_count_pass': path_count <= max_regions,
            'svg_size_bytes': svg_size,
            'original_size_bytes': original_size,
            'size_reduction': (1 - svg_size / original_size) * 100 if original_size > 0 else 0,
            'overall_pass': path_count <= max_regions
        }
        
        return results
