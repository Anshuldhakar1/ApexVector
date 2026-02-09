"""Contrast-aware vectorization pipeline with multi-scale quantization."""
from pathlib import Path
from typing import Union, Optional, List
import time

from apexvec.types import VectorizationError
from apexvec.raster_ingest import ingest
from apexvec.multi_scale_quantizer import multi_scale_quantize
from apexvec.hierarchical_merger import hierarchical_merge, convert_to_standard_regions
from apexvec.boundary_smoother import smooth_region_boundaries
from apexvec.svg_export import generate_svg, save_svg


class ContrastPipeline:
    """
    Contrast-aware vectorization pipeline.
    
    Pipeline stages:
    1. EXIF fix and linear RGB conversion
    2. Multi-scale color quantization (K-means at 3 levels)
    3. Hierarchical region merging (coarse → fine)
    4. Boundary smoothing with splines
    5. SVG export
    """
    
    def __init__(self, scales: List[int] = None):
        """
        Initialize pipeline with quantization scales.
        
        Args:
            scales: List of color counts for each scale (default: [6, 10, 18])
        """
        self.scales = scales or [6, 10, 18]
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_stages: bool = False
    ) -> str:
        """
        Process an image through the contrast-aware vectorization pipeline.
        
        Args:
            input_path: Path to input image
            output_path: Optional path for output SVG
            save_stages: Whether to save intermediate stage images
            
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
        print("Stage 1/5: Ingesting image...")
        ingest_result = ingest(input_path)
        print(f"  Loaded {ingest_result.width}x{ingest_result.height} image")
        
        # Stage 2: Multi-scale quantization
        print(f"Stage 2/5: Multi-scale quantization...")
        print(f"  Scales: {self.scales}")
        label_maps = multi_scale_quantize(
            ingest_result.image_linear,
            scales=self.scales
        )
        print(f"  Generated {len(label_maps)} quantization levels")
        
        # Stage 3: Hierarchical region merging
        print("Stage 3/5: Hierarchical region merging...")
        hierarchical_regions = hierarchical_merge(
            label_maps,
            ingest_result.image_linear,
            area_threshold=0.02,
            contrast_threshold=10.0
        )
        
        # Convert to standard regions
        regions = convert_to_standard_regions(hierarchical_regions)
        print(f"  Extracted {len(regions)} regions")
        
        # Stage 4: Boundary smoothing
        print("Stage 4/5: Smoothing boundaries...")
        from apexvec.types import ApexConfig
        config = ApexConfig()
        paths = smooth_region_boundaries(regions, config)
        print(f"  Smoothed {len(paths)} boundaries")
        
        # Stage 5: Generate SVG
        print("Stage 5/5: Generating SVG...")
        svg_string = generate_svg(
            paths,
            regions,
            ingest_result.width,
            ingest_result.height,
            config
        )
        
        # Stage 6: Save output
        if output_path:
            output_path = Path(output_path)
            save_svg(svg_string, str(output_path))
            print(f"  Saved to: {output_path}")
        
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
        max_regions: int = 100
    ) -> dict:
        """
        Validate contrast-aware vectorization.
        
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
