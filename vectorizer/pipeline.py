"""Unified vectorization pipeline."""
from pathlib import Path
from typing import Union, Optional
import time

from vectorizer.types import AdaptiveConfig, VectorizationError
from vectorizer.raster_ingest import ingest, IngestResult
from vectorizer.region_decomposer import decompose
from vectorizer.region_classifier import classify
from vectorizer.strategies.router import vectorize_all_regions
from vectorizer.topology_merger import merge_topology
from vectorizer.svg_optimizer import regions_to_svg, get_svg_size
from vectorizer.perceptual_loss import compute_ssim, mean_delta_e


class UnifiedPipeline:
    """Main pipeline for image vectorization."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Adaptive configuration. Uses defaults if None.
        """
        self.config = config or AdaptiveConfig()
    
    def process(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Process an image through the vectorization pipeline.
        
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
        
        # Step 1: Ingest image
        print("Step 1/6: Ingesting image...")
        ingest_result = ingest(input_path)
        
        # Step 2: Decompose into regions
        print("Step 2/6: Decomposing into regions...")
        regions = decompose(ingest_result.image_linear, self.config)
        print(f"  Found {len(regions)} regions")
        
        # Step 3: Classify regions
        print("Step 3/6: Classifying regions...")
        regions = classify(regions, ingest_result.image_linear, self.config)
        
        # Count by type
        from vectorizer.types import RegionKind
        type_counts = {}
        for r in regions:
            kind = getattr(r, 'kind', RegionKind.FLAT)
            type_counts[kind.name] = type_counts.get(kind.name, 0) + 1
        print(f"  Region types: {type_counts}")
        
        # Step 4: Vectorize regions
        print("Step 4/6: Vectorizing regions...")
        vector_regions = vectorize_all_regions(
            regions,
            ingest_result.image_linear,
            self.config,
            parallel=False  # Disable parallel for now
        )
        print(f"  Vectorized {len(vector_regions)} regions")
        
        # Step 5: Merge topology
        print("Step 5/6: Merging adjacent regions...")
        vector_regions = merge_topology(vector_regions, self.config.merge_threshold_delta_e)
        print(f"  After merging: {len(vector_regions)} regions")
        
        # Step 6: Generate SVG
        print("Step 6/6: Generating SVG...")
        svg_string = regions_to_svg(
            vector_regions,
            ingest_result.width,
            ingest_result.height,
            self.config.precision
        )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(svg_string, encoding='utf-8')
            print(f"  Saved to: {output_path}")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        
        return svg_string
    
    def validate(
        self,
        input_path: Union[str, Path],
        svg_string: str,
        ssim_threshold: float = 0.75,
        delta_e_threshold: float = 15.0
    ) -> dict:
        """
        Validate vectorization quality.
        
        Args:
            input_path: Path to original image
            svg_string: Generated SVG string
            ssim_threshold: Minimum SSIM score
            delta_e_threshold: Maximum Delta E
            
        Returns:
            Dictionary with validation results
        """
        # Load original
        ingest_result = ingest(input_path)
        original = ingest_result.image_srgb
        
        # Rasterize SVG for comparison
        from vectorizer.perceptual_loss import rasterize_svg
        rasterized = rasterize_svg(svg_string, ingest_result.width, ingest_result.height)
        
        # Compute metrics
        ssim_score = compute_ssim(original, rasterized)
        delta_e = mean_delta_e(original, rasterized)
        
        # Get SVG size
        svg_size = get_svg_size(svg_string)
        original_size = Path(input_path).stat().st_size
        
        results = {
            'ssim': ssim_score,
            'ssim_pass': ssim_score >= ssim_threshold,
            'delta_e': delta_e,
            'delta_e_pass': delta_e <= delta_e_threshold,
            'svg_size_bytes': svg_size,
            'original_size_bytes': original_size,
            'size_reduction': (1 - svg_size / original_size) * 100 if original_size > 0 else 0,
            'overall_pass': ssim_score >= ssim_threshold and delta_e <= delta_e_threshold
        }
        
        return results
