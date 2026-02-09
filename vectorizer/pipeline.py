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
from vectorizer.svg_optimizer import regions_to_svg, get_svg_size, generate_optimized_svg
from vectorizer.perceptual_loss import compute_ssim, mean_delta_e
from vectorizer import viz_utils


class UnifiedPipeline:
    """Main pipeline for image vectorization."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Adaptive configuration. Uses defaults if None.
        """
        self.config = config or AdaptiveConfig()
    
    def process(self, input_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, optimize: Union[bool, str] = True) -> str:
        """
        Process an image through the vectorization pipeline.
        
        Args:
            input_path: Path to input image
            output_path: Optional path for output SVG
            optimize: Whether to apply SVG compression optimizations
            
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

        # Save stage 1: ingest visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "01_ingest.png"
            viz_utils.visualize_ingest(ingest_result.image_srgb, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 1 to: {viz_path}")

        # Step 2: Decompose into regions
        print("Step 2/6: Decomposing into regions...")
        regions = decompose(ingest_result.image_linear, self.config)
        print(f"  Found {len(regions)} regions")

        # Save stage 2: region decomposition visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "02_regions.png"
            viz_utils.visualize_regions(ingest_result.image_linear, regions, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 2 to: {viz_path}")
        
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

        # Save stage 3: classification visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "03_classified.png"
            viz_utils.visualize_classification(ingest_result.image_linear, regions, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 3 to: {viz_path}")
        
        # Step 4: Vectorize regions
        print("Step 4/6: Vectorizing regions...")
        vector_regions = vectorize_all_regions(
            regions,
            ingest_result.image_linear,
            self.config,
            parallel=False  # Disable parallel for now
        )
        print(f"  Vectorized {len(vector_regions)} regions")

        # Save stage 4: vectorized regions visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "04_vectorized.png"
            viz_utils.visualize_vectorized(ingest_result.image_linear, vector_regions, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 4 to: {viz_path}")
        
        # Step 5: Merge topology
        print("Step 5/6: Merging adjacent regions...")
        vector_regions = merge_topology(vector_regions, self.config.merge_threshold_delta_e)
        print(f"  After merging: {len(vector_regions)} regions")

        # Save stage 5: merged topology visualization
        if self.config.save_stages:
            viz_path = self.config.save_stages / "05_merged.png"
            viz_utils.visualize_topology(ingest_result.image_linear, vector_regions, viz_path, self.config.stage_dpi)
            print(f"  Saved stage 5 to: {viz_path}")
        
        # Step 6: Generate SVG
        print("Step 6/6: Generating SVG...")
        
        # Map old optimize modes to new presets
        preset_map = {
            'ultra': 'compact',
            'symbol': 'standard', 
            'merged': 'standard',
            'extreme': 'thumbnail',
            'insane': 'thumbnail',  # Redirect to thumbnail with warning
            'monochrome': None,     # Disabled - causes unacceptable quality loss
            True: 'standard',       # Default when optimize=True
            False: 'lossless'       # No optimization
        }
        
        # Handle disabled modes
        effective_optimize = optimize
        if optimize == 'monochrome':
            print("  Warning: 'monochrome' mode disabled. Use external tools for monochrome conversion.")
            effective_optimize = 'standard'
        
        preset = preset_map.get(effective_optimize, 'standard')
        
        if effective_optimize is False:
            # No optimization - use pretty output
            svg_string = regions_to_svg(
                vector_regions,
                ingest_result.width,
                ingest_result.height,
                self.config.precision,
                compact=False
            )
        else:
            # Use new preset-based optimization
            svg_string = generate_optimized_svg(
                vector_regions,
                ingest_result.width,
                ingest_result.height,
                preset_name=preset,
                base_precision=self.config.precision
            )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(svg_string, encoding='utf-8')
            print(f"  Saved to: {output_path}")

        # Save stage 6: final SVG and comparison visualization
        if self.config.save_stages:
            # Save the SVG
            svg_path = self.config.save_stages / "06_final.svg"
            svg_path.write_text(svg_string, encoding='utf-8')
            print(f"  Saved stage 6 SVG to: {svg_path}")

            # Save comparison visualization
            viz_path = self.config.save_stages / "06_rasterized.png"
            viz_utils.visualize_final(
                ingest_result.image_linear,
                svg_string,
                viz_path,
                ingest_result.width,
                ingest_result.height,
                self.config.stage_dpi
            )
            print(f"  Saved stage 6 comparison to: {viz_path}")

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
