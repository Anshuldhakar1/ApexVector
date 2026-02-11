"""Fixed poster-style vectorization pipeline with save stages."""
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.color import rgb2lab, lab2rgb

from apexvec.types import Region, VectorRegion, RegionKind, AdaptiveConfig
from apexvec.raster_ingest import ingest
from apexvec.svg_optimizer import regions_to_svg
from apexvec.boundary_smoother import (
    smooth_boundary_infallible,
    extract_contours_subpixel
)
from apexvec.svg_to_png import svg_to_png
from apexvec.quantization import slic_quantize
from apexvec.region_merger import merge_small_regions_same_color

logger = logging.getLogger(__name__)


class PosterPipeline:
    """Poster-style vectorization with flat colors and smooth boundaries."""
    
    def __init__(
        self,
        config: Optional[AdaptiveConfig] = None,
        num_colors: int = 12,
        save_stages: bool = False,
        stages_dir: Optional[Path] = None
    ):
        """
        Initialize poster pipeline.
        
        Args:
            config: Configuration (uses defaults if None)
            num_colors: Number of colors for quantization (default: 12)
            save_stages: Whether to save intermediate stage results
            stages_dir: Directory to save stage results (default: ./stages)
        """
        self.config = config or AdaptiveConfig()
        self.num_colors = num_colors
        self.save_stages = save_stages
        self.stages_dir = stages_dir or Path('./stages')
        
        if self.save_stages:
            self.stages_dir.mkdir(parents=True, exist_ok=True)
    
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Process an image through the poster vectorization pipeline.
        
        Args:
            input_path: Path to input image
            output_path: Optional path for output SVG
            
        Returns:
            SVG string
        """
        start_time = time.time()
        input_path = Path(input_path)
        
        # Step 1: Ingest image
        print("Step 1/6: Ingesting image...")
        ingest_result = ingest(input_path)
        print(f"  Image: {ingest_result.width}x{ingest_result.height}")
        
        if self.save_stages:
            self._save_stage_image(
                ingest_result.image_srgb,
                "stage_01_original.png"
            )
        
        # Step 2: Color quantization
        print(f"Step 2/6: Quantizing to {self.num_colors} colors...")
        label_map, palette = quantize_colors(
            ingest_result.image_srgb,
            num_colors=self.num_colors
        )
        print(f"  Quantized to {len(palette)} colors")
        
        if self.save_stages:
            quantized_img = palette[label_map]
            self._save_stage_image(quantized_img, "stage_02_quantized.png")
        
        # Step 3: Extract regions from quantized image
        print("Step 3/6: Extracting regions...")
        regions = extract_regions_from_quantized(
            label_map, palette, 
            min_region_size=self.config.min_region_size
        )
        print(f"  Found {len(regions)} regions")
        
        # Merge small isolated regions to reduce artifacts (same-color only)
        if len(regions) > 0:
            regions = merge_small_regions_same_color(
                regions,
                min_area=self.config.min_region_size * 2
            )
            print(f"  After merging: {len(regions)} regions")
        
        if self.save_stages:
            self._save_region_mask(regions, ingest_result.image_srgb.shape, "stage_03_regions.png")
        
        # Step 4: Vectorize regions with boundary smoothing
        print("Step 4/6: Vectorizing regions with smooth boundaries...")
        
        # Adaptive smoothing: reduce passes for large images
        total_pixels = ingest_result.width * ingest_result.height
        if total_pixels > 1000000:  # >1MP
            smoothing_passes = 1
            print(f"  Large image detected ({total_pixels:,} px), reducing smoothing to {smoothing_passes} pass")
        else:
            smoothing_passes = self.config.boundary_smoothing_passes
        
        # Use parallel processing for moderate batches only
        # Large batches cause memory issues during serialization
        use_parallel = 50 < len(regions) < 1000
        if use_parallel:
            print(f"  Using parallel processing for {len(regions)} regions...")
        elif len(regions) >= 1000:
            print(f"  Using sequential processing for {len(regions)} regions (too many for parallel)")
        
        vector_regions = vectorize_poster_regions(
            regions,
            smoothness_factor=self.config.boundary_smoothing_strength,
            smoothing_passes=smoothing_passes,
            parallel=use_parallel
        )
        print(f"  Vectorized {len(vector_regions)} regions")
        
        if len(vector_regions) == 0:
            raise RuntimeError("No regions were successfully vectorized!")
        
        if self.save_stages:
            self._save_vector_regions_preview(vector_regions, ingest_result.image_srgb.shape, "stage_04_vectorized.png")
            self._save_region_statistics(vector_regions, "stage_04_stats.txt")
        
        # Validate color fidelity: all regions must have fill_color from palette
        print("  Validating color fidelity...")
        for i, vr in enumerate(vector_regions):
            if vr.fill_color is None:
                logger.warning(f"Region {i} has no fill_color, using default")
                vr.fill_color = palette[0] if len(palette) > 0 else np.array([128, 128, 128])
        
        # Step 5: Generate SVG with transparent background
        print("Step 5/6: Generating SVG...")
        svg_string = regions_to_svg(
            vector_regions,
            ingest_result.width,
            ingest_result.height,
            self.config.precision,
            config=self.config
        )
        
        # Save SVG
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(svg_string, encoding='utf-8')
            print(f"  Saved SVG to: {output_path}")
        
        if self.save_stages:
            # Save intermediate SVG
            intermediate_svg = self.stages_dir / "stage_05_svg.svg"
            intermediate_svg.write_text(svg_string, encoding='utf-8')
            print(f"  Saved stage: {intermediate_svg}")
        
        # Step 6: Convert to PNG
        print("Step 6/6: Converting SVG to PNG...")
        if output_path:
            png_path = output_path.with_suffix('.png')
        else:
            png_path = input_path.with_suffix('.png')
        
        png_result = svg_to_png(
            output_path or svg_string,
            png_path,
            width=ingest_result.width,
            height=ingest_result.height
        )
        
        if png_result:
            print(f"  Saved PNG to: {png_result}")
        else:
            print("  Warning: PNG conversion failed")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        
        if self.save_stages:
            self._save_timing_report(start_time, {
                "Image size": f"{ingest_result.width}x{ingest_result.height}",
                "Colors": self.num_colors,
                "Regions found": len(regions),
                "Regions vectorized": len(vector_regions)
            }, "stage_06_timing.txt")
        
        return svg_string
    
    def _save_stage_image(self, image: np.ndarray, filename: str):
        """Save an intermediate stage image."""
        try:
            from PIL import Image
            
            # Ensure uint8 format
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            output_path = self.stages_dir / filename
            Image.fromarray(image).save(output_path)
            print(f"  Saved stage: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save stage {filename}: {e}")
    
    def _save_region_mask(self, regions: List[Region], shape: tuple, filename: str):
        """Save region mask visualization."""
        try:
            from PIL import Image
            
            mask_img = np.zeros((*shape[:2], 3), dtype=np.uint8)
            
            for region in regions:
                if region.mean_color is not None:
                    color = region.mean_color
                    if color.max() <= 1.0:
                        color = (color * 255).astype(np.uint8)
                    mask_img[region.mask] = color
            
            output_path = self.stages_dir / filename
            Image.fromarray(mask_img).save(output_path)
            print(f"  Saved stage: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save stage {filename}: {e}")
    
    def _save_vector_regions_preview(self, vector_regions: List[VectorRegion], shape: tuple, filename: str):
        """Save a rasterized preview of vectorized regions."""
        try:
            from PIL import Image, ImageDraw
            
            # Create blank image
            h, w = shape[:2]
            img = Image.new('RGB', (w, h), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw each vector region
            for vr in vector_regions:
                if not vr.path:
                    continue
                
                # Get fill color
                color = vr.fill_color
                if color is not None:
                    if color.max() <= 1.0:
                        color = tuple((color * 255).astype(np.uint8))
                    else:
                        color = tuple(color.astype(np.uint8))
                else:
                    color = (128, 128, 128)
                
                # Convert bezier curves to polygon points
                points = []
                for curve in vr.path:
                    if not points:
                        points.append((curve.p0.x, curve.p0.y))
                    points.append((curve.p3.x, curve.p3.y))
                
                if len(points) >= 3:
                    draw.polygon(points, fill=color)
            
            output_path = self.stages_dir / filename
            img.save(output_path)
            print(f"  Saved stage: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save stage {filename}: {e}")
    
    def _save_region_statistics(self, vector_regions: List[VectorRegion], filename: str):
        """Save region statistics to a text file."""
        try:
            output_path = self.stages_dir / filename
            
            # Calculate statistics
            total_regions = len(vector_regions)
            regions_with_holes = sum(1 for vr in vector_regions if vr.hole_paths)
            
            # Size distribution
            sizes = []
            for vr in vector_regions:
                if vr.path:
                    # Calculate area from path bounds
                    xs = [c.p0.x for c in vr.path] + [c.p3.x for c in vr.path]
                    ys = [c.p0.y for c in vr.path] + [c.p3.y for c in vr.path]
                    if xs and ys:
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        sizes.append(area)
            
            stats = [
                "Region Statistics",
                "=================",
                f"Total regions: {total_regions}",
                f"Regions with holes: {regions_with_holes}",
                f"",
                "Size Distribution:",
                f"  Min: {min(sizes) if sizes else 0:.1f} pixels",
                f"  Max: {max(sizes) if sizes else 0:.1f} pixels",
                f"  Mean: {sum(sizes)/len(sizes) if sizes else 0:.1f} pixels",
                f"  Median: {sorted(sizes)[len(sizes)//2] if sizes else 0:.1f} pixels",
                f"",
                "Top 10 Largest Regions:",
            ]
            
            # Sort by size and show top 10
            region_info = []
            for i, vr in enumerate(vector_regions):
                if vr.path:
                    xs = [c.p0.x for c in vr.path] + [c.p3.x for c in vr.path]
                    ys = [c.p0.y for c in vr.path] + [c.p3.y for c in vr.path]
                    if xs and ys:
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        num_holes = len(vr.hole_paths)
                        color = vr.fill_color
                        if color is not None:
                            color_str = f"RGB({int(color[0])}, {int(color[1])}, {int(color[2])})"
                        else:
                            color_str = "Unknown"
                        region_info.append((i, area, num_holes, color_str))
            
            region_info.sort(key=lambda x: x[1], reverse=True)
            for i, (idx, area, holes, color) in enumerate(region_info[:10]):
                stats.append(f"  {i+1}. Region {idx}: {area:.1f} px, {holes} holes, {color}")
            
            output_path.write_text('\n'.join(stats), encoding='utf-8')
            print(f"  Saved stats: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save stats {filename}: {e}")
    
    def _save_timing_report(self, start_time: float, metadata: dict, filename: str):
        """Save timing and metadata report."""
        try:
            import time
            output_path = self.stages_dir / filename
            
            elapsed = time.time() - start_time
            
            report = [
                "Pipeline Timing Report",
                "=====================",
                f"Total time: {elapsed:.2f} seconds",
                f"",
                "Metadata:",
            ]
            
            for key, value in metadata.items():
                report.append(f"  {key}: {value}")
            
            output_path.write_text('\n'.join(report), encoding='utf-8')
            print(f"  Saved timing: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save timing {filename}: {e}")


def quantize_colors(
    image: np.ndarray,
    num_colors: int = 12,
    compactness: float = 10.0,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors using SLIC superpixels for spatial coherence.
    
    SLIC provides better spatial coherence than K-means by combining
    color similarity with spatial proximity, reducing fragmentation.
    
    Args:
        image: Input image (H, W, 3) uint8
        num_colors: Target number of superpixels/colors
        compactness: Balance between color and spatial proximity
                     (higher = more spatially regular)
        sigma: Gaussian smoothing sigma before segmentation
        
    Returns:
        Tuple of (label_map, palette):
        - label_map: (H, W) array of palette indices
        - palette: (N, 3) uint8 palette in sRGB
    """
    return slic_quantize(
        image,
        n_segments=num_colors,
        compactness=compactness,
        sigma=sigma,
        random_state=42
    )


def extract_regions_from_quantized(
    label_map: np.ndarray,
    palette: np.ndarray,
    min_region_size: int = 100
) -> List[Region]:
    """Extract connected regions from quantized label map with size filtering."""
    from scipy.ndimage import label
    
    regions = []
    h, w = label_map.shape
    
    # Process each color
    for color_idx in range(len(palette)):
        color_mask = (label_map == color_idx)
        
        if not np.any(color_mask):
            continue
        
        # Find connected components
        labeled_array, num_features = label(color_mask)
        
        # Create region for each component
        for feature_idx in range(1, num_features + 1):
            region_mask = (labeled_array == feature_idx)
            
            region_size = np.sum(region_mask)
            
            # Skip small regions (reduces artifacts)
            if region_size < min_region_size:
                continue
            
            region = Region(
                mask=region_mask,
                label=len(regions),
                mean_color=palette[color_idx]
            )
            regions.append(region)
    
    return regions


def merge_small_regions(
    regions: List[Region],
    label_map: np.ndarray,
    merge_threshold_size: int = 200
) -> List[Region]:
    """
    Merge small regions with their largest neighbor.
    
    This reduces artifacts by eliminating tiny isolated regions
    and merging them into adjacent larger regions.
    
    Args:
        regions: List of regions
        label_map: Original label map
        merge_threshold_size: Regions smaller than this get merged
        
    Returns:
        Filtered and merged list of regions
    """
    if not regions:
        return regions
    
    # Build spatial index for finding neighbors
    h, w = label_map.shape
    region_ids = np.full((h, w), -1, dtype=np.int32)
    
    for i, region in enumerate(regions):
        region_ids[region.mask] = i
    
    # Find small regions to merge
    merged = set()
    new_regions = []
    
    for i, region in enumerate(regions):
        if i in merged:
            continue
        
        region_size = np.sum(region.mask)
        
        if region_size >= merge_threshold_size:
            new_regions.append(region)
            continue
        
        # Find neighboring regions
        # Dilate mask slightly to find neighbors
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(region.mask, iterations=1)
        neighbor_mask = dilated & ~region.mask
        
        neighbor_ids = np.unique(region_ids[neighbor_mask])
        neighbor_ids = neighbor_ids[neighbor_ids >= 0]
        
        if len(neighbor_ids) == 0:
            # Isolated small region - keep it
            new_regions.append(region)
            continue
        
        # Find largest neighbor
        largest_neighbor = max(
            neighbor_ids,
            key=lambda nid: np.sum(regions[nid].mask) if nid not in merged else 0
        )
        
        # Merge into largest neighbor
        if largest_neighbor not in merged:
            regions[largest_neighbor].mask = regions[largest_neighbor].mask | region.mask
            merged.add(i)
    
    return new_regions


def _vectorize_single_region(
    region_data: Tuple[int, Region, float, int]
) -> Optional[VectorRegion]:
    """Vectorize a single region - helper for parallel processing."""
    idx, region, smoothness_factor, smoothing_passes = region_data
    
    try:
        # Extract contours (both outer and holes)
        outer_contours, hole_contours = extract_contours_subpixel(region.mask)
        
        if not outer_contours:
            return None
        
        # Use largest outer contour
        longest_outer = max(outer_contours, key=len)
        
        if len(longest_outer) < 3:
            return None
        
        # Apply Chaikin smoothing to outer contour
        smoothed_points = longest_outer.copy()
        for _ in range(min(smoothing_passes, 2)):
            smoothed_points = chaikin_smooth(smoothed_points, iterations=1)
        
        # Smooth outer boundary with infallible smoother
        outer_bezier = smooth_boundary_infallible(
            smoothed_points,
            smoothness_factor=smoothness_factor,
            min_points=3
        )
        
        if not outer_bezier or len(outer_bezier) < 2:
            return None
        
        # Process hole contours
        hole_paths = []
        for hole in hole_contours:
            if len(hole) < 3:
                continue
            
            # Smooth hole contour (less smoothing for holes)
            smoothed_hole = hole.copy()
            for _ in range(min(smoothing_passes, 1)):  # Only 1 pass for holes
                smoothed_hole = chaikin_smooth(smoothed_hole, iterations=1)
            
            # Smooth hole boundary
            hole_bezier = smooth_boundary_infallible(
                smoothed_hole,
                smoothness_factor=smoothness_factor * 0.8,
                min_points=3
            )
            
            if hole_bezier and len(hole_bezier) >= 2:
                hole_paths.append(hole_bezier)
        
        # Create vector region with holes
        return VectorRegion(
            kind=RegionKind.FLAT,
            path=outer_bezier,
            hole_paths=hole_paths,
            fill_color=region.mean_color
        )
        
    except Exception as e:
        logger.error(f"Region {idx}: Error during vectorization: {e}")
        return None


def vectorize_poster_regions(
    regions: List[Region],
    smoothness_factor: float = 0.6,
    smoothing_passes: int = 3,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> List[VectorRegion]:
    """Vectorize poster regions using infallible boundary smoothing, including holes.
    
    Args:
        regions: List of regions to vectorize
        smoothness_factor: Smoothing strength for boundaries
        smoothing_passes: Number of smoothing iterations
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers (None = auto)
    """
    import os
    
    # Prepare region data for processing
    region_data = [
        (idx, region, smoothness_factor, smoothing_passes)
        for idx, region in enumerate(regions)
    ]
    
    vector_regions = []
    
    if parallel and len(regions) > 10:
        # Use parallel processing for many regions
        workers = max_workers or min(os.cpu_count() or 4, 8)
        logger.info(f"Vectorizing {len(regions)} regions using {workers} workers...")
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_vectorize_single_region, region_data))
        
        vector_regions = [vr for vr in results if vr is not None]
    else:
        # Sequential processing for small batches
        for data in region_data:
            result = _vectorize_single_region(data)
            if result is not None:
                vector_regions.append(result)
    
    logger.info(f"Poster vectorization: {len(regions)} input -> {len(vector_regions)} output")
    
    return vector_regions


def chaikin_smooth(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply Chaikin corner-cutting algorithm for smooth curves."""
    if len(points) < 3:
        return points
    
    result = points.copy()
    
    for _ in range(iterations):
        if len(result) < 3:
            break
        
        new_points = []
        n = len(result)
        
        for i in range(n):
            p0 = result[i]
            p1 = result[(i + 1) % n]
            
            # Chaikin corner-cutting
            q = 0.25 * p0 + 0.75 * p1
            r = 0.75 * p0 + 0.25 * p1
            
            new_points.append(r)
            new_points.append(q)
        
        result = np.array(new_points)
    
    return result
