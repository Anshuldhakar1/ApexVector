"""Fixed poster-style vectorization pipeline with save stages."""
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time
import logging

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

from apexvec.types import Region, VectorRegion, RegionKind, AdaptiveConfig
from apexvec.raster_ingest import ingest
from apexvec.svg_optimizer import regions_to_svg
from apexvec.boundary_smoother import (
    smooth_boundary_infallible,
    extract_contours_subpixel
)
from apexvec.svg_to_png import svg_to_png

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
        regions = extract_regions_from_quantized(label_map, palette)
        print(f"  Found {len(regions)} regions")
        
        if self.save_stages:
            self._save_region_mask(regions, ingest_result.image_srgb.shape, "stage_03_regions.png")
        
        # Step 4: Vectorize regions with boundary smoothing
        print("Step 4/6: Vectorizing regions with smooth boundaries...")
        vector_regions = vectorize_poster_regions(
            regions,
            smoothness_factor=self.config.boundary_smoothing_strength,
            smoothing_passes=self.config.boundary_smoothing_passes
        )
        print(f"  Vectorized {len(vector_regions)} regions")
        
        if len(vector_regions) == 0:
            raise RuntimeError("No regions were successfully vectorized!")
        
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


def quantize_colors(
    image: np.ndarray,
    num_colors: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize image colors using K-means in LAB space."""
    # Ensure image is in [0, 1] range for skimage
    if image.max() > 1.0:
        image_float = image / 255.0
    else:
        image_float = image
    
    # Convert to LAB color space
    image_lab = rgb2lab(image_float)
    
    # Reshape for K-means
    h, w = image_lab.shape[:2]
    pixels_lab = image_lab.reshape(-1, 3)
    
    # Run K-means clustering
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_lab)
    
    # Reshape labels back to image shape
    label_map = labels.reshape(h, w)
    
    # Convert cluster centers back to RGB
    centers_lab = kmeans.cluster_centers_
    palette_lab = centers_lab.reshape(num_colors, 1, 3)
    palette_rgb = lab2rgb(palette_lab).reshape(num_colors, 3)
    
    # Convert to uint8
    palette = (np.clip(palette_rgb, 0, 1) * 255).astype(np.uint8)
    
    return label_map, palette


def extract_regions_from_quantized(
    label_map: np.ndarray,
    palette: np.ndarray
) -> List[Region]:
    """Extract connected regions from quantized label map."""
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
            
            # Skip very small regions
            if np.sum(region_mask) < 10:
                continue
            
            region = Region(
                mask=region_mask,
                label=len(regions),
                mean_color=palette[color_idx]
            )
            regions.append(region)
    
    return regions


def vectorize_poster_regions(
    regions: List[Region],
    smoothness_factor: float = 0.6,
    smoothing_passes: int = 3
) -> List[VectorRegion]:
    """Vectorize poster regions using infallible boundary smoothing."""
    vector_regions = []
    
    for idx, region in enumerate(regions):
        try:
            # Extract contours
            outer_contours, hole_contours = extract_contours_subpixel(region.mask)
            
            if not outer_contours:
                logger.warning(f"Region {idx}: No outer contours found")
                continue
            
            # Use largest contour
            longest_outer = max(outer_contours, key=len)
            
            if len(longest_outer) < 3:
                logger.warning(f"Region {idx}: Contour too short ({len(longest_outer)} points)")
                continue
            
            # Apply Chaikin smoothing (less aggressive than before)
            smoothed_points = longest_outer.copy()
            for _ in range(min(smoothing_passes, 2)):  # Max 2 passes to avoid over-smoothing
                smoothed_points = chaikin_smooth(smoothed_points, iterations=1)
            
            # Smooth with infallible smoother
            bezier_curves = smooth_boundary_infallible(
                smoothed_points,
                smoothness_factor=smoothness_factor,
                min_points=3
            )
            
            if not bezier_curves:
                logger.warning(f"Region {idx}: Failed to create bezier curves")
                continue
            
            if len(bezier_curves) < 2:
                logger.warning(f"Region {idx}: Too few curves ({len(bezier_curves)})")
                continue
            
            # Create vector region
            vector_region = VectorRegion(
                kind=RegionKind.FLAT,
                path=bezier_curves,
                fill_color=region.mean_color
            )
            
            vector_regions.append(vector_region)
            
        except Exception as e:
            logger.error(f"Region {idx}: Error during vectorization: {e}")
            continue
    
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
