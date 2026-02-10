"""Poster-style vectorization pipeline with color quantization."""
from pathlib import Path
from typing import Union, Optional, List, Tuple
import time
import logging

import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb

from vectorizer.types import Region, VectorRegion, RegionKind, AdaptiveConfig
from vectorizer.raster_ingest import ingest
from vectorizer.svg_optimizer import regions_to_svg
from vectorizer.boundary_smoother import (
    smooth_boundary_infallible,
    extract_contours_subpixel,
    simplify_bezier_curves
)

logger = logging.getLogger(__name__)


class PosterPipeline:
    """Poster-style vectorization with flat colors and smooth boundaries."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None, num_colors: int = 12):
        """
        Initialize poster pipeline.
        
        Args:
            config: Configuration (uses defaults if None)
            num_colors: Number of colors for quantization (default: 12)
        """
        self.config = config or AdaptiveConfig()
        self.num_colors = num_colors
    
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
        
        # Step 1: Ingest image
        print("Step 1/5: Ingesting image...")
        ingest_result = ingest(input_path)
        
        # Step 2: Color quantization
        print(f"Step 2/5: Quantizing to {self.num_colors} colors...")
        label_map, palette = quantize_colors(
            ingest_result.image_srgb,
            num_colors=self.num_colors
        )
        print(f"  Quantized to {len(palette)} colors")
        
        # Step 3: Extract regions from quantized image
        print("Step 3/5: Extracting regions...")
        regions = extract_regions_from_quantized(label_map, palette)
        print(f"  Found {len(regions)} regions")
        
        # Step 4: Vectorize regions with new boundary smoothing
        print("Step 4/5: Vectorizing regions with smooth boundaries...")
        vector_regions = vectorize_poster_regions(
            regions,
            smoothness_factor=self.config.boundary_smoothing_strength,
            smoothing_passes=self.config.boundary_smoothing_passes
        )
        print(f"  Vectorized {len(vector_regions)} regions")
        
        # Step 5: Generate SVG with transparent background
        print("Step 5/5: Generating SVG...")
        svg_string = regions_to_svg(
            vector_regions,
            ingest_result.width,
            ingest_result.height,
            self.config.precision,
            config=self.config
        )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(svg_string, encoding='utf-8')
            print(f"  Saved to: {output_path}")
        
        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.2f}s")
        
        return svg_string


def quantize_colors(
    image: np.ndarray,
    num_colors: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image colors using K-means in LAB space.
    
    Args:
        image: Input image (H, W, 3) in sRGB [0-255] or [0-1]
        num_colors: Number of colors to quantize to
        
    Returns:
        Tuple of (label_map, palette)
        - label_map: (H, W) array of color indices
        - palette: (num_colors, 3) array of RGB colors in uint8
    """
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
    """
    Extract connected regions from quantized label map.
    
    Args:
        label_map: (H, W) array of color indices
        palette: (num_colors, 3) array of RGB colors
        
    Returns:
        List of Region objects
    """
    from scipy.ndimage import label
    
    regions = []
    h, w = label_map.shape
    
    # Process each color
    for color_idx in range(len(palette)):
        # Create mask for this color
        color_mask = (label_map == color_idx)
        
        if not np.any(color_mask):
            continue
        
        # Find connected components for this color
        labeled_array, num_features = label(color_mask)
        
        # Create a region for each connected component
        for feature_idx in range(1, num_features + 1):
            region_mask = (labeled_array == feature_idx)
            
            # Skip very small regions (noise)
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
    """
    Vectorize poster regions using infallible boundary smoothing.
    
    Args:
        regions: List of regions
        smoothness_factor: Boundary smoothing amount
        smoothing_passes: Number of smoothing passes
        
    Returns:
        List of VectorRegion objects
    """
    vector_regions = []
    
    for idx, region in enumerate(regions):
        # Extract contours (outer and holes)
        outer_contours, hole_contours = extract_contours_subpixel(region.mask)
        
        if not outer_contours:
            logger.warning(f"Region {idx}: No outer contours found")
            continue
        
        # Use the largest outer contour
        longest_outer = max(outer_contours, key=len)
        
        # Apply multiple passes of smoothing for better results
        smoothed_points = longest_outer.copy()
        for _ in range(smoothing_passes):
            smoothed_points = chaikin_smooth(smoothed_points, iterations=2)
        
        # Smooth the boundary with infallible smoother
        bezier_curves = smooth_boundary_infallible(
            smoothed_points,
            smoothness_factor=smoothness_factor,
            min_points=3
        )
        
        if not bezier_curves:
            logger.warning(f"Region {idx}: Failed to create curves")
            continue
        
        # Simplify bezier curves to reduce file size
        simplified_curves = simplify_bezier_curves(bezier_curves, tolerance=1.5)
        
        logger.debug(
            f"Region {idx}: {len(bezier_curves)} curves -> {len(simplified_curves)} simplified curves"
        )
        
        # Create vector region
        vector_region = VectorRegion(
            kind=RegionKind.FLAT,
            path=simplified_curves,
            fill_color=region.mean_color
        )
        
        vector_regions.append(vector_region)
    
    logger.info(f"Poster vectorization: {len(regions)} regions -> {len(vector_regions)} vector regions")
    
    return vector_regions


def chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """
    Apply Chaikin corner-cutting algorithm for smooth curves.
    
    Args:
        points: Input points (N, 2)
        iterations: Number of iterations
        
    Returns:
        Smoothed points
    """
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
            
            # Chaikin corner-cutting (creates C1 continuous curve)
            q = 0.25 * p0 + 0.75 * p1
            r = 0.75 * p0 + 0.25 * p1
            
            new_points.append(r)
            new_points.append(q)
        
        result = np.array(new_points)
    
    return result
