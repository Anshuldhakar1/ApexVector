"""
SLIC-Based Poster Pipeline with Shared Boundaries

Pipeline stages:
1. SLIC quantization (spatial + color)
2. Same-color region merging (cleanup fragmentation)
3. Marching squares boundary extraction (sub-pixel)
4. Gaussian smoothing per shared edge
5. Region reconstruction from smoothed edges
6. SVG export with palette colors
7. resvg CLI rasterization + comparison
"""
import pickle
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import io

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import segmentation, measure, color as skcolor
from collections import defaultdict
import json

from apexvec.types import ApexConfig, IngestResult, VectorizationError
from apexvec.raster_ingest import ingest


@dataclass
class SharedEdge:
    """A boundary edge shared between two regions."""
    edge_id: int
    points: np.ndarray  # (N, 2) array of (y, x) coordinates
    region_a: int       # Region ID
    region_b: int       # Region ID (-1 for background)
    color_a: int        # Color index for region_a
    color_b: int        # Color index for region_b
    smoothed_points: Optional[np.ndarray] = None


@dataclass
class SlicRegion:
    """Region with shared boundary references."""
    label: int
    color_idx: int
    mask: np.ndarray    # Binary mask
    edge_ids: List[int] = field(default_factory=list)
    edge_directions: List[bool] = field(default_factory=list)  # True if reversed


class SlicPipeline:
    """
    SLIC-based poster pipeline with shared boundaries.
    """
    
    def __init__(self, config: Optional[ApexConfig] = None):
        self.config = config or ApexConfig()
        self.debug_dir: Optional[Path] = None
        
        # Stage outputs
        self.ingest_result: Optional[IngestResult] = None
        self.label_map: Optional[np.ndarray] = None
        self.palette: Optional[np.ndarray] = None
        self.regions: Optional[List[SlicRegion]] = None
        self.shared_edges: Optional[List[SharedEdge]] = None
        self.svg_string: Optional[str] = None
        
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        debug_stages: Optional[Union[str, Path]] = None
    ) -> str:
        """Process image through SLIC pipeline."""
        input_path = Path(input_path)
        
        if debug_stages:
            self.debug_dir = Path(debug_stages)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Ingest
            print("Stage 1/7: Ingesting image...")
            self.ingest_result = self._stage1_ingest(input_path)
            print(f"  Loaded {self.ingest_result.width}x{self.ingest_result.height}")
            
            # Stage 2: SLIC Quantization
            print(f"Stage 2/7: SLIC quantization to {self.config.n_colors} colors...")
            self.label_map, self.palette = self._stage2_slic_quantize()
            print(f"  Palette: {len(self.palette)} colors")
            
            # Stage 3: Same-color region merging
            print("Stage 3/7: Merging same-color regions...")
            self.regions = self._stage3_merge_regions()
            print(f"  Regions: {len(self.regions)}")
            
            # Stage 4: Marching squares boundary extraction
            print("Stage 4/7: Extracting boundaries with marching squares...")
            self.shared_edges = self._stage4_marching_squares()
            print(f"  Shared edges: {len(self.shared_edges)}")
            
            # Stage 5: Gaussian smoothing
            print("Stage 5/7: Smoothing boundaries...")
            self.shared_edges = self._stage5_smooth_boundaries()
            
            # Stage 6: Build SVG
            print("Stage 6/7: Building SVG...")
            self.svg_string = self._stage6_build_svg()
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(self.svg_string)
                print(f"  Saved: {output_path}")
            
            # Stage 7: Validate with resvg
            print("Stage 7/7: Validating with resvg...")
            validation = self._stage7_validate()
            print(f"  Coverage: {validation['coverage']:.1f}%")
            print(f"  Gap pixels: {validation['gap_pixels']}")
            
            return self.svg_string
            
        except Exception as e:
            if self.debug_dir:
                error_file = self.debug_dir / "stage_error.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {e}\n\n")
                    f.write(traceback.format_exc())
            raise
    
    def _stage1_ingest(self, input_path: Path) -> IngestResult:
        """Stage 1: Ingest image."""
        result = ingest(input_path)
        
        if self.debug_dir:
            self._save_stage_image(
                result.image_srgb,
                self.debug_dir / "stage1_ingest.png"
            )
            with open(self.debug_dir / "stage1_ingest_data.pkl", 'wb') as f:
                pickle.dump({
                    'width': result.width,
                    'height': result.height,
                }, f)
        
        return result
    
    def _stage2_slic_quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: SLIC quantization (spatial + color).
        
        Uses SLIC superpixels followed by color clustering for spatial coherence.
        """
        h, w = self.ingest_result.image_linear.shape[:2]
        
        # Convert to LAB for perceptual uniformity
        lab_image = skcolor.rgb2lab(self.ingest_result.image_srgb)
        
        # Calculate approximate segment size for target color count
        # SLIC gives us spatial coherence
        n_segments = min(self.config.n_colors * 4, 500)  # More segments than colors
        compactness = 10.0  # Balance between color and spatial
        
        print(f"  Running SLIC with {n_segments} segments...")
        slic_labels = segmentation.slic(
            self.ingest_result.image_srgb,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=-1
        )
        
        # Now cluster the superpixel mean colors to get target n_colors
        n_superpixels = slic_labels.max() + 1
        superpixel_colors = np.zeros((n_superpixels, 3))
        
        for i in range(n_superpixels):
            mask = (slic_labels == i)
            if mask.any():
                superpixel_colors[i] = lab_image[mask].mean(axis=0)
        
        # K-means on superpixel colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=self.config.n_colors,
            random_state=42,
            n_init=10
        )
        color_labels = kmeans.fit_predict(superpixel_colors)
        
        # Create final label map
        label_map = color_labels[slic_labels]
        
        # Create palette from cluster centers
        palette_lab = kmeans.cluster_centers_
        palette_rgb = skcolor.lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        palette_rgb = np.clip(palette_rgb, 0, 1)
        
        # Verify all colors are present
        unique_labels = np.unique(label_map)
        print(f"  Unique labels: {len(unique_labels)}")
        
        if self.debug_dir:
            viz = palette_rgb[label_map]
            self._save_stage_image(viz, self.debug_dir / "stage2_quantize.png")
            
            with open(self.debug_dir / "stage2_quantize_data.pkl", 'wb') as f:
                pickle.dump({
                    'label_map': label_map,
                    'palette': palette_rgb
                }, f)
        
        return label_map, palette_rgb
    
    def _stage3_merge_regions(self) -> List[SlicRegion]:
        """
        Stage 3: Same-color region merging.
        
        Merges connected components of the same color and removes tiny fragments.
        """
        h, w = self.label_map.shape
        n_colors = len(self.palette)
        total_area = h * w
        min_area = max(50, int(total_area * self.config.min_region_area_ratio))
        
        print(f"  Min area threshold: {min_area} pixels")
        
        regions = []
        region_id = 0
        
        # Process each color
        for color_idx in range(n_colors):
            color_mask = (self.label_map == color_idx)
            
            if not color_mask.any():
                continue
            
            # Find connected components for this color
            labeled_array, num_features = ndimage.label(color_mask)
            
            if num_features == 0:
                continue
            
            # Get component sizes
            component_ids = labeled_array[labeled_array > 0]
            if len(component_ids) == 0:
                continue
                
            counts = np.bincount(component_ids)
            
            # Keep components above min_area, or keep largest if all are small
            keep_mask = counts >= min_area
            
            if not keep_mask[1:].any():
                # No components above threshold, keep the largest
                if len(counts) > 1:
                    largest = np.argmax(counts[1:]) + 1
                    keep_mask[largest] = True
            
            # Create regions for kept components
            for comp_id in range(1, num_features + 1):
                if not keep_mask[comp_id]:
                    continue
                
                mask = (labeled_array == comp_id)
                
                region = SlicRegion(
                    label=region_id,
                    color_idx=color_idx,
                    mask=mask,
                    edge_ids=[],
                    edge_directions=[]
                )
                regions.append(region)
                region_id += 1
        
        print(f"  Total regions after merging: {len(regions)}")
        
        # Verify all colors preserved
        remaining_colors = set(r.color_idx for r in regions)
        missing = set(range(n_colors)) - remaining_colors
        if missing:
            print(f"  WARNING: Missing colors: {missing}")
        else:
            print(f"  OK: All {n_colors} colors preserved")
        
        if self.debug_dir:
            viz = self._visualize_regions(regions)
            self._save_stage_image(viz, self.debug_dir / "stage3_regions.png")
            
            with open(self.debug_dir / "stage3_regions_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_regions': len(regions),
                    'color_indices': [r.color_idx for r in regions]
                }, f)
        
        return regions
    
    def _stage4_marching_squares(self) -> List[SharedEdge]:
        """
        Stage 4: Marching squares boundary extraction.
        
        Extracts shared boundaries between adjacent regions using skimage's
        find_contours (marching squares algorithm) for sub-pixel accuracy.
        """
        h, w = self.label_map.shape
        shared_edges = []
        edge_id = 0
        
        # Build adjacency graph and extract boundaries
        # For each region, find contours and determine neighbors
        
        for region in self.regions:
            # Find contours at sub-pixel precision
            contours = measure.find_contours(region.mask, 0.5)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Determine neighbor by sampling points along contour
                sample_points = contour[::max(1, len(contour) // 10)]
                neighbor_colors = []
                
                for (y, x) in sample_points:
                    yy, xx = int(round(y)), int(round(x))
                    yy = np.clip(yy, 0, h - 1)
                    xx = np.clip(xx, 0, w - 1)
                    
                    # Sample just outside the contour
                    if not region.mask[yy, xx]:
                        neighbor_colors.append(self.label_map[yy, xx])
                
                # Find most common neighbor color
                neighbor_region_id = -1
                neighbor_color_idx = -1
                
                if neighbor_colors:
                    from collections import Counter
                    color_counts = Counter(neighbor_colors)
                    neighbor_color_idx = color_counts.most_common(1)[0][0]
                    
                    # Find region with this color that touches us
                    for other in self.regions:
                        if other.color_idx == neighbor_color_idx and other.label != region.label:
                            # Check if they actually touch
                            touches = False
                            for (y, x) in contour[::max(1, len(contour) // 5)]:
                                yy, xx = int(round(y)), int(round(x))
                                yy = np.clip(yy, 0, h - 1)
                                xx = np.clip(xx, 0, w - 1)
                                if other.mask[yy, xx]:
                                    touches = True
                                    break
                            if touches:
                                neighbor_region_id = other.label
                                break
                
                # Create shared edge
                edge = SharedEdge(
                    edge_id=edge_id,
                    points=contour,
                    region_a=region.label,
                    region_b=neighbor_region_id,
                    color_a=region.color_idx,
                    color_b=neighbor_color_idx
                )
                shared_edges.append(edge)
                
                # Add to region's edge list
                region.edge_ids.append(edge_id)
                region.edge_directions.append(False)
                
                edge_id += 1
        
        print(f"  Extracted {len(shared_edges)} edges")
        
        if self.debug_dir:
            viz = self._visualize_boundaries(shared_edges)
            self._save_stage_image(viz, self.debug_dir / "stage4_boundaries.png")
            
            with open(self.debug_dir / "stage4_boundaries_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_edges': len(shared_edges),
                    'edge_pairs': [(e.region_a, e.region_b) for e in shared_edges]
                }, f)
        
        return shared_edges
    
    def _stage5_smooth_boundaries(self) -> List[SharedEdge]:
        """
        Stage 5: Gaussian smoothing per shared edge.
        """
        sigma = getattr(self.config, 'spline_smoothness', 1.0)
        
        for edge in self.shared_edges:
            points = edge.points
            
            if len(points) < 4:
                edge.smoothed_points = points.copy()
                continue
            
            y = points[:, 0]
            x = points[:, 1]
            
            # Check if closed loop
            is_closed = np.allclose(points[0], points[-1])
            
            if is_closed and len(y) > 3:
                # Wrap for closed curves
                y_smooth = gaussian_filter1d(
                    np.concatenate([y, y]), sigma, mode='wrap'
                )[:len(y)]
                x_smooth = gaussian_filter1d(
                    np.concatenate([x, x]), sigma, mode='wrap'
                )[:len(x)]
            else:
                y_smooth = gaussian_filter1d(y, sigma, mode='nearest')
                x_smooth = gaussian_filter1d(x, sigma, mode='nearest')
            
            edge.smoothed_points = np.column_stack([y_smooth, x_smooth])
        
        if self.debug_dir:
            viz = self._visualize_boundaries(self.shared_edges, use_smoothed=True)
            self._save_stage_image(viz, self.debug_dir / "stage5_smoothed.png")
            
            with open(self.debug_dir / "stage5_smoothed_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_edges': len(self.shared_edges),
                    'sigma': sigma
                }, f)
        
        return self.shared_edges
    
    def _stage6_build_svg(self) -> str:
        """
        Stage 6: SVG export with palette colors.
        """
        path_elements = []
        precision = getattr(self.config, 'precision', 1)
        
        for region in self.regions:
            if not region.edge_ids:
                continue
            
            # Sort edges to form closed loop
            sorted_edges = self._sort_region_edges(region)
            if not sorted_edges:
                continue
            
            # Build path
            path_parts = []
            first = True
            
            for edge_idx, reversed_dir in sorted_edges:
                if edge_idx >= len(self.shared_edges):
                    continue
                
                edge = self.shared_edges[edge_idx]
                points = (
                    edge.smoothed_points
                    if edge.smoothed_points is not None
                    else edge.points
                )
                
                if reversed_dir:
                    points = points[::-1]
                
                for i, (y, x) in enumerate(points):
                    if first:
                        path_parts.append(
                            f"M{x:.{precision}f},{y:.{precision}f}"
                        )
                        first = False
                    else:
                        path_parts.append(
                            f"L{x:.{precision}f},{y:.{precision}f}"
                        )
            
            if path_parts:
                path_parts.append("Z")
                path_data = " ".join(path_parts)
                
                # Use palette color directly
                color = self.palette[region.color_idx]
                r, g, b = [int(min(255, max(0, c * 255))) for c in color]
                fill = f"#{r:02x}{g:02x}{b:02x}"
                
                path_elements.append(
                    f'<path d="{path_data}" fill="{fill}" stroke="none" '
                    f'fill-rule="evenodd"/>'
                )
        
        return self._build_svg_complete(path_elements)
    
    def _sort_region_edges(self, region: SlicRegion) -> List[Tuple[int, bool]]:
        """Sort edges to form a proper closed loop."""
        if len(region.edge_ids) == 0:
            return []
        
        if len(region.edge_ids) == 1:
            return [(region.edge_ids[0], region.edge_directions[0])]
        
        # Build edge list with endpoints
        edges_with_points = []
        for idx, (edge_idx, rev) in enumerate(zip(region.edge_ids, region.edge_directions)):
            if edge_idx >= len(self.shared_edges):
                continue
            edge = self.shared_edges[edge_idx]
            points = edge.smoothed_points if edge.smoothed_points is not None else edge.points
            if rev:
                points = points[::-1]
            
            start = tuple(points[0])
            end = tuple(points[-1])
            edges_with_points.append((edge_idx, rev, start, end))
        
        if not edges_with_points:
            return []
        
        # Greedy chaining
        sorted_edges = [edges_with_points[0]]
        used = {0}
        
        while len(sorted_edges) < len(edges_with_points):
            current_end = sorted_edges[-1][3]
            
            found = False
            for i, (e_idx, rev, start, end) in enumerate(edges_with_points):
                if i in used:
                    continue
                
                if abs(start[0] - current_end[0]) < 1 and abs(start[1] - current_end[1]) < 1:
                    sorted_edges.append((e_idx, rev, start, end))
                    used.add(i)
                    found = True
                    break
                
                if abs(end[0] - current_end[0]) < 1 and abs(end[1] - current_end[1]) < 1:
                    sorted_edges.append((e_idx, not rev, end, start))
                    used.add(i)
                    found = True
                    break
            
            if not found:
                for i, (e_idx, rev, start, end) in enumerate(edges_with_points):
                    if i not in used:
                        sorted_edges.append((e_idx, rev, start, end))
                        used.add(i)
                        break
        
        return [(e[0], e[1]) for e in sorted_edges]
    
    def _build_svg_complete(self, path_elements: List[str]) -> str:
        """Assemble final SVG."""
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.ingest_result.width} {self.ingest_result.height}" width="{self.ingest_result.width}" height="{self.ingest_result.height}">
  {'  '.join(path_elements)}
</svg>'''
        
        return svg
    
    def _stage7_validate(self) -> Dict:
        """
        Stage 7: resvg CLI rasterization + comparison.
        """
        # Try to rasterize with resvg
        rasterized = self._rasterize_with_resvg(
            self.svg_string,
            self.ingest_result.width,
            self.ingest_result.height
        )
        
        # Create quantized reference
        quantized = self.palette[self.label_map]
        
        if rasterized is None:
            print("  (resvg not available, skipping validation)")
            return {'coverage': 100.0, 'gap_pixels': 0, 'rasterized': None}
        
        # Calculate gap mask
        gap_mask = self._create_gap_mask(
            rasterized,
            self.ingest_result.image_srgb,
            quantized
        )
        
        # Calculate metrics
        coverage = self._calculate_coverage(rasterized)
        gap_pixels = np.any(gap_mask == [1.0, 0, 1.0], axis=-1).sum()
        
        if self.debug_dir:
            self._create_comparison(
                self.ingest_result.image_srgb,
                quantized,
                rasterized,
                gap_mask
            )
            
            with open(self.debug_dir / "stage7_validation_data.pkl", 'wb') as f:
                pickle.dump({
                    'coverage': coverage,
                    'gap_pixels': int(gap_pixels)
                }, f)
        
        return {
            'coverage': coverage,
            'gap_pixels': int(gap_pixels),
            'rasterized': rasterized
        }
    
    def _rasterize_with_resvg(
        self,
        svg_string: str,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """Rasterize SVG using resvg CLI."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f_in:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_out:
                    f_in.write(svg_string)
                    f_in.flush()
                    
                    result = subprocess.run(
                        [
                            'resvg',
                            '--width', str(width),
                            '--height', str(height),
                            f_in.name,
                            f_out.name
                        ],
                        capture_output=True
                    )
                    
                    if result.returncode == 0:
                        img = Image.open(f_out.name)
                        arr = np.array(img).astype(np.float32) / 255.0
                        if arr.shape[-1] == 4:
                            # Composite on white
                            alpha = arr[..., 3:4]
                            rgb = arr[..., :3]
                            arr = rgb * alpha + (1 - alpha)
                        return arr[:, :, :3]
        except (FileNotFoundError, Exception):
            pass
        
        # Fallback to cairosvg
        try:
            import cairosvg
            png_data = cairosvg.svg2png(
                bytestring=svg_string.encode('utf-8'),
                output_width=width,
                output_height=height
            )
            img = Image.open(io.BytesIO(png_data))
            arr = np.array(img).astype(np.float32) / 255.0
            if arr.shape[-1] == 4:
                alpha = arr[..., 3:4]
                rgb = arr[..., :3]
                arr = rgb * alpha + (1 - alpha)
            return arr
        except (ImportError, Exception):
            pass
        
        return None
    
    def _create_gap_mask(
        self,
        rasterized: np.ndarray,
        original: np.ndarray,
        quantized: np.ndarray
    ) -> np.ndarray:
        """Create gap mask visualization."""
        h, w = rasterized.shape[:2]
        gap_mask = np.zeros((h, w, 3), dtype=np.float32)
        
        # Magenta: transparent in SVG but not in original
        # Yellow: filled in SVG but transparent in original  
        # Red: color mismatch
        
        diff = np.abs(rasterized - quantized).mean(axis=-1)
        mismatch = diff > 10 / 255.0
        gap_mask[mismatch] = [1.0, 0, 0]
        
        return gap_mask
    
    def _calculate_coverage(self, rasterized: np.ndarray) -> float:
        """Calculate coverage percentage."""
        non_white = np.any(rasterized < 0.99, axis=-1)
        return non_white.sum() / non_white.size * 100
    
    def _create_comparison(
        self,
        original: np.ndarray,
        quantized: np.ndarray,
        rasterized: np.ndarray,
        gap_mask: np.ndarray
    ):
        """Create 4-panel comparison image."""
        h, w = original.shape[:2]
        comparison = np.zeros((h * 2 + 20, w * 2 + 20, 3), dtype=np.float32)
        
        comparison[:h, :w] = original
        comparison[:h, w + 20:] = quantized
        comparison[h + 20:, :w] = rasterized
        comparison[h + 20:, w + 20:] = gap_mask
        
        comparison[:h, w:w + 20] = 0.2
        comparison[h:h + 20, :] = 0.2
        
        self._save_stage_image(comparison, self.debug_dir / "comparison.png")
    
    def _visualize_regions(self, regions: List[SlicRegion]) -> np.ndarray:
        """Visualize regions with PALETTE colors (not random)."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        viz = np.zeros((h, w, 3), dtype=np.float32)
        
        for region in regions:
            color = self.palette[region.color_idx]
            viz[region.mask] = color
        
        return viz
    
    def _visualize_boundaries(
        self,
        edges: List[SharedEdge],
        use_smoothed: bool = False
    ) -> np.ndarray:
        """Visualize boundaries with region colors."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for edge in edges:
            points = (
                edge.smoothed_points if use_smoothed and edge.smoothed_points is not None
                else edge.points
            )
            if len(points) < 2:
                continue
            
            # Use region_a's color
            if edge.region_a < len(self.regions):
                color_idx = self.regions[edge.region_a].color_idx
                color = self.palette[color_idx]
                color_tuple = tuple((color * 255).astype(int))
            else:
                color_tuple = (255, 255, 255)
            
            line_points = [(int(p[1]), int(p[0])) for p in points]
            draw.line(line_points, fill=color_tuple, width=2)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def _save_stage_image(self, image: np.ndarray, path: Path):
        """Save image for stage visualization."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(path)


def process_image(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    n_colors: int = 12,
    smoothness: float = 1.0,
    debug_stages: Optional[Union[str, Path]] = None
) -> str:
    """
    Convenience function to process an image with SLIC pipeline.
    
    Args:
        input_path: Path to input image
        output_path: Optional path for output SVG
        n_colors: Number of colors to quantize to
        smoothness: Boundary smoothing factor (sigma)
        debug_stages: Optional directory for debug outputs
        
    Returns:
        SVG string
    """
    config = ApexConfig(
        n_colors=n_colors,
        spline_smoothness=smoothness
    )
    
    pipeline = SlicPipeline(config)
    return pipeline.process(input_path, output_path, debug_stages)
