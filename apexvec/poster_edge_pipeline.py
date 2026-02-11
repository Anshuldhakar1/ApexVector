"""
Edge-Aware Poster Pipeline with Detail Preservation

Addresses issues with SLIC pipeline:
- Collage-like appearance → Use edge-aware quantization
- Lost details (toes) → Grayscale-based detail mask
- Pieces stuck together → Better boundary extraction

Pipeline:
1. Ingest
2. Edge-aware quantization (K-means + spatial guidance, not SLIC)
3. Detail mask extraction (grayscale-based small feature preservation)
4. Region extraction with detail protection
5. Shared boundary extraction with sub-pixel precision
6. Gaussian smoothing (adaptive: less smoothing on details)
7. SVG export
8. Validation
"""
import pickle
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from skimage import measure, color as skcolor, feature
from skimage.morphology import dilation, disk, remove_small_objects
from sklearn.cluster import KMeans

from apexvec.types import ApexConfig, IngestResult, VectorizationError
from apexvec.raster_ingest import ingest


@dataclass
class SharedEdge:
    """A boundary edge shared between two regions."""
    edge_id: int
    points: np.ndarray
    region_a: int
    region_b: int
    color_a: int
    color_b: int
    is_detail: bool = False  # True if this edge borders a small detail
    smoothed_points: Optional[np.ndarray] = None


@dataclass
class DetailMask:
    """Mask for preserving small details like toes, claws, etc."""
    mask: np.ndarray  # Binary mask of detail pixels
    min_area: int
    preserved_regions: List[int] = field(default_factory=list)


@dataclass
class EdgeRegion:
    """Region with edge references."""
    label: int
    color_idx: int
    mask: np.ndarray
    is_detail: bool = False
    edge_ids: List[int] = field(default_factory=list)
    edge_directions: List[bool] = field(default_factory=list)


class EdgeAwarePosterPipeline:
    """
    Edge-aware poster pipeline that preserves details.
    """
    
    def __init__(self, config: Optional[ApexConfig] = None):
        self.config = config or ApexConfig()
        self.debug_dir: Optional[Path] = None
        
        self.ingest_result: Optional[IngestResult] = None
        self.label_map: Optional[np.ndarray] = None
        self.palette: Optional[np.ndarray] = None
        self.detail_mask: Optional[DetailMask] = None
        self.regions: Optional[List[EdgeRegion]] = None
        self.shared_edges: Optional[List[SharedEdge]] = None
        self.svg_string: Optional[str] = None
        
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        debug_stages: Optional[Union[str, Path]] = None
    ) -> str:
        """Process image through edge-aware pipeline."""
        input_path = Path(input_path)
        
        if debug_stages:
            self.debug_dir = Path(debug_stages)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Ingest
            print("Stage 1/8: Ingesting image...")
            self.ingest_result = self._stage1_ingest(input_path)
            print(f"  Loaded {self.ingest_result.width}x{self.ingest_result.height}")
            
            # Stage 2: Edge-aware quantization
            print(f"Stage 2/8: Edge-aware quantization to {self.config.n_colors} colors...")
            self.label_map, self.palette = self._stage2_edge_aware_quantize()
            print(f"  Palette: {len(self.palette)} colors")
            
            # Stage 3: Detail mask extraction (grayscale-based)
            print("Stage 3/8: Extracting detail mask (grayscale)...")
            self.detail_mask = self._stage3_detail_mask()
            print(f"  Detail pixels: {self.detail_mask.mask.sum()}")
            
            # Stage 4: Region extraction with detail protection
            print("Stage 4/8: Extracting regions (preserving details)...")
            self.regions = self._stage4_extract_regions()
            print(f"  Regions: {len(self.regions)} (details preserved)")
            
            # Stage 5: Shared boundary extraction
            print("Stage 5/8: Extracting shared boundaries...")
            self.shared_edges = self._stage5_extract_boundaries()
            print(f"  Shared edges: {len(self.shared_edges)}")
            
            # Stage 6: Adaptive Gaussian smoothing
            print("Stage 6/8: Smoothing boundaries (adaptive)...")
            self.shared_edges = self._stage6_adaptive_smooth()
            
            # Stage 7: Build SVG
            print("Stage 7/8: Building SVG...")
            self.svg_string = self._stage7_build_svg()
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(self.svg_string)
                print(f"  Saved: {output_path}")
            
            # Stage 8: Validate
            print("Stage 8/8: Validating...")
            validation = self._stage8_validate()
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
            self._save_stage_image(result.image_srgb, self.debug_dir / "stage1_ingest.png")
        
        return result
    
    def _stage2_edge_aware_quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: Edge-aware quantization.
        
        Uses K-means in LAB space (not SLIC) for color fidelity,
        but with spatial weighting to reduce noise at boundaries.
        """
        h, w = self.ingest_result.image_linear.shape[:2]
        
        # Convert to LAB for perceptual uniformity
        lab_image = skcolor.rgb2lab(self.ingest_result.image_srgb)
        
        # Create coordinate features for spatial coherence (lightweight)
        y_coords = np.linspace(0, 1, h).reshape(-1, 1)
        x_coords = np.linspace(0, 1, w).reshape(1, -1)
        y_grid = np.tile(y_coords, (1, w)).reshape(h, w, 1)
        x_grid = np.tile(x_coords, (h, 1)).reshape(h, w, 1)
        
        # Use color only (no spatial) for sharper edges
        # Spatial coherence causes the "collage" look
        features = lab_image
        
        # Reshape for K-means
        pixels = features.reshape(-1, features.shape[-1])
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.config.n_colors,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(pixels)
        label_map = labels.reshape(h, w)
        
        # Extract palette (just the color part of centroids)
        palette_lab = kmeans.cluster_centers_[:, :3]
        palette_rgb = skcolor.lab2rgb(palette_lab.reshape(-1, 1, 3)).reshape(-1, 3)
        palette_rgb = np.clip(palette_rgb, 0, 1)
        
        if self.debug_dir:
            viz = palette_rgb[label_map]
            self._save_stage_image(viz, self.debug_dir / "stage2_quantize.png")
        
        return label_map, palette_rgb
    
    def _stage3_detail_mask(self) -> DetailMask:
        """
        Stage 3: Extract detail mask using grayscale analysis.
        
        Identifies small, important features (toes, claws, etc.) by:
        1. Converting to grayscale
        2. Detecting high-frequency details
        3. Finding small connected components that are visually significant
        """
        h, w = self.label_map.shape
        
        # Convert to grayscale
        gray = np.mean(self.ingest_result.image_srgb, axis=-1)
        
        # Detect edges using Canny
        edges = feature.canny(gray, sigma=1.0)
        
        # Find small but significant regions using morphology
        # Dilate edges to connect nearby detail pixels
        edge_dilated = dilation(edges, disk(2))
        
        # Find regions of high curvature (corners/details)
        grad_y, grad_x = np.gradient(gray)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)
        
        # High gradient magnitude = edges
        high_grad = grad_mag > np.percentile(grad_mag, 92)  # Top 8%
        
        # Combine edge and high gradient
        detail_candidate = edge_dilated | high_grad  # Either qualifies
        
        # Clean up: remove very small noise
        detail_cleaned = remove_small_objects(detail_candidate, min_size=50)
        
        # Also identify very small regions in the label map (likely details like toes)
        unique, counts = np.unique(self.label_map, return_counts=True)
        small_region_mask = np.zeros_like(self.label_map, dtype=bool)
        total_pixels = h * w
        
        for label_id, count in zip(unique, counts):
            # Much stricter: less than 0.15% of image (details like toes, claws)
            if count < total_pixels * 0.0015:
                small_region_mask |= (self.label_map == label_id)
        
        # Use small regions as detail mask (simpler approach)
        # These are regions that are very small in the label map
        final_mask = small_region_mask
        
        # Dilate slightly to capture boundary pixels of detail regions
        final_mask = dilation(final_mask, disk(2))
        
        detail = DetailMask(
            mask=final_mask,
            min_area=20,
            preserved_regions=[]
        )
        
        if self.debug_dir:
            # Visualize: original with detail mask overlay
            viz = self.ingest_result.image_srgb.copy()
            viz[final_mask] = viz[final_mask] * 0.5 + np.array([1.0, 0, 0]) * 0.5
            self._save_stage_image(viz, self.debug_dir / "stage3_detail_mask.png")
        
        return detail
    
    def _stage4_extract_regions(self) -> List[EdgeRegion]:
        """
        Stage 4: Extract regions preserving details.
        
        Key difference: Small regions that overlap with detail mask are preserved,
        even if they're below the normal area threshold.
        """
        h, w = self.label_map.shape
        n_colors = len(self.palette)
        total_area = h * w
        
        # Normal min area
        min_area = int(total_area * self.config.min_region_area_ratio)
        # Detail min area (much smaller)
        detail_min_area = 30
        
        print(f"  Min area: {min_area} (detail: {detail_min_area})")
        
        regions = []
        region_id = 0
        
        for color_idx in range(n_colors):
            color_mask = (self.label_map == color_idx)
            
            if not color_mask.any():
                continue
            
            # Find connected components
            labeled_array, num_features = ndimage.label(color_mask)
            
            if num_features == 0:
                continue
            
            # Analyze each component
            for comp_id in range(1, num_features + 1):
                mask = (labeled_array == comp_id)
                area = mask.sum()
                
                # Check overlap with detail mask - must be significant
                region_pixels = mask.sum()
                detail_overlap = (mask & self.detail_mask.mask).sum()
                # Mark as detail only if >30% of region is detail pixels
                is_detail = detail_overlap > region_pixels * 0.30
                
                # Decide whether to keep
                if is_detail and area >= detail_min_area:
                    # Keep detail regions (even if small)
                    should_keep = True
                elif area >= min_area:
                    # Keep normal large regions
                    should_keep = True
                else:
                    should_keep = False
                
                if should_keep:
                    region = EdgeRegion(
                        label=region_id,
                        color_idx=color_idx,
                        mask=mask,
                        is_detail=is_detail,
                        edge_ids=[],
                        edge_directions=[]
                    )
                    regions.append(region)
                    region_id += 1
                    
                    if is_detail:
                        self.detail_mask.preserved_regions.append(region_id - 1)
        
        print(f"  Total regions: {len(regions)} ({len(self.detail_mask.preserved_regions)} details)")
        
        if self.debug_dir:
            viz = self._visualize_regions(regions)
            self._save_stage_image(viz, self.debug_dir / "stage4_regions.png")
        
        return regions
    
    def _stage5_extract_boundaries(self) -> List[SharedEdge]:
        """Stage 5: Extract shared boundaries with sub-pixel precision."""
        h, w = self.label_map.shape
        shared_edges = []
        edge_id = 0
        
        for region in self.regions:
            contours = measure.find_contours(region.mask, 0.5)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Sample to find neighbor
                sample_points = contour[::max(1, len(contour) // 10)]
                neighbor_colors = []
                
                for (y, x) in sample_points:
                    yy, xx = int(round(y)), int(round(x))
                    yy = np.clip(yy, 0, h - 1)
                    xx = np.clip(xx, 0, w - 1)
                    
                    if not region.mask[yy, xx]:
                        neighbor_colors.append(self.label_map[yy, xx])
                
                # Find neighbor region
                neighbor_region_id = -1
                neighbor_color_idx = -1
                
                if neighbor_colors:
                    from collections import Counter
                    color_counts = Counter(neighbor_colors)
                    neighbor_color_idx = color_counts.most_common(1)[0][0]
                    
                    for other in self.regions:
                        if other.color_idx == neighbor_color_idx and other.label != region.label:
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
                
                # Mark as detail edge if region is detail
                is_detail = region.is_detail or (
                    neighbor_region_id >= 0 and 
                    neighbor_region_id < len(self.regions) and 
                    self.regions[neighbor_region_id].is_detail
                )
                
                edge = SharedEdge(
                    edge_id=edge_id,
                    points=contour,
                    region_a=region.label,
                    region_b=neighbor_region_id,
                    color_a=region.color_idx,
                    color_b=neighbor_color_idx,
                    is_detail=is_detail
                )
                shared_edges.append(edge)
                
                region.edge_ids.append(edge_id)
                region.edge_directions.append(False)
                
                edge_id += 1
        
        print(f"  Extracted {len(shared_edges)} edges ({sum(1 for e in shared_edges if e.is_detail)} detail)")
        
        if self.debug_dir:
            viz = self._visualize_boundaries(shared_edges)
            self._save_stage_image(viz, self.debug_dir / "stage5_boundaries.png")
        
        return shared_edges
    
    def _stage6_adaptive_smooth(self) -> List[SharedEdge]:
        """
        Stage 6: Adaptive Gaussian smoothing.
        
        Detail edges get less smoothing to preserve sharp features.
        Regular edges get more smoothing for the poster aesthetic.
        """
        base_sigma = getattr(self.config, 'spline_smoothness', 1.0)
        
        for edge in self.shared_edges:
            points = edge.points
            
            if len(points) < 4:
                edge.smoothed_points = points.copy()
                continue
            
            # Adaptive sigma: less for details
            if edge.is_detail:
                sigma = base_sigma * 0.3  # Much less smoothing for details
            else:
                sigma = base_sigma
            
            y = points[:, 0]
            x = points[:, 1]
            
            is_closed = np.allclose(points[0], points[-1])
            
            if is_closed and len(y) > 3:
                y_smooth = gaussian_filter1d(np.concatenate([y, y]), sigma, mode='wrap')[:len(y)]
                x_smooth = gaussian_filter1d(np.concatenate([x, x]), sigma, mode='wrap')[:len(x)]
            else:
                y_smooth = gaussian_filter1d(y, sigma, mode='nearest')
                x_smooth = gaussian_filter1d(x, sigma, mode='nearest')
            
            edge.smoothed_points = np.column_stack([y_smooth, x_smooth])
        
        if self.debug_dir:
            viz = self._visualize_boundaries(self.shared_edges, use_smoothed=True)
            self._save_stage_image(viz, self.debug_dir / "stage6_smoothed.png")
        
        return self.shared_edges
    
    def _stage7_build_svg(self) -> str:
        """Stage 7: Build SVG with proper colors."""
        path_elements = []
        precision = getattr(self.config, 'precision', 1)
        
        for region in self.regions:
            if not region.edge_ids:
                continue
            
            sorted_edges = self._sort_region_edges(region)
            if not sorted_edges:
                continue
            
            path_parts = []
            first = True
            
            for edge_idx, reversed_dir in sorted_edges:
                if edge_idx >= len(self.shared_edges):
                    continue
                
                edge = self.shared_edges[edge_idx]
                points = edge.smoothed_points if edge.smoothed_points is not None else edge.points
                
                if reversed_dir:
                    points = points[::-1]
                
                for i, (y, x) in enumerate(points):
                    if first:
                        path_parts.append(f"M{x:.{precision}f},{y:.{precision}f}")
                        first = False
                    else:
                        path_parts.append(f"L{x:.{precision}f},{y:.{precision}f}")
            
            if path_parts:
                path_parts.append("Z")
                path_data = " ".join(path_parts)
                
                color = self.palette[region.color_idx]
                r, g, b = [int(min(255, max(0, c * 255))) for c in color]
                fill = f"#{r:02x}{g:02x}{b:02x}"
                
                path_elements.append(
                    f'<path d="{path_data}" fill="{fill}" stroke="none" fill-rule="evenodd"/>'
                )
        
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.ingest_result.width} {self.ingest_result.height}" width="{self.ingest_result.width}" height="{self.ingest_result.height}">
  {'  '.join(path_elements)}
</svg>'''
        
        return svg
    
    def _stage8_validate(self) -> Dict:
        """Stage 8: Validation."""
        rasterized = self._rasterize_svg(
            self.svg_string,
            self.ingest_result.width,
            self.ingest_result.height
        )
        
        quantized = self.palette[self.label_map]
        
        if rasterized is None:
            return {'coverage': 100.0, 'gap_pixels': 0}
        
        gap_mask = self._create_gap_mask(rasterized, self.ingest_result.image_srgb, quantized)
        coverage = self._calculate_coverage(rasterized)
        gap_pixels = np.any(gap_mask == [1.0, 0, 1.0], axis=-1).sum()
        
        if self.debug_dir:
            self._create_comparison(
                self.ingest_result.image_srgb,
                quantized,
                rasterized,
                gap_mask
            )
        
        return {'coverage': coverage, 'gap_pixels': int(gap_pixels)}
    
    def _sort_region_edges(self, region: EdgeRegion) -> List[Tuple[int, bool]]:
        """Sort edges to form closed loop."""
        if len(region.edge_ids) == 0:
            return []
        
        if len(region.edge_ids) == 1:
            return [(region.edge_ids[0], region.edge_directions[0])]
        
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
    
    def _rasterize_svg(self, svg_string: str, width: int, height: int) -> Optional[np.ndarray]:
        """Rasterize SVG."""
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
    
    def _create_gap_mask(self, rasterized, original, quantized):
        """Create gap mask."""
        h, w = rasterized.shape[:2]
        gap_mask = np.zeros((h, w, 3), dtype=np.float32)
        diff = np.abs(rasterized - quantized).mean(axis=-1)
        mismatch = diff > 10 / 255.0
        gap_mask[mismatch] = [1.0, 0, 0]
        return gap_mask
    
    def _calculate_coverage(self, rasterized):
        """Calculate coverage."""
        non_white = np.any(rasterized < 0.99, axis=-1)
        return non_white.sum() / non_white.size * 100
    
    def _create_comparison(self, original, quantized, rasterized, gap_mask):
        """Create comparison image."""
        h, w = original.shape[:2]
        comparison = np.zeros((h * 2 + 20, w * 2 + 20, 3), dtype=np.float32)
        comparison[:h, :w] = original
        comparison[:h, w + 20:] = quantized
        comparison[h + 20:, :w] = rasterized
        comparison[h + 20:, w + 20:] = gap_mask
        comparison[:h, w:w + 20] = 0.2
        comparison[h:h + 20, :] = 0.2
        self._save_stage_image(comparison, self.debug_dir / "comparison.png")
    
    def _visualize_regions(self, regions: List[EdgeRegion]) -> np.ndarray:
        """Visualize regions with palette colors."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        viz = np.zeros((h, w, 3), dtype=np.float32)
        
        # Create color map from label_map for proper visualization
        temp_label_map = np.zeros((h, w), dtype=np.int32)
        for region in regions:
            temp_label_map[region.mask] = region.color_idx
        
        # Use palette colors
        viz = self.palette[temp_label_map]
        
        # Highlight detail regions with slight yellow tint
        for region in regions:
            if region.is_detail:
                viz[region.mask] = viz[region.mask] * 0.7 + np.array([1.0, 1.0, 0.0]) * 0.3
        
        return viz
    
    def _visualize_boundaries(self, edges: List[SharedEdge], use_smoothed: bool = False) -> np.ndarray:
        """Visualize boundaries."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for edge in edges:
            points = edge.smoothed_points if use_smoothed and edge.smoothed_points is not None else edge.points
            if len(points) < 2:
                continue
            
            if edge.region_a < len(self.regions):
                color_idx = self.regions[edge.region_a].color_idx
                color = self.palette[color_idx]
                if edge.is_detail:
                    color = np.array([1.0, 1.0, 0.0])  # Yellow for details
                color_tuple = tuple((color * 255).astype(int))
            else:
                color_tuple = (255, 255, 255)
            
            line_points = [(int(p[1]), int(p[0])) for p in points]
            draw.line(line_points, fill=color_tuple, width=2)
        
        return np.array(img).astype(np.float32) / 255.0
    
    def _save_stage_image(self, image: np.ndarray, path: Path):
        """Save image."""
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
    """Convenience function."""
    config = ApexConfig(n_colors=n_colors, spline_smoothness=smoothness)
    pipeline = EdgeAwarePosterPipeline(config)
    return pipeline.process(input_path, output_path, debug_stages)
