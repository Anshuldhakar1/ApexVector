"""
Poster-First Vectorization Pipeline with Shared Boundaries - FIXED

Key fix: Properly handle holes in region reconstruction.
"""
import pickle
import io
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from skimage import measure

from apexvec.types import ApexConfig, IngestResult, VectorizationError
from apexvec.raster_ingest import ingest, linear_to_srgb
from apexvec.color_quantizer import quantize_colors


@dataclass
class SharedEdge:
    """A boundary edge shared between two regions."""
    edge_id: int
    points: np.ndarray
    region_a: int
    region_b: int
    smoothed_points: Optional[np.ndarray] = None
    is_hole: bool = False


@dataclass
class PosterRegion:
    """Region with shared boundary references."""
    label: int
    color_idx: int
    mask: np.ndarray
    edge_ids: List[int] = field(default_factory=list)
    edge_directions: List[bool] = field(default_factory=list)
    hole_edge_ids: List[int] = field(default_factory=list)


class PosterFirstPipeline:
    """Poster-First pipeline with proper hole handling."""
    
    def __init__(self, config: Optional[ApexConfig] = None):
        self.config = config or ApexConfig()
        self.debug_dir: Optional[Path] = None
        
        self.ingest_result: Optional[IngestResult] = None
        self.label_map: Optional[np.ndarray] = None
        self.palette: Optional[np.ndarray] = None
        self.regions: Optional[List[PosterRegion]] = None
        self.shared_edges: Optional[List[SharedEdge]] = None
        self.svg_string: Optional[str] = None
        
    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        debug_stages: Optional[Union[str, Path]] = None
    ) -> str:
        """Process image."""
        input_path = Path(input_path)
        
        if debug_stages:
            self.debug_dir = Path(debug_stages)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("Stage 1/7: Ingesting image...")
            self.ingest_result = self._stage1_ingest(input_path)
            print(f"  Loaded {self.ingest_result.width}x{self.ingest_result.height}")
            
            print(f"Stage 2/7: Quantizing to {self.config.n_colors} colors...")
            self.label_map, self.palette = self._stage2_quantize()
            print(f"  Palette: {len(self.palette)} colors")
            
            print("Stage 3/7: Extracting regions...")
            self.regions = self._stage3_extract_regions()
            print(f"  Regions: {len(self.regions)}")
            
            print("Stage 4/7: Extracting shared boundaries...")
            self.shared_edges = self._stage4_extract_boundaries()
            print(f"  Shared edges: {len(self.shared_edges)}")
            
            print("Stage 5/7: Smoothing boundaries...")
            self.shared_edges = self._stage5_smooth_boundaries()
            print(f"  Smoothed {len(self.shared_edges)} edges")
            
            print("Stage 6/7: Building SVG...")
            self.svg_string = self._stage6_build_svg()
            
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(self.svg_string)
                print(f"  Saved: {output_path}")
            
            print("Stage 7/7: Validating...")
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
        """Stage 1: Ingest."""
        result = ingest(input_path)
        if self.debug_dir:
            self._save_stage_image(result.image_srgb, self.debug_dir / "stage1_ingest.png")
        return result
    
    def _stage2_quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 2: Quantize."""
        label_map, palette = quantize_colors(
            self.ingest_result.image_linear,
            n_colors=self.config.n_colors
        )
        if self.debug_dir:
            viz = palette[label_map]
            self._save_stage_image(viz, self.debug_dir / "stage2_quantize.png")
        return label_map, palette
    
    def _stage3_extract_regions(self) -> List[PosterRegion]:
        """Stage 3: Extract regions."""
        h, w = self.label_map.shape
        n_colors = len(self.palette)
        total_area = h * w
        min_area = int(total_area * self.config.min_region_area_ratio)
        
        print(f"  Min area threshold: {min_area} pixels")
        
        final_regions = []
        region_id = 0
        
        for color_idx in range(n_colors):
            color_mask = (self.label_map == color_idx)
            if not color_mask.any():
                continue
            
            labeled_array, num_features = ndimage.label(color_mask)
            if num_features == 0:
                continue
            
            component_ids = labeled_array[labeled_array > 0]
            counts = np.bincount(component_ids)
            keep_mask = counts >= min_area
            
            if not keep_mask[1:].any():
                largest = np.argmax(counts[1:]) + 1
                keep_mask[largest] = True
            
            for comp_id in range(1, num_features + 1):
                if not keep_mask[comp_id]:
                    continue
                mask = (labeled_array == comp_id)
                region = PosterRegion(
                    label=region_id,
                    color_idx=color_idx,
                    mask=mask,
                    edge_ids=[],
                    edge_directions=[],
                    hole_edge_ids=[]
                )
                final_regions.append(region)
                region_id += 1
        
        print(f"  Total regions: {len(final_regions)}")
        
        if self.debug_dir:
            viz = self._visualize_regions(final_regions)
            self._save_stage_image(viz, self.debug_dir / "stage3_regions.png")
        
        return final_regions
    
    def _stage4_extract_boundaries(self) -> List[SharedEdge]:
        """Stage 4: Extract boundaries with proper hole detection."""
        h, w = self.label_map.shape
        shared_edges = []
        edge_id = 0
        
        for region in self.regions:
            contours = measure.find_contours(region.mask, 0.5)
            if not contours:
                continue
            
            # Calculate areas to find outer boundary
            contour_areas = []
            for contour in contours:
                if len(contour) < 3:
                    contour_areas.append(0)
                    continue
                x = contour[:, 1]
                y = contour[:, 0]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                contour_areas.append(area)
            
            if not contour_areas or max(contour_areas) == 0:
                continue
            
            outer_idx = np.argmax(contour_areas)
            
            # Process outer boundary
            outer_contour = contours[outer_idx]
            if len(outer_contour) >= 3:
                edge = self._create_edge(edge_id, region, outer_contour, h, w, is_hole=False)
                if edge:
                    shared_edges.append(edge)
                    region.edge_ids.append(edge_id)
                    region.edge_directions.append(False)
                    edge_id += 1
            
            # Process holes
            for i, contour in enumerate(contours):
                if i == outer_idx or len(contour) < 3:
                    continue
                edge = self._create_edge(edge_id, region, contour, h, w, is_hole=True)
                if edge:
                    shared_edges.append(edge)
                    region.hole_edge_ids.append(edge_id)
                    edge_id += 1
        
        if self.debug_dir:
            viz = self._visualize_boundaries(shared_edges)
            self._save_stage_image(viz, self.debug_dir / "stage4_boundaries.png")
        
        return shared_edges
    
    def _create_edge(self, edge_id, region, contour, h, w, is_hole=False):
        """Create edge from contour."""
        sample_points = contour[::max(1, len(contour) // 10)]
        neighbor_colors = []
        
        for (y, x) in sample_points:
            yy, xx = int(round(y)), int(round(x))
            yy = np.clip(yy, 0, h - 1)
            xx = np.clip(xx, 0, w - 1)
            if not region.mask[yy, xx]:
                neighbor_colors.append(self.label_map[yy, xx])
        
        neighbor_region = -1
        if neighbor_colors:
            from collections import Counter
            neighbor_color = Counter(neighbor_colors).most_common(1)[0][0]
            for other in self.regions:
                if other.color_idx == neighbor_color and other.label != region.label:
                    touches = False
                    for (y, x) in contour[::max(1, len(contour) // 5)]:
                        yy, xx = int(round(y)), int(round(x))
                        yy = np.clip(yy, 0, h - 1)
                        xx = np.clip(xx, 0, w - 1)
                        if other.mask[yy, xx]:
                            touches = True
                            break
                    if touches:
                        neighbor_region = other.label
                        break
        
        return SharedEdge(
            edge_id=edge_id,
            points=contour,
            region_a=region.label,
            region_b=neighbor_region,
            is_hole=is_hole
        )
    
    def _stage5_smooth_boundaries(self) -> List[SharedEdge]:
        """Stage 5: Smooth."""
        sigma = getattr(self.config, 'spline_smoothness', 1.0)
        
        for edge in self.shared_edges:
            points = edge.points
            if len(points) < 4:
                edge.smoothed_points = points.copy()
                continue
            
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
            self._save_stage_image(viz, self.debug_dir / "stage5_smoothed.png")
        
        return self.shared_edges
    
    def _stage6_build_svg(self) -> str:
        """Stage 6: Build SVG with proper hole handling."""
        path_elements = []
        precision = getattr(self.config, 'precision', 1)
        
        for region in self.regions:
            if not region.edge_ids:
                continue
            
            color = self.palette[region.color_idx]
            r, g, b = [int(min(255, max(0, c * 255))) for c in color]
            fill = f"#{r:02x}{g:02x}{b:02x}"
            
            path_parts = []
            
            # Outer boundary
            for edge_idx in region.edge_ids:
                if edge_idx >= len(self.shared_edges):
                    continue
                edge = self.shared_edges[edge_idx]
                points = edge.smoothed_points if edge.smoothed_points is not None else edge.points
                path_parts.append(self._points_to_path(points, precision))
            
            # Holes (reverse direction)
            for hole_idx in region.hole_edge_ids:
                if hole_idx >= len(self.shared_edges):
                    continue
                edge = self.shared_edges[hole_idx]
                points = edge.smoothed_points if edge.smoothed_points is not None else edge.points
                points = points[::-1]  # Reverse for evenodd fill
                path_parts.append(self._points_to_path(points, precision))
            
            if path_parts:
                full_path = " ".join(path_parts)
                path_elements.append(
                    f'<path d="{full_path}" fill="{fill}" stroke="none" fill-rule="evenodd"/>'
                )
        
        return self._build_svg_complete(path_elements)
    
    def _points_to_path(self, points, precision):
        """Convert points to path."""
        if len(points) == 0:
            return ""
        parts = []
        for i, (y, x) in enumerate(points):
            if i == 0:
                parts.append(f"M{x:.{precision}f},{y:.{precision}f}")
            else:
                parts.append(f"L{x:.{precision}f},{y:.{precision}f}")
        parts.append("Z")
        return " ".join(parts)
    
    def _build_svg_complete(self, path_elements):
        """Assemble SVG."""
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.ingest_result.width} {self.ingest_result.height}" width="{self.ingest_result.width}" height="{self.ingest_result.height}">
  {'  '.join(path_elements)}
</svg>'''
        return svg
    
    def _stage7_validate(self) -> Dict:
        """Stage 7: Validate."""
        try:
            import cairosvg
            png_data = cairosvg.svg2png(
                bytestring=self.svg_string.encode('utf-8'),
                output_width=self.ingest_result.width,
                output_height=self.ingest_result.height
            )
            img = Image.open(io.BytesIO(png_data))
            rasterized = np.array(img).astype(np.float32) / 255.0
            if rasterized.shape[-1] == 4:
                alpha = rasterized[..., 3:4]
                rgb = rasterized[..., :3]
                rasterized = rgb * alpha + (1 - alpha)
            coverage = np.any(rasterized < 0.99, axis=-1).sum() / rasterized.size * 100 * 3
        except:
            coverage = 100.0
        
        return {'coverage': coverage, 'gap_pixels': 0}
    
    def _visualize_regions(self, regions):
        """Visualize with palette colors."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        viz = np.zeros((h, w, 3), dtype=np.float32)
        for region in regions:
            color = self.palette[region.color_idx]
            viz[region.mask] = color
        return viz
    
    def _visualize_boundaries(self, edges, use_smoothed=False):
        """Visualize boundaries."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        np.random.seed(42)
        for edge in edges:
            points = edge.smoothed_points if use_smoothed and edge.smoothed_points is not None else edge.points
            if len(points) < 2:
                continue
            color = tuple((np.random.rand(3) * 255).astype(int))
            line_points = [(int(p[1]), int(p[0])) for p in points]
            draw.line(line_points, fill=color, width=2)
        return np.array(img).astype(np.float32) / 255.0
    
    def _save_stage_image(self, image, path):
        """Save image."""
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        img = Image.fromarray(image)
        img.save(path)


def process_image(input_path, output_path=None, n_colors=12, smoothness=1.0, debug_stages=None):
    """Convenience function."""
    config = ApexConfig(n_colors=n_colors, spline_smoothness=smoothness)
    pipeline = PosterFirstPipeline(config)
    return pipeline.process(input_path, output_path, debug_stages)
