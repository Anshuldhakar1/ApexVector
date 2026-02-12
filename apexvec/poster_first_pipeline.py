"""
Poster-First Vectorization Pipeline with Shared Boundaries

Based on validation spike results showing shared-boundary + Gaussian smoothing
produces gap-free, flat-color SVGs.

Key features:
1. Locked palette - quantized colors are immutable
2. Shared boundary extraction - one edge between each region pair
3. Gaussian smoothing per-edge - smooth once, use twice (both directions)
4. Per-stage debug visualization with gap mask
"""
import pickle
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
    points: np.ndarray  # (N, 2) array of (y, x) coordinates
    region_a: int       # Region ID (not color index)
    region_b: int       # Region ID (-1 for background)
    smoothed_points: Optional[np.ndarray] = None


@dataclass
class PosterRegion:
    """Region with shared boundary references for poster pipeline."""
    label: int
    color_idx: int
    mask: np.ndarray    # Binary mask
    edge_ids: List[int] = field(default_factory=list)
    edge_directions: List[bool] = field(default_factory=list)  # True if reversed


class PosterFirstPipeline:
    """
    Poster-First vectorization pipeline with shared boundaries.
    
    Pipeline stages:
    1. Ingest - Load image, convert to linear RGB
    2. Quantize - K-means in LAB space, locked palette
    3. Extract regions - Connected components with color preservation
    4. Extract shared boundaries - One edge per adjacent region pair
    5. Smooth boundaries - Gaussian smoothing per shared edge
    6. Build SVG - Reconstruct regions from smoothed edges
    7. Rasterize & validate - Compare output to original
    """
    
    def __init__(self, config: Optional[ApexConfig] = None):
        """
        Initialize pipeline.
        
        Args:
            config: Apex configuration. Uses defaults if None.
        """
        self.config = config or ApexConfig()
        self.debug_dir: Optional[Path] = None
        
        # Stage outputs for debug
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
        """
        Process image through poster-first pipeline.
        
        Args:
            input_path: Path to input image
            output_path: Optional path for output SVG
            debug_stages: Optional directory to save stage visualizations
            
        Returns:
            SVG string
        """
        input_path = Path(input_path)
        
        # Setup debug directory
        if debug_stages:
            self.debug_dir = Path(debug_stages)
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Stage 1: Ingest
            print("Stage 1/7: Ingesting image...")
            self.ingest_result = self._stage1_ingest(input_path)
            print(f"  Loaded {self.ingest_result.width}x{self.ingest_result.height}")
            
            # Stage 2: Quantize
            print(f"Stage 2/7: Quantizing to {self.config.n_colors} colors...")
            self.label_map, self.palette = self._stage2_quantize()
            print(f"  Palette: {len(self.palette)} colors")
            
            # Stage 3: Extract regions
            print("Stage 3/7: Extracting regions...")
            self.regions = self._stage3_extract_regions()
            print(f"  Regions: {len(self.regions)}")
            
            # Stage 4: Extract shared boundaries
            print("Stage 4/7: Extracting shared boundaries...")
            self.shared_edges = self._stage4_extract_boundaries()
            print(f"  Shared edges: {len(self.shared_edges)}")
            
            # Stage 5: Smooth boundaries
            print("Stage 5/7: Smoothing boundaries...")
            self.shared_edges = self._stage5_smooth_boundaries()
            print(f"  Smoothed {len(self.shared_edges)} edges")
            
            # Stage 6: Build SVG
            print("Stage 6/7: Building SVG...")
            self.svg_string = self._stage6_build_svg()
            
            # Save output
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(self.svg_string)
                print(f"  Saved: {output_path}")
            
            # Stage 7: Validate
            print("Stage 7/7: Validating...")
            validation = self._stage7_validate()
            print(f"  Coverage: {validation['coverage']:.1f}%")
            print(f"  Gap pixels: {validation['gap_pixels']}")
            
            return self.svg_string
            
        except Exception as e:
            # Write error file if debugging
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
            # Save stage visualization
            self._save_stage_image(
                result.image_srgb,
                self.debug_dir / "stage1_ingest.png"
            )
            # Save stage data
            with open(self.debug_dir / "stage1_ingest_data.pkl", 'wb') as f:
                pickle.dump({
                    'width': result.width,
                    'height': result.height,
                    'has_alpha': result.has_alpha
                }, f)
        
        return result
    
    def _stage2_quantize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 2: Color quantization with locked palette."""
        label_map, palette = quantize_colors(
            self.ingest_result.image_linear,
            n_colors=self.config.n_colors
        )
        
        # Verify invariant: all labels map to valid palette indices
        unique_labels = np.unique(label_map)
        expected = set(range(len(palette)))
        actual = set(unique_labels)
        
        if not actual.issubset(expected):
            raise VectorizationError(
                f"Label map contains invalid indices: {actual - expected}"
            )
        
        if self.debug_dir:
            # Create quantization visualization
            viz = self._visualize_quantization(label_map, palette)
            self._save_stage_image(viz, self.debug_dir / "stage2_quantize.png")
            
            with open(self.debug_dir / "stage2_quantize_data.pkl", 'wb') as f:
                pickle.dump({
                    'label_map': label_map,
                    'palette': palette
                }, f)
        
        return label_map, palette
    
# This file contains the optimized _stage3_extract_regions function
# Replace the function in poster_first_pipeline.py with this one

    def _stage3_extract_regions(self) -> List[PosterRegion]:
        """Stage 3: Extract regions preserving all colors (fast version)."""
        h, w = self.label_map.shape
        n_colors = len(self.palette)
        total_area = h * w
        min_area = int(total_area * self.config.min_region_area_ratio)
        
        print(f"  Min area threshold: {min_area} pixels")
        
        # Strategy: Extract connected components per color, filter tiny speckles
        final_regions = []
        region_id = 0
        
        for color_idx in range(n_colors):
            color_mask = (self.label_map == color_idx)
            
            if not color_mask.any():
                continue
            
            # Find connected components for this color
            labeled_array, num_features = ndimage.label(color_mask)
            
            if num_features == 0:
                continue
            
            # Get region properties using bincount for speed
            component_ids = labeled_array[labeled_array > 0]
            counts = np.bincount(component_ids)
            
            # Keep components above min_area, OR if it's the only one of this color
            keep_mask = counts >= min_area
            
            if not keep_mask[1:].any():
                # No components above threshold, keep the largest one
                largest = np.argmax(counts[1:]) + 1
                keep_mask[largest] = True
            
            # Create regions for kept components
            for comp_id in range(1, num_features + 1):
                if not keep_mask[comp_id]:
                    continue
                    
                mask = (labeled_array == comp_id)
                
                region = PosterRegion(
                    label=region_id,
                    color_idx=color_idx,
                    mask=mask,
                    edge_ids=[],
                    edge_directions=[]
                )
                final_regions.append(region)
                region_id += 1
        
        print(f"  Total regions: {len(final_regions)}")
        
        # Verify all colors preserved
        remaining_colors = set(r.color_idx for r in final_regions)
        missing = set(range(n_colors)) - remaining_colors
        if missing:
            print(f"  WARNING: Missing colors: {missing}")
        else:
            print(f"  OK: All {n_colors} colors preserved")
        
        if self.debug_dir:
            # Create region visualization
            viz = self._visualize_regions(final_regions)
            self._save_stage_image(viz, self.debug_dir / "stage3_regions.png")
            
            with open(self.debug_dir / "stage3_regions_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_regions': len(final_regions),
                    'color_indices': [r.color_idx for r in final_regions]
                }, f)
        
        return final_regions
    
    def _stage4_extract_boundaries(self) -> List[SharedEdge]:
        """Stage 4: Extract shared boundaries between adjacent regions."""
        h, w = self.label_map.shape
        shared_edges = []
        edge_id = 0
        
        for region in self.regions:
            # Find contours
            contours = measure.find_contours(region.mask, 0.5)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                # Determine neighbor by sampling
                sample_points = contour[::max(1, len(contour) // 10)]
                neighbor_colors = []
                
                for (y, x) in sample_points:
                    yy, xx = int(round(y)), int(round(x))
                    yy = np.clip(yy, 0, h - 1)
                    xx = np.clip(xx, 0, w - 1)
                    
                    if not region.mask[yy, xx]:
                        neighbor_colors.append(self.label_map[yy, xx])
                
                # Find neighbor region
                neighbor_region = -1
                if neighbor_colors:
                    from collections import Counter
                    neighbor_color = Counter(neighbor_colors).most_common(1)[0][0]
                    
                    for other in self.regions:
                        if other.color_idx == neighbor_color:
                            # Check if touches
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
                
                # Create shared edge
                edge = SharedEdge(
                    edge_id=edge_id,
                    points=contour,
                    region_a=region.label,
                    region_b=neighbor_region
                )
                shared_edges.append(edge)
                
                # Add to region
                region.edge_ids.append(edge_id)
                region.edge_directions.append(False)
                
                edge_id += 1
        
        if self.debug_dir:
            # Create boundary visualization
            viz = self._visualize_boundaries(shared_edges)
            self._save_stage_image(viz, self.debug_dir / "stage4_boundaries.png")
            
            with open(self.debug_dir / "stage4_boundaries_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_edges': len(shared_edges),
                    'edge_pairs': [(e.region_a, e.region_b) for e in shared_edges]
                }, f)
        
        return shared_edges
    
    def _stage5_smooth_boundaries(self) -> List[SharedEdge]:
        """Stage 5: Gaussian smoothing on shared edges."""
        sigma = self.config.spline_smoothness  # Use directly, default is 0.1
        
        for edge in self.shared_edges:
            points = edge.points
            
            if len(points) < 4:
                edge.smoothed_points = points.copy()
                continue
            
            y = points[:, 0]
            x = points[:, 1]
            
            # Check if closed
            is_closed = np.allclose(points[0], points[-1])
            
            if is_closed and len(y) > 3:
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
            # Create smoothed boundary visualization
            viz = self._visualize_boundaries(
                self.shared_edges, use_smoothed=True
            )
            self._save_stage_image(viz, self.debug_dir / "stage5_smoothed.png")
            
            with open(self.debug_dir / "stage5_smoothed_data.pkl", 'wb') as f:
                pickle.dump({
                    'n_edges': len(self.shared_edges),
                    'sigma': sigma
                }, f)
        
        return self.shared_edges
    
    def _stage6_build_svg(self) -> str:
        """Stage 6: Build SVG from regions and smoothed edges."""
        path_elements = []
        precision = self.config.precision
        
        for region in self.regions:
            if not region.edge_ids:
                continue
            
            # Sort edges to form a closed loop
            sorted_edges = self._sort_region_edges(region)
            if not sorted_edges:
                continue
            
            # Build path from sorted edges
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
                    elif i == 0:
                        # Start of new edge - should connect to previous
                        # Skip if same as last point to avoid duplicates
                        last_point = path_parts[-1]
                        # Extract coords from last "Lx,y" or "Mx,y"
                        try:
                            last_coords = last_point[1:].split(',')
                            last_x, last_y = float(last_coords[0]), float(last_coords[1])
                            if abs(x - last_x) > 0.01 or abs(y - last_y) > 0.01:
                                path_parts.append(
                                    f"L{x:.{precision}f},{y:.{precision}f}"
                                )
                        except:
                            path_parts.append(
                                f"L{x:.{precision}f},{y:.{precision}f}"
                            )
                    else:
                        path_parts.append(
                            f"L{x:.{precision}f},{y:.{precision}f}"
                        )
            
            if path_parts:
                path_parts.append("Z")
                path_data = " ".join(path_parts)
                
                # Get color from palette - KEY: use palette, not mean_color
                color = self.palette[region.color_idx]
                r, g, b = [int(min(255, max(0, c * 255))) for c in color]
                fill = f"#{r:02x}{g:02x}{b:02x}"
                
                path_elements.append(
                    f'<path d="{path_data}" fill="{fill}" stroke="none" '
                    f'fill-rule="evenodd"/>'
                )
        
        return self._stage6_build_svg_complete(path_elements)
    
    def _sort_region_edges(self, region: PosterRegion) -> List[Tuple[int, bool]]:
        """
        Sort edges to form a proper closed loop.
        
        Returns list of (edge_idx, reversed) tuples in order.
        """
        if len(region.edge_ids) == 0:
            return []
        
        if len(region.edge_ids) == 1:
            return [(region.edge_ids[0], region.edge_directions[0])]
        
        # Build edge list with their endpoints
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
        
        # Simple greedy chaining
        sorted_edges = [edges_with_points[0]]
        used = {0}
        
        while len(sorted_edges) < len(edges_with_points):
            current_end = sorted_edges[-1][3]  # End of last edge
            
            # Find next edge that starts at current_end
            found = False
            for i, (e_idx, rev, start, end) in enumerate(edges_with_points):
                if i in used:
                    continue
                
                # Check if this edge starts where current ends
                if abs(start[0] - current_end[0]) < 1 and abs(start[1] - current_end[1]) < 1:
                    sorted_edges.append((e_idx, rev, start, end))
                    used.add(i)
                    found = True
                    break
                
                # Check if reversed edge starts where current ends
                if abs(end[0] - current_end[0]) < 1 and abs(end[1] - current_end[1]) < 1:
                    # Use reversed
                    sorted_edges.append((e_idx, not rev, end, start))
                    used.add(i)
                    found = True
                    break
            
            if not found:
                # Can't find connected edge, just add remaining
                for i, (e_idx, rev, start, end) in enumerate(edges_with_points):
                    if i not in used:
                        sorted_edges.append((e_idx, rev, start, end))
                        used.add(i)
                        break
        
        return [(e[0], e[1]) for e in sorted_edges]
    
    def _stage6_build_svg_complete(self, path_elements):
        """Assemble final SVG."""
        svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.ingest_result.width} {self.ingest_result.height}" width="{self.ingest_result.width}" height="{self.ingest_result.height}">
  {'  '.join(path_elements)}
</svg>'''
        
        if self.debug_dir:
            with open(self.debug_dir / "stage6_final_data.pkl", 'wb') as f:
                pickle.dump({'svg_length': len(svg)}, f)
        
        return svg
    
    def _stage7_validate(self) -> Dict:
        """Stage 7: Validate and create comparison."""
        # Rasterize SVG
        rasterized = self._rasterize_svg(
            self.svg_string,
            self.ingest_result.width,
            self.ingest_result.height
        )
        
        # Create quantized reference
        quantized = self.palette[self.label_map]
        
        if rasterized is None:
            # Rasterization not available, skip validation
            print("  (Rasterization not available, skipping validation)")
            return {
                'coverage': 100.0,
                'gap_pixels': 0,
                'rasterized': None
            }
        
        # Calculate gap mask
        # Magenta: transparent in SVG but not in original
        # Yellow: filled in SVG but transparent in original
        # Red overlay: color mismatch
        gap_mask = self._create_gap_mask(
            rasterized,
            self.ingest_result.image_srgb,
            quantized
        )
        
        # Calculate metrics
        coverage = self._calculate_coverage(rasterized)
        gap_pixels = np.any(gap_mask == [1.0, 0, 1.0], axis=-1).sum()
        
        if self.debug_dir:
            # Create comparison visualization
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
    
    def _rasterize_svg(
        self,
        svg_string: str,
        width: int,
        height: int
    ) -> Optional[np.ndarray]:
        """Rasterize SVG to numpy array. Returns None if rasterization fails."""
        import subprocess
        import tempfile
        import io
        
        # Try CairoSVG first
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
                # Composite on white
                alpha = arr[..., 3:4]
                rgb = arr[..., :3]
                arr = rgb * alpha + (1 - alpha)
            return arr
        except (ImportError, Exception):
            pass
        
        # Try rsvg-convert
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(svg_string)
                f.flush()
                svg_path = f.name
                
                result = subprocess.run(
                    ['rsvg-convert', '-w', str(width), '-h', str(height), svg_path],
                    capture_output=True
                )
                
                if result.returncode == 0:
                    img = Image.open(io.BytesIO(result.stdout))
                    arr = np.array(img).astype(np.float32) / 255.0
                    if len(arr.shape) == 2:
                        arr = np.stack([arr] * 3, axis=-1)
                    return arr[:, :, :3]
        except (FileNotFoundError, Exception):
            pass
        
        # Try Inkscape
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f_in:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f_out:
                    f_in.write(svg_string)
                    f_in.flush()
                    
                    result = subprocess.run(
                        [
                            'inkscape',
                            '--export-type=png',
                            f'--export-filename={f_out.name}',
                            f'--export-width={width}',
                            f'--export-height={height}',
                            f_in.name
                        ],
                        capture_output=True
                    )
                    
                    if result.returncode == 0:
                        img = Image.open(f_out.name)
                        arr = np.array(img).astype(np.float32) / 255.0
                        return arr[:, :, :3]
        except (FileNotFoundError, Exception):
            pass
        
        # Return None if all rasterizers failed
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
        
        # Detect gaps (shouldn't happen with solid fills)
        # For now, just show color mismatch
        diff = np.abs(rasterized - quantized).mean(axis=-1)
        mismatch = diff > 10 / 255.0
        
        gap_mask[mismatch] = [1.0, 0, 0]  # Red for mismatch
        
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
        
        # Top-left: original
        comparison[:h, :w] = original
        
        # Top-right: quantized
        comparison[:h, w + 20:] = quantized
        
        # Bottom-left: SVG rasterized
        comparison[h + 20:, :w] = rasterized
        
        # Bottom-right: gap mask
        comparison[h + 20:, w + 20:] = gap_mask
        
        # Separators
        comparison[:h, w:w + 20] = 0.2
        comparison[h:h + 20, :] = 0.2
        
        self._save_stage_image(comparison, self.debug_dir / "comparison.png")
    
    def _visualize_quantization(
        self,
        label_map: np.ndarray,
        palette: np.ndarray
    ) -> np.ndarray:
        """Visualize quantized image."""
        return palette[label_map]
    
    def _visualize_regions(self, regions: List[PosterRegion]) -> np.ndarray:
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
        """Visualize boundaries."""
        h, w = self.ingest_result.image_srgb.shape[:2]
        img = Image.new('RGB', (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        np.random.seed(42)
        for edge in edges:
            points = (
                edge.smoothed_points if use_smoothed and edge.smoothed_points is not None
                else edge.points
            )
            if len(points) < 2:
                continue
            
            color = tuple((np.random.rand(3) * 255).astype(int))
            line_points = [(int(p[1]), int(p[0])) for p in points]
            draw.line(line_points, fill=color, width=2)
        
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
    smoothness: float = 0.5,
    debug_stages: Optional[Union[str, Path]] = None
) -> str:
    """
    Convenience function to process an image.
    
    Args:
        input_path: Path to input image
        output_path: Optional path for output SVG
        n_colors: Number of colors to quantize to
        smoothness: Boundary smoothing factor
        debug_stages: Optional directory for debug outputs
        
    Returns:
        SVG string
    """
    config = ApexConfig(
        n_colors=n_colors,
        spline_smoothness=smoothness
    )
    
    pipeline = PosterFirstPipeline(config)
    return pipeline.process(input_path, output_path, debug_stages)
