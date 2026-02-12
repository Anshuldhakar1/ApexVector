#!/usr/bin/env python3
"""
Poster Vectorization Prototype
Single-file validation of Felzenszwalb + shared boundaries + Gaussian smoothing
Pure Python / matplotlib only - no external SVG renderers
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Core dependencies (pure Python pip installable)
from skimage.segmentation import felzenszwalb
from skimage.measure import find_contours
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PosterConfig:
    """Configuration for poster vectorization."""
    n_colors: int = 12
    felzenszwalb_scale: float = 150.0
    felzenszwalb_sigma: float = 0.8
    felzenszwalb_min_size: int = 100
    gaussian_sigma: float = 1.5
    min_region_area_ratio: float = 0.001


# =============================================================================
# STAGE 1: COLOR QUANTIZATION
# =============================================================================

def quantize_colors(image: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize image to n_colors using K-means in LAB space."""
    h, w = image.shape[:2]
    lab = rgb2lab(image)
    pixels_lab = lab.reshape(-1, 3)
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=2048,
        n_init=3,
        max_iter=200
    )
    labels = kmeans.fit_predict(pixels_lab)
    centers_lab = kmeans.cluster_centers_
    
    centers_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    centers_rgb = np.clip(centers_rgb, 0, 1)
    palette = (centers_rgb * 255).astype(np.uint8)
    
    label_map = labels.reshape(h, w)
    return label_map, palette


# =============================================================================
# STAGE 2: SPATIAL COHERENCE (Felzenszwalb)
# =============================================================================

def apply_spatial_coherence(
    image: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    config: PosterConfig
) -> np.ndarray:
    """Refine quantization with Felzenszwalb segmentation on color-class image."""
    h, w = label_map.shape
    class_image = palette[label_map]
    
    segments = felzenszwalb(
        class_image,
        scale=config.felzenszwalb_scale,
        sigma=config.felzenszwalb_sigma,
        min_size=config.felzenszwalb_min_size,
        channel_axis=2
    )
    
    refined_labels = np.zeros_like(label_map)
    for seg_id in range(segments.max() + 1):
        seg_mask = segments == seg_id
        if not seg_mask.any():
            continue
        seg_labels = label_map[seg_mask]
        dominant = np.bincount(seg_labels, minlength=len(palette)).argmax()
        refined_labels[seg_mask] = dominant
    
    return refined_labels


# =============================================================================
# STAGE 3: REGION EXTRACTION
# =============================================================================

@dataclass
class Region:
    label_id: int
    mask: np.ndarray
    color: np.ndarray
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    _slice: Tuple[slice, slice] = None


def extract_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    config: PosterConfig
) -> List[Region]:
    """Extract connected components, merging small regions only within same color."""
    h, w = label_map.shape
    total_area = h * w
    min_area = int(total_area * config.min_region_area_ratio)
    
    regions = []
    
    for color_idx in range(len(palette)):
        color_mask = (label_map == color_idx)
        if not color_mask.any():
            continue
        
        labeled, num_features = ndimage.label(color_mask)
        if num_features == 0:
            continue
        
        slices = ndimage.find_objects(labeled)
        
        for comp_id in range(1, num_features + 1):
            sl = slices[comp_id - 1]
            if sl is None:
                continue
            
            comp_mask_local = (labeled[sl] == comp_id)
            area = int(comp_mask_local.sum())
            if area < 10:
                continue
            
            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[sl][comp_mask_local] = True
            
            y_coords, x_coords = np.where(full_mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))
            
            x_min, y_min = int(x_coords.min()), int(y_coords.min())
            x_max, y_max = int(x_coords.max()), int(y_coords.max())
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            
            region = Region(
                label_id=color_idx,
                mask=full_mask,
                color=palette[color_idx],
                area=area,
                centroid=centroid,
                bbox=bbox,
                _slice=sl
            )
            regions.append(region)
    
    regions = _merge_small_regions(regions, label_map, palette, min_area)
    return regions


def _merge_small_regions(
    regions: List[Region],
    label_map: np.ndarray,
    palette: np.ndarray,
    min_area: int
) -> List[Region]:
    """Merge small regions only to neighbors of the same color."""
    if not regions:
        return regions
    
    h, w = label_map.shape
    region_map = np.full((h, w), -1, dtype=np.int32)
    for i, r in enumerate(regions):
        region_map[r.mask] = i
    
    sorted_indices = sorted(range(len(regions)), key=lambda i: regions[i].area)
    merged = set()
    
    for idx in sorted_indices:
        if idx in merged:
            continue
        
        region = regions[idx]
        if region.area >= min_area:
            continue
        
        dilated = ndimage.binary_dilation(region.mask, iterations=1)
        neighbor_pixels = region_map[dilated & ~region.mask]
        neighbor_ids = set(neighbor_pixels[neighbor_pixels >= 0].tolist())
        
        same_color_neighbors = [
            nid for nid in neighbor_ids
            if nid not in merged and regions[nid].label_id == region.label_id
        ]
        
        if not same_color_neighbors:
            continue
        
        target_nid = max(same_color_neighbors, key=lambda nid: regions[nid].area)
        target = regions[target_nid]
        
        target.mask = target.mask | region.mask
        target.area = int(target.mask.sum())
        
        y_coords, x_coords = np.where(target.mask)
        target.centroid = (float(x_coords.mean()), float(y_coords.mean()))
        
        region_map[region.mask] = target_nid
        merged.add(idx)
    
    return [r for i, r in enumerate(regions) if i not in merged]


# =============================================================================
# STAGE 4: SHARED BOUNDARY EXTRACTION
# =============================================================================

@dataclass
class SharedEdge:
    points: np.ndarray
    region_a: int
    region_b: int
    length: float


def extract_shared_boundaries(
    regions: List[Region],
    label_map: np.ndarray,
    h: int,
    w: int
) -> List[SharedEdge]:
    """Extract boundaries shared between adjacent regions using marching squares."""
    if not regions:
        return []
    
    region_map = np.full((h, w), -1, dtype=np.int32)
    for i, r in enumerate(regions):
        region_map[r.mask] = i
    
    edges = []
    processed_pairs = set()
    
    # Horizontal adjacencies
    left = region_map[:, :-1]
    right = region_map[:, 1:]
    mask = (left != right) & (left >= 0) & (right >= 0)
    y_coords, x_coords = np.where(mask)
    
    for y, x in zip(y_coords, x_coords):
        r_a, r_b = int(left[y, x]), int(right[y, x])
        pair = tuple(sorted([r_a, r_b]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        
        edge = _extract_edge_between_regions(region_map, r_a, r_b, h, w)
        if edge:
            edges.append(edge)
    
    # Vertical adjacencies
    top = region_map[:-1, :]
    bottom = region_map[1:, :]
    mask = (top != bottom) & (top >= 0) & (bottom >= 0)
    y_coords, x_coords = np.where(mask)
    
    for y, x in zip(y_coords, x_coords):
        r_a, r_b = int(top[y, x]), int(bottom[y, x])
        pair = tuple(sorted([r_a, r_b]))
        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)
        
        edge = _extract_edge_between_regions(region_map, r_a, r_b, h, w)
        if edge:
            edges.append(edge)
    
    return edges


def _extract_edge_between_regions(
    region_map: np.ndarray,
    r_a: int,
    r_b: int,
    h: int,
    w: int
) -> Optional[SharedEdge]:
    """Extract the shared boundary contour between two specific regions."""
    mask = np.full((h, w), 0.5)
    mask[region_map == r_a] = 1.0
    mask[region_map == r_b] = 0.0
    
    if not ((mask == 1.0).any() and (mask == 0.0).any()):
        return None
    
    contours = find_contours(mask, level=0.5)
    if not contours:
        return None
    
    longest = max(contours, key=len)
    if len(longest) < 4:
        return None
    
    return SharedEdge(
        points=longest,
        region_a=r_a,
        region_b=r_b,
        length=float(len(longest))
    )


# =============================================================================
# STAGE 5: GAUSSIAN SMOOTHING
# =============================================================================

def smooth_shared_edges(edges: List[SharedEdge], sigma: float) -> List[SharedEdge]:
    """Apply Gaussian smoothing to each shared edge."""
    smoothed = []
    
    for edge in edges:
        points = edge.points.copy()
        if len(points) < 4:
            smoothed.append(edge)
            continue
        
        y = points[:, 0]
        x = points[:, 1]
        
        is_closed = np.allclose(points[0], points[-1])
        
        if is_closed and len(y) > 3:
            y_smooth = ndimage.gaussian_filter1d(
                np.concatenate([y, y]), sigma, mode='wrap'
            )[:len(y)]
            x_smooth = ndimage.gaussian_filter1d(
                np.concatenate([x, x]), sigma, mode='wrap'
            )[:len(x)]
        else:
            y_smooth = ndimage.gaussian_filter1d(y, sigma, mode='nearest')
            x_smooth = ndimage.gaussian_filter1d(x, sigma, mode='nearest')
        
        smoothed_points = np.column_stack([y_smooth, x_smooth])
        
        smoothed.append(SharedEdge(
            points=smoothed_points,
            region_a=edge.region_a,
            region_b=edge.region_b,
            length=edge.length
        ))
    
    return smoothed


# =============================================================================
# STAGE 6: REGION RECONSTRUCTION
# =============================================================================

def build_region_paths(regions: List[Region], edges: List[SharedEdge]) -> List[dict]:
    """Build closed paths for each region from smoothed shared edges."""
    if not regions or not edges:
        return []
    
    region_edges = {i: [] for i in range(len(regions))}
    
    for edge in edges:
        region_edges[edge.region_a].append((edge, False))
        region_edges[edge.region_b].append((edge, True))
    
    region_paths = []
    
    for r_idx, r_edges in region_edges.items():
        if not r_edges:
            continue
        
        region = regions[r_idx]
        paths = []
        
        for edge, reversed in r_edges:
            pts = edge.points.copy()
            if reversed:
                pts = pts[::-1]
            paths.append(pts)
        
        region_paths.append({
            'region_idx': r_idx,
            'color': region.color,
            'paths': paths
        })
    
    return region_paths


# =============================================================================
# STAGE 7: SVG GENERATION
# =============================================================================

def generate_svg(region_paths: List[dict], width: int, height: int) -> str:
    """Generate SVG with flat color fills from region paths."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    ]
    
    for rp in region_paths:
        color = rp['color']
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        path_datas = []
        for pts in rp['paths']:
            if len(pts) < 2:
                continue
            
            commands = [f"M {pts[0][1]:.2f},{pts[0][0]:.2f}"]
            for i in range(1, len(pts)):
                commands.append(f"L {pts[i][1]:.2f},{pts[i][0]:.2f}")
            commands.append("Z")
            path_datas.append(" ".join(commands))
        
        if not path_datas:
            continue
        
        d = " ".join(path_datas)
        svg_parts.append(
            f'<path d="{d}" fill="{hex_color}" '
            f'fill-rule="evenodd" stroke="none"/>'
        )
    
    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


# =============================================================================
# STAGE 8: MATPLOTLIB RASTERIZATION (Pure Python)
# =============================================================================

def render_svg_matplotlib(svg_string: str, width: int, height: int) -> np.ndarray:
    """
    Render SVG to RGBA numpy array using matplotlib.
    Pure Python - no external dependencies.
    """
    # Use exact figure size in inches with exact DPI to get precise pixel dimensions
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_alpha(0)
    
    # Parse paths from SVG
    path_pattern = r'<path d="([^"]*)" fill="([^"]*)"'
    
    for d, fill in re.findall(path_pattern, svg_string):
        coords = re.findall(r'[ML]\s+([-\d.]+),([-\d.]+)', d)
        if len(coords) < 3:
            continue
        
        points = np.array([[float(x), float(y)] for x, y in coords])
        
        polygon = Polygon(points, closed=True, facecolor=fill, 
                         edgecolor='none', linewidth=0)
        ax.add_patch(polygon)
    
    # Force exact dimensions in output
    fig.canvas.draw()
    
    # Get the exact buffer size and reshape precisely
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # The canvas may be slightly different size due to rounding; sample or resize
    actual_w, actual_h = fig.canvas.get_width_height()
    
    if actual_w != width or actual_h != height:
        # Reshape to what we got, then resize to target
        img = buf.reshape(actual_h, actual_w, 4)
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(img)
        img_pil = img_pil.resize((width, height), PILImage.Resampling.LANCZOS)
        img = np.array(img_pil)
    else:
        img = buf.reshape(height, width, 4)
    
    plt.close(fig)
    return img
# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_figure(
    original: np.ndarray,
    quantized: np.ndarray,
    label_map: np.ndarray,
    regions: List[Region],
    edges: List[SharedEdge],
    smoothed_edges: List[SharedEdge],
    svg_render: np.ndarray,
    config: PosterConfig
) -> plt.Figure:
    """Create comprehensive 8-panel visualization of all pipeline stages."""
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original)
    ax1.set_title("1. Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(quantized)
    ax2.set_title(f"2. Quantized ({config.n_colors} colors)")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(2, 4, 3)
    np.random.seed(42)
    label_colors = np.random.rand(label_map.max() + 1, 3)
    label_vis = label_colors[label_map]
    ax3.imshow(label_vis)
    ax3.set_title(f"3. Label Map ({len(np.unique(label_map))} labels)")
    ax3.axis('off')
    
    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(original, alpha=0.3)
    for r in regions[:50]:
        y, x = np.where(r.mask)
        if len(y) > 0:
            ax4.scatter(x[::10], y[::10], c=[r.color/255], s=1, alpha=0.5)
    ax4.set_title(f"4. Regions ({len(regions)} total)")
    ax4.axis('off')
    
    # Row 2
    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(quantized, alpha=0.3)
    for edge in edges[:20]:
        ax5.plot(edge.points[:, 1], edge.points[:, 0], 'r-', 
                alpha=0.5, linewidth=0.5)
    ax5.set_title(f"5. Raw Edges ({len(edges)} total)")
    ax5.axis('off')
    
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(quantized, alpha=0.3)
    for edge in smoothed_edges[:20]:
        ax6.plot(edge.points[:, 1], edge.points[:, 0], 'g-', 
                alpha=0.5, linewidth=1)
    ax6.set_title(f"6. Smoothed (σ={config.gaussian_sigma})")
    ax6.axis('off')
    
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(svg_render)
    ax7.set_title("7. SVG Render (matplotlib)")
    ax7.axis('off')
    
    # In create_comparison_figure, replace the gap analysis section:
    ax8 = fig.add_subplot(2, 4, 8)

    # Ensure same shape for comparison
    from PIL import Image as PILImage
    h, w = original.shape[:2]
    if svg_render.shape[:2] != (h, w):
        svg_pil = PILImage.fromarray(svg_render)
        svg_pil = svg_pil.resize((w, h), PILImage.Resampling.LANCZOS)
        svg_render_resized = np.array(svg_pil)
    else:
        svg_render_resized = svg_render

    render_gray = svg_render_resized[:, :, :3].mean(axis=2)
    orig_gray = original.mean(axis=2)

    # Gap: where render is nearly transparent/empty but original has content
    gaps = (render_gray < 15) & (orig_gray > 20)

    gap_vis = quantized.copy()
    gap_vis[gaps] = [255, 0, 255]
    ax8.imshow(gap_vis)

    gap_pct = gaps.sum() / gaps.size * 100
    ax8.set_title(f"8. Gap Analysis ({gap_pct:.3f}% gaps)")
    ax8.axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_image(image_path: str, output_path: str, config: Optional[PosterConfig] = None):
    """Run full poster vectorization pipeline and save comparison figure."""
    config = config or PosterConfig()
    
    print(f"Loading {image_path}...")
    image = np.array(Image.open(image_path).convert('RGB'))
    h, w = image.shape[:2]
    print(f"  Size: {w}x{h}")
    
    print(f"Quantizing to {config.n_colors} colors...")
    label_map, palette = quantize_colors(image, config.n_colors)
    quantized = palette[label_map]
    print(f"  Palette: {palette.shape}")
    
    print("Applying spatial coherence (Felzenszwalb)...")
    label_map = apply_spatial_coherence(image, label_map, palette, config)
    print(f"  Final labels: {len(np.unique(label_map))}")
    
    print("Extracting regions...")
    regions = extract_regions(label_map, palette, config)
    print(f"  Regions: {len(regions)}")
    
    print("Extracting shared boundaries...")
    edges = extract_shared_boundaries(regions, label_map, h, w)
    print(f"  Edges: {len(edges)}")
    
    print(f"Smoothing boundaries (sigma={config.gaussian_sigma})...")
    smoothed_edges = smooth_shared_edges(edges, config.gaussian_sigma)
    
    print("Building region paths...")
    region_paths = build_region_paths(regions, smoothed_edges)
    
    print("Generating SVG...")
    svg = generate_svg(region_paths, w, h)
    
    print("Rendering with matplotlib...")
    svg_render = render_svg_matplotlib(svg, w, h)
    
    print("Creating comparison figure...")
    fig = create_comparison_figure(
        image, quantized, label_map, regions,
        edges, smoothed_edges, svg_render,
        config
    )
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")
    
    svg_path = Path(output_path).with_suffix('.svg')
    svg_path.write_text(svg)
    print(f"Saved SVG: {svg_path}")
    
    metrics = {
        'image_size': (w, h),
        'n_colors': config.n_colors,
        'n_regions': len(regions),
        'n_edges': len(edges),
        'svg_size_bytes': len(svg.encode('utf-8')),
    }
    print("\nMetrics:", metrics)
    
    plt.close(fig)
    return metrics


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Poster vectorization prototype - pure Python'
    )
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', default='comparison.png',
                       help='Output comparison image')
    parser.add_argument('--colors', type=int, default=12,
                       help='Number of colors')
    parser.add_argument('--sigma', type=float, default=1.5,
                       help='Boundary smoothing sigma')
    parser.add_argument('--scale', type=float, default=150.0,
                       help='Felzenszwalb scale')
    
    args = parser.parse_args()
    
    config = PosterConfig(
        n_colors=args.colors,
        gaussian_sigma=args.sigma,
        felzenszwalb_scale=args.scale
    )
    
    process_image(args.input, args.output, config)