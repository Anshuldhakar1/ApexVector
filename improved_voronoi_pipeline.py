#!/usr/bin/env python3
"""
Improved Voronoi Pipeline - Better Segmentation
Key improvements:
1. Morphological pre-processing to merge fragments
2. Aggressive gap filling with watershed
3. Smart region merging based on color similarity
4. Overlap-based rendering to eliminate gaps
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from skimage.measure import find_contours, regionprops, label as skimage_label
from skimage.color import rgb2lab, lab2rgb
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    disk,
    remove_small_objects,
)
from skimage.segmentation import watershed, find_boundaries
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


@dataclass
class VoronoiConfig:
    """Configuration for improved Voronoi pipeline."""

    n_colors: int = 16  # Reduced for cleaner segmentation
    gaussian_sigma: float = 1.5  # Smoothing for boundaries
    mask_dilate: int = 2  # Pixels to dilate for overlap
    min_region_area: int = 30  # Lower to catch more details
    min_text_area: int = 5
    text_aspect_ratio: float = 2.0
    text_stroke_cv: float = 0.3
    circularity_threshold: float = 0.8
    pa_ratio_threshold: float = 10.0
    voronoi_max_gap: int = 3
    merge_threshold: float = 8.0  # Delta E threshold for merging
    morphological_closing: int = 1  # Iterations of morphological closing
    # Per-feature smoothing sigmas
    text_sigma: float = 0.3
    circle_sigma: float = 0.0
    general_sigma: float = 1.0


def quantize_colors(
    image: np.ndarray, n_colors: int, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize image colors using K-means in LAB space with fixed seed for determinism."""
    h, w = image.shape[:2]
    lab = rgb2lab(image)
    pixels_lab = lab.reshape(-1, 3)

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=random_state,
        batch_size=2048,
        n_init=3,
        max_iter=100,
    )
    labels = kmeans.fit_predict(pixels_lab)
    centers_lab = kmeans.cluster_centers_

    centers_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    centers_rgb = np.clip(centers_rgb, 0, 1)
    palette = (centers_rgb * 255).astype(np.uint8)

    label_map = labels.reshape(h, w)
    return label_map, palette


def morphological_cleanup(label_map: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply morphological closing to each color layer to merge nearby fragments.
    This reduces over-segmentation by connecting close regions of the same color.
    """
    result = label_map.copy()
    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    h, w = label_map.shape
    cleaned = np.zeros_like(label_map)

    for label in unique_labels:
        mask = result == label
        if not mask.any():
            continue

        # Apply morphological closing to connect nearby fragments
        if iterations > 0:
            mask = ndimage.binary_closing(mask, iterations=iterations)

        cleaned[mask] = label

    return cleaned


def watershed_gap_fill(
    label_map: np.ndarray, image: np.ndarray, max_gap_size: int = 3
) -> np.ndarray:
    """
    Fill gaps using watershed segmentation on gradient magnitude.
    More robust than simple Voronoi for complex boundaries.
    """
    h, w = label_map.shape
    result = label_map.copy()

    # Find gaps (unassigned or boundary pixels)
    unique_labels = np.unique(label_map)
    valid_labels = unique_labels[unique_labels >= 0]

    if len(valid_labels) == 0:
        return result

    # Create markers from existing regions
    markers = np.zeros((h, w), dtype=int)
    for idx, label in enumerate(valid_labels):
        markers[label_map == label] = idx + 1

    # Compute gradient from image edges
    from skimage.filters import sobel

    gray = image.mean(axis=2) if len(image.shape) == 3 else image
    gradient = sobel(gray)

    # Apply watershed
    segmented = watershed(gradient, markers, mask=None)

    # Map back to original labels
    for idx, label in enumerate(valid_labels):
        result[segmented == (idx + 1)] = label

    return result


def merge_similar_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    image: np.ndarray,
    threshold: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge adjacent regions with similar colors (Delta E < threshold).
    Reduces over-segmentation from color quantization.
    """
    h, w = label_map.shape
    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    if len(unique_labels) <= 1:
        return label_map, palette

    # Build adjacency graph
    adjacency = defaultdict(set)
    boundaries = find_boundaries(label_map, mode="thick")

    for i in range(h):
        for j in range(w):
            if not boundaries[i, j]:
                continue

            # Check neighbors
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    l1, l2 = label_map[i, j], label_map[ni, nj]
                    if l1 != l2 and l1 >= 0 and l2 >= 0:
                        adjacency[l1].add(l2)
                        adjacency[l2].add(l1)

    # Convert palette to LAB for perceptual comparison
    from skimage.color import rgb2lab

    palette_lab = rgb2lab(palette.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    # Find regions to merge
    merged = set()
    merge_map = {label: label for label in unique_labels}

    for label in unique_labels:
        if label in merged:
            continue

        color_lab = palette_lab[label]

        for neighbor in adjacency[label]:
            if neighbor in merged or neighbor <= label:
                continue

            neighbor_lab = palette_lab[neighbor]
            delta_e = np.sqrt(np.sum((color_lab - neighbor_lab) ** 2))

            if delta_e < threshold:
                # Merge neighbor into label
                merge_map[neighbor] = label
                merged.add(neighbor)

    # Apply merges
    new_label_map = label_map.copy()
    for old_label, new_label in merge_map.items():
        new_label_map[label_map == old_label] = new_label

    # Relabel to contiguous indices
    unique_remaining = np.unique(new_label_map)
    unique_remaining = unique_remaining[unique_remaining >= 0]

    relabel_map = {old: new for new, old in enumerate(unique_remaining)}
    final_label_map = np.zeros_like(new_label_map)
    for old, new in relabel_map.items():
        final_label_map[new_label_map == old] = new

    # Update palette
    new_palette = palette[unique_remaining]

    return final_label_map, new_palette


@dataclass
class Boundary:
    """Represents a shared boundary between two regions."""

    label_a: int
    label_b: int
    points: np.ndarray
    feature_type: str = "general"


def extract_shared_boundaries(label_map: np.ndarray) -> List[Boundary]:
    """Extract boundaries as the exact interface between regions."""
    h, w = label_map.shape
    boundaries = []
    processed_pairs: Set[Tuple[int, int]] = set()

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    for i, label_a in enumerate(unique_labels):
        for label_b in unique_labels[i + 1 :]:
            pair = (int(label_a), int(label_b))

            if pair in processed_pairs:
                continue

            mask = np.zeros((h, w), dtype=float)
            mask[label_map == label_a] = 1.0
            mask[label_map == label_b] = 0.0

            a_mask = label_map == label_a
            b_mask = label_map == label_b

            a_dilated = ndimage.binary_dilation(a_mask, iterations=1)
            if not np.any(a_dilated & b_mask):
                continue

            contours = find_contours(mask, level=0.5)

            if contours:
                longest = max(contours, key=len)
                if len(longest) >= 4:
                    boundary = Boundary(
                        label_a=pair[0], label_b=pair[1], points=longest
                    )
                    boundaries.append(boundary)
                    processed_pairs.add(pair)

    return boundaries


def classify_region_type(
    region_mask: np.ndarray, original_image: Optional[np.ndarray] = None
) -> str:
    """Detect if region contains text, circles, or general shapes."""
    props = regionprops(region_mask.astype(int))
    if not props:
        return "unknown"

    prop = props[0]
    area = max(prop.area, 1)
    perimeter = max(prop.perimeter, 1)
    pa_ratio = (perimeter**2) / (4 * np.pi * area)
    circularity = (4 * np.pi * area) / (perimeter**2)
    euler = prop.euler_number

    if pa_ratio > 10:
        return "text_thin"
    elif circularity > 0.8 and euler <= 0:
        return "circular"
    else:
        return "general"


def fit_circle_to_points(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Fit a circle to points using least squares."""
    if len(points) < 3:
        center = np.mean(points, axis=0) if len(points) > 0 else np.array([0, 0])
        return (center[1], center[0]), 1.0

    y_coords = points[:, 0]
    x_coords = points[:, 1]

    A = np.column_stack([x_coords, y_coords, np.ones(len(x_coords))])
    B = x_coords**2 + y_coords**2

    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        a = coeffs[0] / 2
        b = coeffs[1] / 2
        r = np.sqrt(coeffs[2] + a**2 + b**2)
        return (a, b), r
    except:
        center = np.mean(points, axis=0)
        return (center[1], center[0]), np.std(points) + 1


def generate_circle_contour(
    center: Tuple[float, float], radius: float, n_points: int = 100
) -> np.ndarray:
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([y, x])


def smooth_boundary(
    points: np.ndarray, feature_type: str, config: VoronoiConfig
) -> np.ndarray:
    """Apply appropriate smoothing based on feature type."""
    if len(points) < 4:
        return points

    if feature_type == "text_thin":
        if config.text_sigma > 0:
            from scipy.ndimage import gaussian_filter1d

            smoothed = points.copy().T
            smoothed[0] = gaussian_filter1d(
                smoothed[0], sigma=config.text_sigma, mode="wrap"
            )
            smoothed[1] = gaussian_filter1d(
                smoothed[1], sigma=config.text_sigma, mode="wrap"
            )
            return smoothed.T
        return points

    elif feature_type == "circular":
        center, radius = fit_circle_to_points(points)
        return generate_circle_contour(center, radius, n_points=len(points))

    else:
        if config.general_sigma > 0:
            from scipy.ndimage import gaussian_filter1d

            smoothed = points.copy().T
            smoothed[0] = gaussian_filter1d(
                smoothed[0], sigma=config.general_sigma, mode="wrap"
            )
            smoothed[1] = gaussian_filter1d(
                smoothed[1], sigma=config.general_sigma, mode="wrap"
            )
            return smoothed.T
        return points


@dataclass
class VoronoiRegion:
    """Region with Voronoi-based shared boundaries."""

    label_id: int
    color: np.ndarray
    boundaries: List[Boundary]
    area: int
    centroid: Tuple[float, float]
    mask: np.ndarray = field(repr=False)
    feature_type: str = "general"


def extract_voronoi_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    feature_types: Dict[int, str],
    config: VoronoiConfig,
) -> List[VoronoiRegion]:
    """Extract regions with pre-processing to reduce fragmentation."""
    h, w = label_map.shape
    regions = []
    region_counter = 0

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    for label in unique_labels:
        color_mask = label_map == label
        if not color_mask.any():
            continue

        # Apply morphological closing to merge fragments
        merged_mask = ndimage.binary_closing(
            color_mask, iterations=config.morphological_closing
        )

        # Remove small isolated noise
        merged_mask = remove_small_objects(
            merged_mask, min_size=config.min_region_area // 3
        )

        # Label connected components
        labeled, num_features = ndimage.label(merged_mask)

        for comp_id in range(1, num_features + 1):
            mask = labeled == comp_id
            area = int(mask.sum())

            feature_type = feature_types.get(label, "general")
            min_area = (
                config.min_text_area
                if feature_type == "text_thin"
                else config.min_region_area
            )

            if area < min_area:
                continue

            y_coords, x_coords = np.where(mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))

            region = VoronoiRegion(
                label_id=region_counter,
                color=palette[label],
                boundaries=[],
                area=area,
                centroid=centroid,
                feature_type=feature_type,
                mask=mask,
            )
            regions.append(region)
            region_counter += 1

    return regions


def mask_to_smooth_contour(
    mask: np.ndarray, sigma: float = 1.0
) -> Optional[np.ndarray]:
    """Extract smooth contour from binary mask with distance transform."""
    if not mask.any():
        return None

    area = mask.sum()
    if area < 100:
        contours = find_contours(mask.astype(float), level=0.5)
        if contours:
            longest = max(contours, key=len)
            if len(longest) >= 3:
                return longest
        return None

    padded = np.pad(mask, 1, mode="constant")
    dist = ndimage.distance_transform_edt(padded) - ndimage.distance_transform_edt(
        ~padded
    )

    if sigma > 0:
        smooth_dist = ndimage.gaussian_filter(dist, sigma=sigma)
    else:
        smooth_dist = dist

    contours = find_contours(smooth_dist, level=0)

    if not contours:
        contours = find_contours(smooth_dist, level=0.5)

    if not contours and sigma > 0:
        contours = find_contours(dist, level=0)

    if not contours:
        contours = find_contours(mask.astype(float), level=0.5)

    if not contours:
        return None

    longest = max(contours, key=len)
    longest = longest - 1

    h, w = mask.shape
    longest = np.clip(longest, [0, 0], [h - 1, w - 1])

    if len(longest) < 3:
        return None

    return longest


def generate_svg_voronoi(
    regions: List[VoronoiRegion],
    boundaries: List[Boundary],
    feature_types: Dict[int, str],
    width: int,
    height: int,
    config: VoronoiConfig,
) -> str:
    """Generate SVG with dilated masks for gap-free rendering."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" '
        f'fill-rule="evenodd">'
    ]

    # Sort regions by area (largest first) so smaller regions paint on top
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

    for region in regions_sorted:
        feature_type = feature_types.get(region.label_id, "general")

        if feature_type == "text_thin":
            sigma = config.text_sigma
        elif feature_type == "circular":
            sigma = config.circle_sigma
        else:
            sigma = config.general_sigma

        # Multi-stage dilation for guaranteed overlap:
        # 1. Dilate to create overlap with neighbors
        # 2. Closing to smooth boundary
        # 3. Additional dilation for safety margin
        dilated_mask = region.mask.copy()

        # Primary dilation for overlap
        if config.mask_dilate > 0:
            dilated_mask = ndimage.binary_dilation(
                dilated_mask, iterations=config.mask_dilate
            )

        # Morphological closing to fill small holes and smooth
        dilated_mask = ndimage.binary_closing(dilated_mask, iterations=1)

        # Safety margin dilation for edge regions
        boundary_touching = (
            dilated_mask[0, :].any()
            or dilated_mask[-1, :].any()
            or dilated_mask[:, 0].any()
            or dilated_mask[:, -1].any()
        )
        if boundary_touching:
            dilated_mask = ndimage.binary_dilation(dilated_mask, iterations=1)

        contour = mask_to_smooth_contour(dilated_mask, sigma=sigma)

        if contour is None:
            continue

        contour = smooth_boundary(contour, feature_type, config)

        # Ensure contour is closed
        if len(contour) > 0 and not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])

        coords = " ".join([f"{pt[1]:.2f},{pt[0]:.2f}" for pt in contour])
        path_d = f"M {coords} Z"

        color = region.color
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        svg_parts.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_svg_matplotlib(svg_string: str, width: int, height: int) -> np.ndarray:
    """Render SVG to RGBA using matplotlib."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)

    path_pattern = r'<path d="([^"]*)" fill="([^"]*)"'

    for d, fill in re.findall(path_pattern, svg_string):
        coords = re.findall(r"([-\d.]+),([-\d.]+)", d)
        if len(coords) < 3:
            continue

        points = np.array([[float(x), float(y)] for x, y in coords])

        polygon = Polygon(
            points, closed=True, facecolor=fill, edgecolor="none", linewidth=0
        )
        ax.add_patch(polygon)

    fig.canvas.draw()

    width_px = int(fig.get_figwidth() * fig.dpi)
    height_px = int(fig.get_figheight() * fig.dpi)

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(height_px, width_px, 4)

    if width_px != width or height_px != height:
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
        img = np.array(img_pil)

    plt.close(fig)
    return img


def create_comparison_figure(
    original: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    regions: List[VoronoiRegion],
    boundaries: List[Boundary],
    svg_render: np.ndarray,
    config: VoronoiConfig,
) -> plt.Figure:
    """Create 6-panel comparison figure."""
    h, w = original.shape[:2]
    quantized = palette[label_map]

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(original)
    ax1.set_title("1. Original")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(quantized)
    ax2.set_title(f"2. Quantized ({len(palette)} colors)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 3, 3)
    np.random.seed(42)
    label_colors = np.random.rand(len(palette), 3)
    label_vis = label_colors[label_map]
    ax3.imshow(label_vis)
    ax3.set_title(f"3. Label Map ({len(np.unique(label_map))} unique)")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 3, 4)
    overlay = original.copy().astype(float)
    for region in regions:
        mask = region.mask
        color = region.color / 255.0
        overlay[mask] = overlay[mask] * 0.3 + color * 0.7
    ax4.imshow(overlay.astype(np.uint8))
    ax4.set_title(f"4. Voronoi Regions ({len(regions)} total)")
    ax4.axis("off")

    for boundary in boundaries[:50]:
        points = boundary.points
        ax4.plot(points[:, 1], points[:, 0], "r-", linewidth=0.5, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(svg_render)
    ax5.set_title("5. SVG Render")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    if svg_render.shape[:2] != (h, w):
        svg_pil = Image.fromarray(svg_render)
        svg_pil = svg_pil.resize((w, h), Image.Resampling.LANCZOS)
        svg_resized = np.array(svg_pil)
    else:
        svg_resized = svg_render

    render_gray = svg_resized[:, :, :3].mean(axis=2)
    orig_gray = original.mean(axis=2)

    gaps = (render_gray < 20) & (orig_gray > 30)

    gap_vis = quantized.copy()
    gap_vis[gaps] = [255, 0, 255]
    ax6.imshow(gap_vis)

    gap_pct = gaps.sum() / gaps.size * 100
    ax6.set_title(f"6. Gap Analysis ({gap_pct:.3f}% gaps)")
    ax6.axis("off")

    plt.tight_layout()
    return fig


def process_image_voronoi(
    image_path: str, output_path: str, config: Optional[VoronoiConfig] = None
):
    """Run improved Voronoi-based poster vectorization pipeline."""
    config = config or VoronoiConfig()

    print(f"Loading {image_path}...")
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"  Size: {w}x{h}")

    print(f"\nPhase 0: Color Quantization")
    print(f"  Quantizing to {config.n_colors} colors...")
    label_map, palette = quantize_colors(image, config.n_colors)
    print(f"  Initial palette: {len(palette)} colors")
    print(f"  Initial unique labels: {len(np.unique(label_map))}")

    print(f"\nPhase 1: Morphological Cleanup")
    label_map_clean = morphological_cleanup(
        label_map, iterations=config.morphological_closing
    )
    print(f"  After cleanup: {len(np.unique(label_map_clean))} unique labels")

    print(f"\nPhase 2: Merge Similar Regions")
    label_map_merged, palette_merged = merge_similar_regions(
        label_map_clean, palette, image, threshold=config.merge_threshold
    )
    print(
        f"  After merging: {len(palette_merged)} colors, {len(np.unique(label_map_merged))} labels"
    )

    print(f"\nPhase 3: Watershed Gap Filling")
    label_map_filled = watershed_gap_fill(
        label_map_merged, image, config.voronoi_max_gap
    )
    print(f"  After gap filling: {len(np.unique(label_map_filled))} regions")

    print(f"\nPhase 4: Shared Boundary Extraction")
    boundaries = extract_shared_boundaries(label_map_filled)
    print(f"  Extracted {len(boundaries)} shared boundaries")

    print(f"\nPhase 5: Feature Type Detection")
    feature_types = {}
    unique_labels = np.unique(label_map_filled)
    for label in unique_labels:
        if label < 0:
            continue
        mask = label_map_filled == label
        feature_types[label] = classify_region_type(mask, image)

    type_counts = {}
    for ft in feature_types.values():
        type_counts[ft] = type_counts.get(ft, 0) + 1
    print(f"  Feature types: {type_counts}")

    print(f"\nPhase 6: Region Extraction")
    regions = extract_voronoi_regions(
        label_map_filled, palette_merged, feature_types, config
    )
    print(f"  Extracted {len(regions)} regions")

    print(f"\nPhase 7: SVG Generation with Overlap")
    svg = generate_svg_voronoi(regions, boundaries, feature_types, w, h, config)

    print(f"Rendering with matplotlib...")
    svg_render = render_svg_matplotlib(svg, w, h)

    print(f"Creating comparison figure...")
    fig = create_comparison_figure(
        image, label_map_filled, palette_merged, regions, boundaries, svg_render, config
    )

    fig.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Saved: {output_path}")

    svg_path = Path(output_path).with_suffix(".svg")
    svg_path.write_text(svg)
    print(f"Saved SVG: {svg_path}")

    # Calculate final metrics
    render_gray = svg_render[:, :, :3].mean(axis=2)
    orig_gray = image.mean(axis=2)
    gaps = (render_gray < 20) & (orig_gray > 30)
    gap_pct = gaps.sum() / gaps.size * 100

    metrics = {
        "image_size": (w, h),
        "n_colors": len(palette_merged),
        "n_regions": len(regions),
        "n_boundaries": len(boundaries),
        "feature_types": type_counts,
        "gap_percentage": gap_pct,
        "svg_bytes": len(svg.encode("utf-8")),
    }
    print("\nMetrics:", metrics)

    plt.close(fig)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Improved Voronoi poster vectorization with better segmentation"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default="improved_voronoi_comparison.png",
        help="Output comparison image",
    )
    parser.add_argument("--colors", type=int, default=16, help="Number of colors")
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=8.0,
        help="Delta E threshold for merging",
    )
    parser.add_argument(
        "--min-region-area", type=int, default=30, help="Minimum region area"
    )
    parser.add_argument(
        "--mask-dilate", type=int, default=2, help="Mask dilation iterations"
    )
    parser.add_argument(
        "--morphological-closing",
        type=int,
        default=1,
        help="Morphological closing iterations",
    )

    args = parser.parse_args()

    config = VoronoiConfig(
        n_colors=args.colors,
        merge_threshold=args.merge_threshold,
        min_region_area=args.min_region_area,
        mask_dilate=args.mask_dilate,
        morphological_closing=args.morphological_closing,
    )

    process_image_voronoi(args.input, args.output, config)
