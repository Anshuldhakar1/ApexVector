#!/usr/bin/env python3
"""
Poster Vectorization Pipeline - Feature-Aware Version
Based on corrected.py with added feature detection and per-type smoothing
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from skimage.measure import find_contours, regionprops
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class FeatureAwareConfig:
    n_colors: int = 24
    gaussian_sigma: float = 0.0  # No smoothing by default (like corrected.py)
    mask_dilate: int = 2  # 2px dilation (like corrected.py)
    min_region_area: int = 50
    min_text_area: int = 10  # Lower threshold for text
    text_aspect_ratio: float = 2.0
    text_stroke_cv: float = 0.3
    circularity_threshold: float = 0.8
    pa_ratio_threshold: float = 10.0
    # Per-feature smoothing sigmas
    text_sigma: float = 0.5
    circle_sigma: float = 0.0
    general_sigma: float = 1.5


# =============================================================================
# STAGE 1: COLOR QUANTIZATION (Preserved from corrected.py)
# =============================================================================


def quantize_colors(image: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize image colors using K-means in LAB space."""
    h, w = image.shape[:2]
    lab = rgb2lab(image)
    pixels_lab = lab.reshape(-1, 3)

    kmeans = MiniBatchKMeans(
        n_clusters=n_colors, random_state=42, batch_size=2048, n_init=3, max_iter=100
    )
    labels = kmeans.fit_predict(pixels_lab)
    centers_lab = kmeans.cluster_centers_

    centers_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    centers_rgb = np.clip(centers_rgb, 0, 1)
    palette = (centers_rgb * 255).astype(np.uint8)

    label_map = labels.reshape(h, w)
    return label_map, palette


# =============================================================================
# FEATURE TYPE DETECTION
# =============================================================================


def classify_region_type(mask: np.ndarray) -> str:
    """
    Classify region as 'text_thin', 'circular', or 'general'.
    """
    props = regionprops(mask.astype(int))
    if not props:
        return "general"

    prop = props[0]
    area = max(prop.area, 1)
    perimeter = max(prop.perimeter, 1)

    # Perimeter/area ratio for thinness
    pa_ratio = (perimeter**2) / (4 * np.pi * area)

    # Circularity: 1 = perfect circle
    circularity = (4 * np.pi * area) / (perimeter**2)

    # Euler number: holes
    euler = prop.euler_number

    if pa_ratio > 10:  # Thin, elongated structure
        return "text_thin"
    elif circularity > 0.8 and euler <= 0:  # Solid or single hole, very circular
        return "circular"
    else:
        return "general"


def detect_text_regions(regions: List, config: FeatureAwareConfig) -> set:
    """
    Identify text-like regions based on aspect ratio and stroke width consistency.
    Returns set of region indices that are text-like.
    """
    text_labels = set()

    for i, region in enumerate(regions):
        mask = region.mask if hasattr(region, "mask") else region.original_mask
        props = regionprops(mask.astype(int))
        if not props:
            continue

        prop = props[0]

        # Aspect ratio check
        minr, minc, maxr, maxc = prop.bbox
        height = maxr - minr
        width = maxc - minc
        aspect = width / max(height, 1)

        # Stroke width consistency via distance transform
        dist = ndimage.distance_transform_edt(mask)
        stroke_widths = dist[mask]
        if len(stroke_widths) > 0 and np.mean(stroke_widths) > 0:
            width_variance = np.std(stroke_widths) / np.mean(stroke_widths)
        else:
            width_variance = 1.0

        # Text criteria: wide aspect ratio, consistent stroke width
        if aspect > config.text_aspect_ratio and width_variance < config.text_stroke_cv:
            text_labels.add(i)

    return text_labels


# =============================================================================
# REGION EXTRACTION (Based on corrected.py with feature detection)
# =============================================================================


@dataclass
class FeatureAwareRegion:
    """Region with feature type information."""

    label_id: int  # Index in palette
    color: np.ndarray  # RGB from palette
    smoothed_mask: np.ndarray  # Dilated mask (like corrected.py)
    original_mask: np.ndarray  # Original connected component
    area: int
    centroid: Tuple[float, float]
    feature_type: str = "general"


def extract_and_smooth_regions(
    label_map: np.ndarray, palette: np.ndarray, config: FeatureAwareConfig
) -> List[FeatureAwareRegion]:
    """
    Extract connected components per color, apply feature-aware smoothing.
    Based on corrected.py extract_and_smooth_regions.
    """
    h, w = label_map.shape
    regions = []

    for color_idx in range(len(palette)):
        color_mask = label_map == color_idx
        if not color_mask.any():
            continue

        # Label connected components of this color
        labeled, num_features = ndimage.label(color_mask)

        for comp_id in range(1, num_features + 1):
            mask = labeled == comp_id
            area = int(mask.sum())

            # Adaptive area threshold
            feature_type = classify_region_type(mask)
            min_area = (
                config.min_text_area
                if feature_type == "text_thin"
                else config.min_region_area
            )

            if area < min_area:
                continue

            # Compute centroid
            y_coords, x_coords = np.where(mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))

            # === BOUNDARY SMOOTHING (like corrected.py) ===
            # Use distance transform + Gaussian for smooth boundaries
            dist_inside = ndimage.distance_transform_edt(mask)
            dist_outside = ndimage.distance_transform_edt(~mask)
            signed_dist = dist_inside - dist_outside

            # Smooth the distance field
            smooth_dist = ndimage.gaussian_filter(
                signed_dist, sigma=config.gaussian_sigma
            )

            # Threshold back to mask
            smooth_mask = smooth_dist > 0

            # Dilate to ensure overlap with neighbors
            if config.mask_dilate > 0:
                smooth_mask = ndimage.binary_dilation(
                    smooth_mask, iterations=config.mask_dilate
                )

            region = FeatureAwareRegion(
                label_id=color_idx,
                color=palette[color_idx],
                smoothed_mask=smooth_mask,
                original_mask=mask,
                area=area,
                centroid=centroid,
                feature_type=feature_type,
            )
            regions.append(region)

    # Sort by area (largest first) so smaller regions paint on top
    regions.sort(key=lambda r: r.area, reverse=True)

    return regions


# =============================================================================
# CONTOUR EXTRACTION WITH FEATURE-AWARE SMOOTHING
# =============================================================================


def smooth_boundary_points(
    points: np.ndarray, feature_type: str, config: FeatureAwareConfig
) -> np.ndarray:
    """
    Apply appropriate smoothing based on feature type.
    """
    if len(points) < 4:
        return points

    if feature_type == "text_thin":
        # Minimal smoothing for text
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
        # Fit circle and return circular contour
        if len(points) < 3:
            return points

        y_coords = points[:, 0]
        x_coords = points[:, 1]

        # Algebraic circle fitting (Kasa method)
        A = np.column_stack([x_coords, y_coords, np.ones(len(x_coords))])
        B = x_coords**2 + y_coords**2

        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            a = coeffs[0] / 2
            b = coeffs[1] / 2
            r = np.sqrt(coeffs[2] + a**2 + b**2)

            # Generate circle points
            angles = np.linspace(0, 2 * np.pi, len(points), endpoint=False)
            x = a + r * np.cos(angles)
            y = b + r * np.sin(angles)
            return np.column_stack([y, x])
        except:
            return points

    else:  # general
        # Standard Gaussian smoothing
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


def mask_to_smooth_contour(
    mask: np.ndarray, sigma: float = 1.0
) -> Optional[np.ndarray]:
    """
    Extract smooth contour from binary mask.
    Same as corrected.py.
    """
    if not mask.any():
        return None

    # Pad to avoid edge issues
    padded = np.pad(mask, 1, mode="constant")

    # Distance transform for sub-pixel boundary
    dist = ndimage.distance_transform_edt(padded) - ndimage.distance_transform_edt(
        ~padded
    )
    smooth_dist = ndimage.gaussian_filter(dist, sigma=sigma)

    # Marching squares at zero level
    contours = find_contours(smooth_dist, level=0)

    if not contours:
        return None

    # Use longest contour, remove padding offset
    longest = max(contours, key=len)
    longest = longest - 1  # Remove padding

    # Clip to image bounds
    h, w = mask.shape
    longest = np.clip(longest, [0, 0], [h - 1, w - 1])

    if len(longest) < 4:
        return None

    return longest


# =============================================================================
# SVG GENERATION
# =============================================================================


def generate_svg(
    regions: List[FeatureAwareRegion],
    width: int,
    height: int,
    config: FeatureAwareConfig,
) -> str:
    """Generate SVG with feature-aware smoothing."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    ]

    for region in regions:
        # Extract contour from smoothed mask (same as corrected.py)
        contour = mask_to_smooth_contour(region.smoothed_mask, sigma=1.0)
        if contour is None:
            continue

        # Apply feature-aware smoothing
        contour = smooth_boundary_points(contour, region.feature_type, config)

        color = region.color
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        # Build path data (note: contour is [row, col] = [y, x])
        coords = " ".join([f"{pt[1]:.2f},{pt[0]:.2f}" for pt in contour])
        path_d = f"M {coords} Z"

        svg_parts.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# =============================================================================
# RASTERIZATION (Preserved from corrected.py)
# =============================================================================


def render_svg_matplotlib(svg_string: str, width: int, height: int) -> np.ndarray:
    """Render SVG to RGBA using matplotlib."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_alpha(0)

    # Parse and render paths
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


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_comparison_figure(
    original: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    regions: List[FeatureAwareRegion],
    svg_render: np.ndarray,
    config: FeatureAwareConfig,
) -> plt.Figure:
    """Create 6-panel comparison figure."""
    h, w = original.shape[:2]
    quantized = palette[label_map]

    fig = plt.figure(figsize=(18, 12))

    # Row 1
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

    # Row 2
    ax4 = fig.add_subplot(2, 3, 4)
    overlay = original.copy().astype(float)
    for region in regions:
        mask = region.smoothed_mask
        color = region.color / 255.0
        overlay[mask] = overlay[mask] * 0.3 + color * 0.7
    ax4.imshow(overlay.astype(np.uint8))
    ax4.set_title(f"4. Regions ({len(regions)} total)")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(svg_render)
    ax5.set_title("5. SVG Render")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 3, 6)
    # Gap analysis
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


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def process_image(
    image_path: str, output_path: str, config: Optional[FeatureAwareConfig] = None
):
    """Run feature-aware poster vectorization pipeline."""
    config = config or FeatureAwareConfig()

    print(f"Loading {image_path}...")
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"  Size: {w}x{h}")

    print(f"\nPhase 1: Color Quantization")
    print(f"  Quantizing to {config.n_colors} colors...")
    label_map, palette = quantize_colors(image, config.n_colors)
    print(f"  Palette: {len(palette)} colors")
    print(f"  Label map unique: {len(np.unique(label_map))}")

    print(f"\nPhase 2: Extract and Smooth Regions")
    print(f"  Using sigma={config.gaussian_sigma}, dilate={config.mask_dilate}")
    regions = extract_and_smooth_regions(label_map, palette, config)
    print(f"  Regions: {len(regions)}")

    # Count feature types
    type_counts = defaultdict(int)
    for r in regions:
        type_counts[r.feature_type] += 1
    print(f"  Feature types: {dict(type_counts)}")

    print(f"\nPhase 3: Generate SVG")
    svg = generate_svg(regions, w, h, config)

    print(f"Rendering with matplotlib...")
    svg_render = render_svg_matplotlib(svg, w, h)

    print(f"Creating comparison figure...")
    fig = create_comparison_figure(
        image, label_map, palette, regions, svg_render, config
    )

    fig.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"Saved: {output_path}")

    svg_path = Path(output_path).with_suffix(".svg")
    svg_path.write_text(svg)
    print(f"Saved SVG: {svg_path}")

    # Metrics
    metrics = {
        "image_size": (w, h),
        "n_colors": len(palette),
        "n_regions": len(regions),
        "feature_types": dict(type_counts),
        "svg_bytes": len(svg.encode("utf-8")),
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
        description="Poster vectorization - feature-aware smoothing"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default="feature_aware_comparison.png",
        help="Output comparison image",
    )
    parser.add_argument("--colors", type=int, default=24, help="Number of colors")
    parser.add_argument(
        "--sigma", type=float, default=0.0, help="Boundary smoothing sigma"
    )
    parser.add_argument(
        "--dilate", type=int, default=2, help="Mask dilation for overlap (pixels)"
    )
    parser.add_argument(
        "--text-sigma", type=float, default=0.5, help="Smoothing for text boundaries"
    )
    parser.add_argument(
        "--general-sigma", type=float, default=1.5, help="Smoothing for general shapes"
    )

    args = parser.parse_args()

    config = FeatureAwareConfig(
        n_colors=args.colors,
        gaussian_sigma=args.sigma,
        mask_dilate=args.dilate,
        text_sigma=args.text_sigma,
        general_sigma=args.general_sigma,
    )

    process_image(args.input, args.output, config)
