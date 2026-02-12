#!/usr/bin/env python3
"""
Poster Vectorization Prototype - Corrected Version
Overlapping regions approach: simpler, no gaps, robust
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

from skimage.measure import find_contours
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PosterConfig:
    n_colors: int = 12
    gaussian_sigma: float = 2.0  # Smoothing for region boundaries
    mask_dilate: int = 2  # Pixels to dilate for overlap
    min_region_area: int = 50  # Minimum region size in pixels


# =============================================================================
# STAGE 1: COLOR QUANTIZATION (UNCHANGED)
# =============================================================================


def quantize_colors(image: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
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
# STAGE 2: REGION EXTRACTION (SIMPLIFIED - NO FELZENSZWALB)
# =============================================================================


@dataclass
class Region:
    label_id: int  # Index in palette
    color: np.ndarray  # RGB from palette
    smoothed_mask: np.ndarray  # Boolean mask after smoothing+dilation
    original_mask: np.ndarray  # Original connected component
    area: int
    centroid: Tuple[float, float]


def extract_and_smooth_regions(
    label_map: np.ndarray, palette: np.ndarray, config: PosterConfig
) -> List[Region]:
    """
    Extract connected components per color, smooth boundaries, dilate for overlap.
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

            if area < config.min_region_area:
                continue

            # Compute centroid
            y_coords, x_coords = np.where(mask)
            centroid = (float(x_coords.mean()), float(y_coords.mean()))

            # === BOUNDARY SMOOTHING ===
            # Use distance transform + Gaussian for smooth boundaries
            # Positive inside, negative outside
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

            region = Region(
                label_id=color_idx,
                color=palette[color_idx],
                smoothed_mask=smooth_mask,
                original_mask=mask,
                area=area,
                centroid=centroid,
            )
            regions.append(region)

    # Sort by area (largest first) so smaller regions paint on top
    regions.sort(key=lambda r: r.area, reverse=True)

    return regions


# =============================================================================
# STAGE 3: CONTOUR EXTRACTION FOR SVG
# =============================================================================


def mask_to_smooth_contour(
    mask: np.ndarray, sigma: float = 1.0
) -> Optional[np.ndarray]:
    """
    Extract smooth contour from binary mask.
    Uses distance transform + marching squares for sub-pixel accuracy.
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
# STAGE 4: SVG GENERATION
# =============================================================================


def generate_svg(regions: List[Region], width: int, height: int) -> str:
    """Generate SVG with overlapping region paths."""
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
    ]

    for region in regions:
        # Extract contour from smoothed mask
        contour = mask_to_smooth_contour(region.smoothed_mask, sigma=1.0)
        if contour is None:
            continue

        color = region.color
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        # Build path data (note: contour is [row, col] = [y, x])
        coords = " ".join([f"{pt[1]:.2f},{pt[0]:.2f}" for pt in contour])
        path_d = f"M {coords} Z"

        svg_parts.append(f'<path d="{path_d}" fill="{hex_color}" stroke="none"/>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


# =============================================================================
# STAGE 5: MATPLOTLIB RASTERIZATION
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

    # Get buffer with correct dimensions
    # Use the figure's actual size in pixels
    width_px = int(fig.get_figwidth() * fig.dpi)
    height_px = int(fig.get_figheight() * fig.dpi)

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(height_px, width_px, 4)

    # Resize if needed
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
    regions: List[Region],
    svg_render: np.ndarray,
    config: PosterConfig,
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
    # Show label map with distinct colors
    np.random.seed(42)
    label_colors = np.random.rand(len(palette), 3)
    label_vis = label_colors[label_map]
    ax3.imshow(label_vis)
    ax3.set_title(f"3. Label Map ({len(np.unique(label_map))} unique)")
    ax3.axis("off")

    # Row 2
    ax4 = fig.add_subplot(2, 3, 4)
    # Show smoothed masks overlay
    overlay = original.copy().astype(float)
    for region in regions:
        mask = region.smoothed_mask
        color = region.color / 255.0
        overlay[mask] = overlay[mask] * 0.3 + color * 0.7
    ax4.imshow(overlay.astype(np.uint8))
    ax4.set_title(f"4. Smoothed Regions ({len(regions)} total)")
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

    # Gap: render is dark/empty but original has content
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
    image_path: str, output_path: str, config: Optional[PosterConfig] = None
):
    """Run corrected poster vectorization pipeline."""
    config = config or PosterConfig()

    print(f"Loading {image_path}...")
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"  Size: {w}x{h}")

    print(f"Quantizing to {config.n_colors} colors...")
    label_map, palette = quantize_colors(image, config.n_colors)
    print(f"  Palette: {len(palette)} colors")
    print(f"  Label map unique: {len(np.unique(label_map))}")

    print("Extracting and smoothing regions...")
    regions = extract_and_smooth_regions(label_map, palette, config)
    print(f"  Regions: {len(regions)}")

    print("Generating SVG...")
    svg = generate_svg(regions, w, h)

    print("Rendering with matplotlib...")
    svg_render = render_svg_matplotlib(svg, w, h)

    print("Creating comparison figure...")
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
        description="Poster vectorization - corrected overlapping regions"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o", "--output", default="comparison.png", help="Output comparison image"
    )
    parser.add_argument("--colors", type=int, default=12, help="Number of colors")
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Boundary smoothing sigma"
    )
    parser.add_argument(
        "--dilate", type=int, default=2, help="Mask dilation for overlap (pixels)"
    )

    args = parser.parse_args()

    config = PosterConfig(
        n_colors=args.colors, gaussian_sigma=args.sigma, mask_dilate=args.dilate
    )

    process_image(args.input, args.output, config)
