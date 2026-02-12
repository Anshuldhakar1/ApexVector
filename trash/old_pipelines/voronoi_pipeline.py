#!/usr/bin/env python3
"""
Poster Vectorization Pipeline - Voronoi-Based Gap Elimination
Implements the three-layer system with feature-aware smoothing
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
from skimage.morphology import binary_dilation, disk
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class VoronoiConfig:
    n_colors: int = 24
    gaussian_sigma: float = 0.0  # Disabled - using per-type smoothing instead
    mask_dilate: int = 0  # Disabled - replaced with Voronoi
    min_region_area: int = 50
    min_text_area: int = 10  # Lower threshold for text regions
    text_aspect_ratio: float = 2.0
    text_stroke_cv: float = 0.3  # Coefficient of variation for stroke width
    circularity_threshold: float = 0.8
    pa_ratio_threshold: float = 10.0  # Perimeter/area ratio for thin structures
    voronoi_max_gap: int = 5
    # Per-feature smoothing sigmas
    text_sigma: float = 0.5
    circle_sigma: float = 0.0  # Circles are fitted, not smoothed
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
# PHASE 1: VORONOI GAP FILLING
# =============================================================================


def voronoi_fill_gaps(label_map: np.ndarray, max_gap_size: int = 5) -> np.ndarray:
    """
    Fill gaps between regions using Voronoi assignment.
    Each unassigned pixel is assigned to the nearest region.
    """
    h, w = label_map.shape
    result = label_map.copy()

    # Find all unique labels
    unique_labels = np.unique(label_map)

    # Create a mask of assigned pixels
    assigned_mask = np.isin(label_map, unique_labels[unique_labels >= 0])

    if assigned_mask.all():
        # No gaps to fill
        return result

    # Distance transform to find nearest assigned pixel
    dist, indices = ndimage.distance_transform_edt(~assigned_mask, return_indices=True)

    # Only fill gaps within max_gap_size
    gap_mask = (~assigned_mask) & (dist <= max_gap_size)

    if gap_mask.any():
        # Assign gap pixels to nearest region
        result[gap_mask] = label_map[indices[0][gap_mask], indices[1][gap_mask]]

    return result


def expand_to_shared_boundaries(label_map: np.ndarray) -> np.ndarray:
    """
    Expand regions until they touch, creating shared boundaries.
    This implements the "balloon" effect mathematically.
    """
    h, w = label_map.shape
    result = label_map.copy()

    # Find background/unassigned pixels
    unique_labels = np.unique(label_map)
    valid_labels = unique_labels[unique_labels >= 0]

    if len(valid_labels) == 0:
        return result

    # Create distance transform from each region
    # For efficiency, compute in batches
    distances = np.full((len(valid_labels), h, w), np.inf)

    for idx, label in enumerate(valid_labels):
        mask = label_map == label
        # Distance from this region
        dist = ndimage.distance_transform_edt(~mask)
        distances[idx] = dist

    # For each pixel, find the nearest region(s)
    min_dist = np.min(distances, axis=0)

    # Assign each pixel to the nearest region
    # In case of tie (equidistant), we need Voronoi decision
    for i in range(h):
        for j in range(w):
            if result[i, j] < 0:
                # Find all regions at minimum distance
                d = min_dist[i, j]
                nearest = [
                    valid_labels[k]
                    for k in range(len(valid_labels))
                    if abs(distances[k, i, j] - d) < 0.5
                ]
                if nearest:
                    # Assign to first nearest (deterministic)
                    result[i, j] = nearest[0]

    return result


# =============================================================================
# PHASE 2: SHARED BOUNDARY EXTRACTION
# =============================================================================


@dataclass
class Boundary:
    """Represents a shared boundary between two regions."""

    label_a: int
    label_b: int
    points: np.ndarray  # Nx2 array of (row, col) coordinates
    feature_type: str = "general"  # 'text_thin', 'circular', 'general'


def extract_shared_boundaries(label_map: np.ndarray) -> List[Boundary]:
    """
    Extract boundaries as the exact interface between Voronoi cells.
    Each boundary segment knows both adjacent regions.
    """
    h, w = label_map.shape
    boundaries = []
    processed_pairs: Set[Tuple[int, int]] = set()

    # For each pair of adjacent labels, extract the shared boundary
    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    for i, label_a in enumerate(unique_labels):
        for label_b in unique_labels[i + 1 :]:
            pair = (int(label_a), int(label_b))

            if pair in processed_pairs:
                continue

            # Create a mask where label_a is 1 and label_b is 0
            # The boundary is at the interface
            mask = np.zeros((h, w), dtype=float)
            mask[label_map == label_a] = 1.0
            mask[label_map == label_b] = 0.0

            # Check if these labels are actually adjacent
            # Quick check: dilate label_a and see if it overlaps label_b
            a_mask = label_map == label_a
            b_mask = label_map == label_b

            a_dilated = ndimage.binary_dilation(a_mask, iterations=1)
            if not np.any(a_dilated & b_mask):
                continue  # Not adjacent

            # Extract contour at the interface (level 0.5)
            contours = find_contours(mask, level=0.5)

            if contours:
                # Use the longest contour
                longest = max(contours, key=len)
                if len(longest) >= 4:
                    boundary = Boundary(
                        label_a=pair[0], label_b=pair[1], points=longest
                    )
                    boundaries.append(boundary)
                    processed_pairs.add(pair)

    return boundaries


# =============================================================================
# PHASE 3: FEATURE-TYPE DETECTION
# =============================================================================


def classify_region_type(
    region_mask: np.ndarray, original_image: Optional[np.ndarray] = None
) -> str:
    """
    Detect if region contains:
    - Text/thin strokes (high perimeter/area ratio, many holes)
    - Circles (low perimeter/area ratio, single hole or solid)
    - General shapes (medium complexity)
    """
    props = regionprops(region_mask.astype(int))
    if not props:
        return "unknown"

    prop = props[0]

    # Perimeter/area ratio for thinness
    # Higher ratio = thinner, more elongated
    area = max(prop.area, 1)
    perimeter = max(prop.perimeter, 1)
    pa_ratio = (perimeter**2) / (4 * np.pi * area)

    # Circularity: 1 = perfect circle, < 1 = less circular
    circularity = (4 * np.pi * area) / (perimeter**2)

    # Euler number: number of objects - number of holes
    euler = prop.euler_number

    # Classification
    if pa_ratio > 10:  # Thin, elongated structure
        return "text_thin"
    elif circularity > 0.8 and euler <= 0:  # Solid or single hole, very circular
        return "circular"
    else:
        return "general"


def detect_text_regions(label_map: np.ndarray, config: VoronoiConfig) -> np.ndarray:
    """
    Identify text-like regions based on aspect ratio and stroke width consistency.
    Returns a boolean mask of text regions.
    """
    h, w = label_map.shape
    text_mask = np.zeros((h, w), dtype=bool)
    text_labels = set()

    unique_labels = np.unique(label_map)

    for region_id in unique_labels:
        if region_id < 0:
            continue

        mask = label_map == region_id
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
            text_labels.add(region_id)
            text_mask[mask] = True

    return text_mask, text_labels


def merge_horizontal_text_regions(
    label_map: np.ndarray, text_labels: Set[int], max_vertical_gap: int = 5
) -> np.ndarray:
    """
    Merge horizontally adjacent text regions to prevent fragmentation.
    """
    if not text_labels:
        return label_map

    result = label_map.copy()
    h, w = label_map.shape

    # Create a text-only label map
    text_only = np.zeros_like(label_map)
    for label in text_labels:
        text_only[label_map == label] = label

    # Dilate horizontally to connect nearby text
    structure = np.zeros((3, max_vertical_gap * 2 + 1))
    structure[1, :] = 1  # Horizontal line

    text_dilated = ndimage.binary_dilation(text_only > 0, structure=structure)

    # Relabel connected text regions
    text_connected, num_features = ndimage.label(text_dilated)

    # For each connected component, assign a single label
    # Use the most common original label in that region
    for comp_id in range(1, num_features + 1):
        comp_mask = text_connected == comp_id
        original_labels = label_map[comp_mask]
        if len(original_labels) > 0:
            # Find most common label (excluding background)
            valid_labels = original_labels[original_labels >= 0]
            if len(valid_labels) > 0:
                most_common = np.bincount(valid_labels).argmax()
                result[comp_mask] = most_common

    return result


# =============================================================================
# PHASE 4: PER-TYPE SMOOTHING
# =============================================================================


def fit_circle_to_points(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """
    Fit a circle to a set of points using least squares.
    Returns (center_x, center_y), radius
    """
    if len(points) < 3:
        # Not enough points to fit a circle
        center = np.mean(points, axis=0) if len(points) > 0 else np.array([0, 0])
        return (center[1], center[0]), 1.0  # Return as (x, y), radius

    # points are (row, col) = (y, x)
    y_coords = points[:, 0]
    x_coords = points[:, 1]

    # Use algebraic circle fitting (Kasa method)
    # Circle equation: (x-a)^2 + (y-b)^2 = r^2
    # Expanding: x^2 + y^2 - 2ax - 2by + (a^2 + b^2 - r^2) = 0

    A = np.column_stack([x_coords, y_coords, np.ones(len(x_coords))])
    B = x_coords**2 + y_coords**2

    # Solve Ax = B
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        a = coeffs[0] / 2
        b = coeffs[1] / 2
        r = np.sqrt(coeffs[2] + a**2 + b**2)
        return (a, b), r
    except:
        # Fallback to centroid
        center = np.mean(points, axis=0)
        return (center[1], center[0]), np.std(points) + 1


def generate_circle_contour(
    center: Tuple[float, float], radius: float, n_points: int = 100
) -> np.ndarray:
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([y, x])  # Return as (row, col) = (y, x)


def smooth_boundary(
    points: np.ndarray, feature_type: str, config: VoronoiConfig
) -> np.ndarray:
    """
    Apply appropriate smoothing based on feature type.
    """
    if len(points) < 4:
        return points

    if feature_type == "text_thin":
        # Minimal smoothing: preserve corners, slight Gaussian
        if config.text_sigma > 0:
            # Apply Gaussian filter to coordinates
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
        center, radius = fit_circle_to_points(points)
        return generate_circle_contour(center, radius, n_points=len(points))

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


# =============================================================================
# PHASE 5: TEXT-SPECIFIC HANDLING (Integrated)
# =============================================================================


def process_with_text_protection(
    label_map: np.ndarray,
    palette: np.ndarray,
    original_image: np.ndarray,
    config: VoronoiConfig,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Process label map with text detection and protection.
    Returns modified label map and feature type dictionary.
    """
    # Detect text regions
    text_mask, text_labels = detect_text_regions(label_map, config)
    print(f"  Detected {len(text_labels)} text-like regions")

    # Merge horizontally adjacent text
    label_map_merged = merge_horizontal_text_regions(label_map, text_labels)

    # Classify all regions
    feature_types = {}
    unique_labels = np.unique(label_map_merged)

    for label in unique_labels:
        if label < 0:
            continue

        mask = label_map_merged == label

        # Check if this was detected as text
        if label in text_labels:
            feature_types[label] = "text_thin"
        else:
            feature_types[label] = classify_region_type(mask, original_image)

    return label_map_merged, feature_types


# =============================================================================
# REGION DATA STRUCTURES
# =============================================================================


@dataclass
class VoronoiRegion:
    """Region with Voronoi-based shared boundaries."""

    label_id: int
    color: np.ndarray
    boundaries: List[Boundary]  # Shared boundaries with other regions
    area: int
    centroid: Tuple[float, float]
    mask: np.ndarray = field(repr=False)  # Boolean mask of region
    feature_type: str = "general"


def extract_voronoi_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    feature_types: Dict[int, str],
    config: VoronoiConfig,
) -> List[VoronoiRegion]:
    """
    Extract connected component regions from the ORIGINAL label map.
    Like corrected.py - don't use Voronoi-filled map for region extraction.
    Merges nearby components of the same color to reduce over-segmentation.
    """
    h, w = label_map.shape
    regions = []
    region_counter = 0

    unique_labels = np.unique(label_map)
    unique_labels = unique_labels[unique_labels >= 0]

    for label in unique_labels:
        color_mask = label_map == label
        if not color_mask.any():
            continue

        # Merge nearby components of the same color using morphological closing
        # This connects fragmented regions that are close together
        merged_mask = ndimage.binary_closing(color_mask, iterations=1)

        # Label connected components of this merged color mask
        labeled, num_features = ndimage.label(merged_mask)

        for comp_id in range(1, num_features + 1):
            mask = labeled == comp_id
            area = int(mask.sum())

            # Adaptive area threshold based on feature type
            feature_type = feature_types.get(label, "general")
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

            region = VoronoiRegion(
                label_id=region_counter,  # Unique ID for each component
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


# =============================================================================
# SVG GENERATION WITH SHARED BOUNDARIES
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

    # For very small regions, use simpler contour extraction
    area = mask.sum()
    if area < 100:
        # Use basic contour extraction without heavy smoothing
        contours = find_contours(mask.astype(float), level=0.5)
        if contours:
            longest = max(contours, key=len)
            if len(longest) >= 3:
                return longest
        return None

    # Pad to avoid edge issues
    padded = np.pad(mask, 1, mode="constant")

    # Distance transform for sub-pixel boundary
    dist = ndimage.distance_transform_edt(padded) - ndimage.distance_transform_edt(
        ~padded
    )

    # Only apply Gaussian if sigma > 0
    if sigma > 0:
        smooth_dist = ndimage.gaussian_filter(dist, sigma=sigma)
    else:
        smooth_dist = dist

    # Marching squares at zero level
    contours = find_contours(smooth_dist, level=0)

    # If no contours at level 0, try level 0.5
    if not contours:
        contours = find_contours(smooth_dist, level=0.5)

    # If still no contours, try without smoothing
    if not contours and sigma > 0:
        contours = find_contours(dist, level=0)

    if not contours:
        # Last resort: use simple contour on original mask
        contours = find_contours(mask.astype(float), level=0.5)

    if not contours:
        return None

    # Use longest contour, remove padding offset
    longest = max(contours, key=len)
    longest = longest - 1  # Remove padding

    # Clip to image bounds
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
    """
    Generate SVG from Voronoi-filled regions with overlap to prevent gaps.
    Extracts closed contours from dilated masks to ensure gap-free rendering.
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" '
        f'fill-rule="evenodd">'
    ]

    # Sort regions by area (largest first) so smaller regions paint on top
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)

    for region in regions_sorted:
        # Get feature type for this region
        feature_type = feature_types.get(region.label_id, "general")

        # Get appropriate sigma for this feature type
        if feature_type == "text_thin":
            sigma = config.text_sigma
        elif feature_type == "circular":
            sigma = config.circle_sigma
        else:
            sigma = config.general_sigma

        # Dilate mask slightly to ensure overlap with neighbors (prevents gaps)
        # Use 3 pixels dilation for better gap coverage
        dilated_mask = ndimage.binary_dilation(region.mask, iterations=3)

        # Extract contour from the DILATED mask
        contour = mask_to_smooth_contour(dilated_mask, sigma=sigma)

        if contour is None:
            continue

        # Apply additional smoothing based on feature type
        contour = smooth_boundary(contour, feature_type, config)

        # Build path data (note: contour is [row, col] = [y, x])
        coords = " ".join([f"{pt[1]:.2f},{pt[0]:.2f}" for pt in contour])
        path_d = f"M {coords} Z"

        color = region.color
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"

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
    regions: List[VoronoiRegion],
    boundaries: List[Boundary],
    svg_render: np.ndarray,
    config: VoronoiConfig,
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
    # Show region overlay with feature type coloring
    overlay = original.copy().astype(float)
    for region in regions:
        mask = region.mask
        color = region.color / 255.0
        overlay[mask] = overlay[mask] * 0.3 + color * 0.7
    ax4.imshow(overlay.astype(np.uint8))
    ax4.set_title(f"4. Voronoi Regions ({len(regions)} total)")
    ax4.axis("off")

    # Show boundaries
    for boundary in boundaries[:50]:  # Limit to first 50 for visibility
        points = boundary.points
        ax4.plot(points[:, 1], points[:, 0], "r-", linewidth=0.5, alpha=0.3)

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
# POST-PROCESSING GAP FILL
# =============================================================================


def fill_gaps_post_process(
    rendered: np.ndarray, label_map: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """
    Fill gaps in rendered image using quantized colors.
    Preserves visual quality by using the original label map colors.
    """
    result = rendered.copy()
    h, w = rendered.shape[:2]

    # Find gap pixels (nearly transparent or very dark)
    if rendered.shape[2] == 4:
        # RGBA
        gap_mask = rendered[:, :, 3] < 128  # Low alpha
        rgb_render = rendered[:, :, :3]
    else:
        # RGB
        rgb_render = rendered[:, :, :3]
        # Gaps appear as very dark pixels where render is empty
        gap_mask = rgb_render.mean(axis=2) < 10

    if not gap_mask.any():
        return result

    # Build quantized image from label map
    quantized = palette[label_map]

    # Fill gaps with quantized colors
    result[gap_mask, :3] = quantized[gap_mask]
    if rendered.shape[2] == 4:
        result[gap_mask, 3] = 255  # Full opacity

    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def process_image_voronoi(
    image_path: str, output_path: str, config: Optional[VoronoiConfig] = None
):
    """Run Voronoi-based poster vectorization pipeline."""
    config = config or VoronoiConfig()

    print(f"Loading {image_path}...")
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    print(f"  Size: {w}x{h}")

    print(f"\nPhase 0: Color Quantization")
    print(f"  Quantizing to {config.n_colors} colors...")
    label_map, palette = quantize_colors(image, config.n_colors)
    print(f"  Palette: {len(palette)} colors")
    print(f"  Label map unique: {len(np.unique(label_map))}")

    print(f"\nPhase 1: Voronoi Gap Filling")
    label_map_filled = voronoi_fill_gaps(label_map, config.voronoi_max_gap)
    # Additional expansion to ensure shared boundaries
    label_map_expanded = expand_to_shared_boundaries(label_map_filled)
    print(f"  Filled gaps, regions: {len(np.unique(label_map_expanded))}")

    print(f"\nPhase 2: Shared Boundary Extraction")
    boundaries = extract_shared_boundaries(label_map_expanded)
    print(f"  Extracted {len(boundaries)} shared boundaries")

    print(f"\nPhase 3-5: Feature Detection & Text Protection")
    # Use ORIGINAL label_map for feature detection (not Voronoi-filled)
    label_map_processed, feature_types = process_with_text_protection(
        label_map, palette, image, config
    )

    # Count feature types
    type_counts = {}
    for ft in feature_types.values():
        type_counts[ft] = type_counts.get(ft, 0) + 1
    print(f"  Feature types: {type_counts}")

    print(f"\nExtracting Voronoi regions...")
    # Use ORIGINAL label_map for region extraction (like corrected.py)
    # Voronoi filling causes gaps - don't use it for extraction
    regions = extract_voronoi_regions(label_map, palette, feature_types, config)
    print(f"  Regions: {len(regions)}")

    print(f"\nGenerating SVG with shared boundaries...")
    svg = generate_svg_voronoi(regions, boundaries, feature_types, w, h, config)

    print(f"Rendering with matplotlib...")
    svg_render = render_svg_matplotlib(svg, w, h)

    print(f"Post-processing: Filling remaining gaps...")
    svg_render = fill_gaps_post_process(svg_render, label_map, palette)

    print(f"Creating comparison figure...")
    fig = create_comparison_figure(
        image, label_map_processed, palette, regions, boundaries, svg_render, config
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
        "n_boundaries": len(boundaries),
        "feature_types": type_counts,
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
        description="Poster vectorization - Voronoi-based shared boundaries"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument(
        "-o",
        "--output",
        default="voronoi_comparison.png",
        help="Output comparison image",
    )
    parser.add_argument("--colors", type=int, default=24, help="Number of colors")
    parser.add_argument(
        "--text-sigma", type=float, default=0.5, help="Smoothing for text boundaries"
    )
    parser.add_argument(
        "--general-sigma", type=float, default=1.5, help="Smoothing for general shapes"
    )
    parser.add_argument(
        "--min-region-area", type=int, default=50, help="Minimum region area (general)"
    )
    parser.add_argument(
        "--min-text-area", type=int, default=10, help="Minimum region area (text)"
    )

    args = parser.parse_args()

    config = VoronoiConfig(
        n_colors=args.colors,
        text_sigma=args.text_sigma,
        general_sigma=args.general_sigma,
        min_region_area=args.min_region_area,
        min_text_area=args.min_text_area,
    )

    process_image_voronoi(args.input, args.output, config)
