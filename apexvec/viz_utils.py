"""Visualization utilities for pipeline stage debugging."""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from apexvec.types import Region, VectorRegion, RegionKind, BezierCurve
from apexvec.raster_ingest import linear_to_srgb


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is RGB format suitable for saving."""
    if image.ndim == 2:
        # Grayscale - convert to RGB
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        # Single channel - convert to RGB
        image = np.repeat(image, 3, axis=-1)

    # Clip to valid range
    if image.max() > 1.0:
        image = np.clip(image / 255.0, 0, 1)
    else:
        image = np.clip(image, 0, 1)

    return image


def save_image(image: np.ndarray, path: Path, dpi: int = 150):
    """Save numpy array as image file."""
    image = _ensure_rgb(image)
    # Convert to uint8
    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save(path, dpi=(dpi, dpi))


def visualize_ingest(image_srgb: np.ndarray, output_path: Path, dpi: int = 150):
    """Visualize ingested image (stage 1).

    Shows the EXIF-corrected, resized input image in sRGB for display.
    """
    save_image(image_srgb, output_path, dpi)


def visualize_regions(
    image: np.ndarray,
    regions: List[Region],
    output_path: Path,
    dpi: int = 150
):
    """Visualize decomposed regions (stage 2).

    Shows the SLIC segmentation with region boundaries overlaid.
    """
    from skimage.segmentation import find_boundaries, mark_boundaries

    # Convert linear to sRGB for display
    if image.max() <= 1.0:
        image_srgb = linear_to_srgb(image)
    else:
        image_srgb = image / 255.0

    # Create segments array
    height, width = image.shape[:2]
    segments = np.zeros((height, width), dtype=int)
    for region in regions:
        segments[region.mask] = region.label

    # Mark boundaries
    visualization = mark_boundaries(image_srgb, segments, color=(1, 0, 0), mode='thick')

    save_image(visualization, output_path, dpi)


def visualize_classification(
    image: np.ndarray,
    regions: List[Region],
    output_path: Path,
    dpi: int = 150
):
    """Visualize classified regions (stage 3).

    Shows regions color-coded by their classification type.
    """
    height, width = image.shape[:2]
    visualization = np.zeros((height, width, 3), dtype=np.float32)

    # Color map for region types
    color_map = {
        RegionKind.FLAT: np.array([0.0, 0.8, 0.0]),      # Green
        RegionKind.GRADIENT: np.array([0.0, 0.0, 1.0]),  # Blue
        RegionKind.EDGE: np.array([1.0, 0.0, 0.0]),      # Red
        RegionKind.DETAIL: np.array([1.0, 1.0, 0.0]),    # Yellow
    }

    # Color each region by its type
    for region in regions:
        kind = getattr(region, 'kind', RegionKind.FLAT)
        color = color_map.get(kind, np.array([0.5, 0.5, 0.5]))
        visualization[region.mask] = color

    save_image(visualization, output_path, dpi)


def _bezier_point(curve: BezierCurve, t: float) -> Tuple[float, float]:
    """Evaluate a cubic Bezier curve at parameter t."""
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    x = (mt3 * curve.p0.x +
         3 * mt2 * t * curve.p1.x +
         3 * mt * t2 * curve.p2.x +
         t3 * curve.p3.x)
    y = (mt3 * curve.p0.y +
         3 * mt2 * t * curve.p1.y +
         3 * mt * t2 * curve.p2.y +
         t3 * curve.p3.y)

    return (x, y)


def _sample_bezier(curve: BezierCurve, num_points: int = 20) -> List[Tuple[float, float]]:
    """Sample points along a Bezier curve."""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        points.append(_bezier_point(curve, t))
    return points


def visualize_vectorized(
    image: np.ndarray,
    vector_regions: List[VectorRegion],
    output_path: Path,
    dpi: int = 150
):
    """Visualize vectorized regions (stage 4).

    Shows the vector paths rendered over the original image.
    """
    # Convert linear to sRGB for display
    if image.max() <= 1.0:
        image_srgb = linear_to_srgb(image)
    else:
        image_srgb = image / 255.0

    # Create faded background
    faded = image_srgb * 0.3

    # Convert to PIL Image
    height, width = image.shape[:2]
    img_array = (faded * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_img)

    # Draw each region's path
    for region in vector_regions:
        if not region.path:
            continue

        # Determine color based on region type
        if region.kind == RegionKind.FLAT and region.fill_color is not None:
            color = tuple((region.fill_color * 255).astype(int))
        else:
            color_map = {
                RegionKind.FLAT: (0, 200, 0),
                RegionKind.GRADIENT: (0, 0, 200),
                RegionKind.EDGE: (200, 0, 0),
                RegionKind.DETAIL: (200, 200, 0),
            }
            color = color_map.get(region.kind, (100, 100, 100))

        # Draw bezier curves
        for curve in region.path:
            points = _sample_bezier(curve, num_points=20)
            if len(points) >= 2:
                # Convert to integer coordinates
                line_points = [(int(p[0]), int(p[1])) for p in points]
                draw.line(line_points, fill=color, width=2)

    pil_img.save(output_path, dpi=(dpi, dpi))


def visualize_topology(
    image: np.ndarray,
    vector_regions: List[VectorRegion],
    output_path: Path,
    dpi: int = 150
):
    """Visualize merged topology (stage 5).

    Shows final regions with fills and highlights shared boundaries.
    """
    height, width = image.shape[:2]
    visualization = np.ones((height, width, 3), dtype=np.float32) * 0.9

    # Draw each region with its fill color
    for region in vector_regions:
        if not region.path:
            continue

        # Create mask from path (simplified - just draw the boundary)
        if region.fill_color is not None:
            color = region.fill_color
            if color.max() > 1.0:
                color = color / 255.0
        else:
            color = np.array([0.5, 0.5, 0.5])

        # Create a simple rasterization by filling pixels near the path
        img_array = (visualization * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(pil_img)

        # Draw filled polygon if possible
        for curve in region.path:
            points = _sample_bezier(curve, num_points=20)
            if len(points) >= 3:
                polygon_points = [(int(p[0]), int(p[1])) for p in points]
                fill_color = tuple((color * 255).astype(int))
                draw.polygon(polygon_points, fill=fill_color)

        visualization = np.array(pil_img).astype(np.float32) / 255.0

    # Highlight boundaries
    img_array = (visualization * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_img)

    for region in vector_regions:
        if not region.path:
            continue

        # Draw boundary in darker color
        for curve in region.path:
            points = _sample_bezier(curve, num_points=20)
            if len(points) >= 2:
                line_points = [(int(p[0]), int(p[1])) for p in points]
                draw.line(line_points, fill=(0, 0, 0), width=1)

    pil_img.save(output_path, dpi=(dpi, dpi))


def visualize_final(
    original_image: np.ndarray,
    svg_string: str,
    output_path: Path,
    width: int,
    height: int,
    dpi: int = 150,
    metrics: Optional[dict] = None
):
    """Visualize final output comparison (stage 6).

    Creates a side-by-side comparison of original vs rasterized SVG.
    Includes optional metrics overlay for quality assessment.
    """
    from apexvec.perceptual_loss import rasterize_svg

    # Convert original to sRGB
    if original_image.max() <= 1.0:
        original_srgb = linear_to_srgb(original_image)
    else:
        original_srgb = original_image / 255.0

    # Rasterize SVG
    try:
        rasterized = rasterize_svg(svg_string, width, height)
        rasterized = rasterized / 255.0
    except Exception:
        # If rasterization fails, create a blank image
        rasterized = np.ones_like(original_srgb) * 0.5

    # Create side-by-side comparison with header space for metrics
    header_height = 60 if metrics else 0
    comparison = np.zeros((height + header_height, width * 2 + 20, 3), dtype=np.float32)

    # Header background
    if metrics:
        comparison[:header_height, :] = np.array([0.15, 0.15, 0.18])

    # Left: original
    comparison[header_height:header_height + height, :width] = original_srgb

    # Separator
    comparison[header_height:header_height + height, width:width+20] = np.array([0.2, 0.2, 0.2])

    # Right: rasterized
    comparison[header_height:header_height + height, width+20:] = rasterized

    # Convert to PIL for text rendering
    img_array = (comparison * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_img)

    # Draw metrics in header
    if metrics:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_large = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_large = font

        # Labels
        draw.text((20, 20), "ORIGINAL", fill=(200, 200, 200), font=font_large)
        draw.text((width + 40, 20), "VECTORIZED", fill=(200, 200, 200), font=font_large)

        # Metrics on the right side
        metric_x = width * 2 - 300
        y_offset = 10

        if 'regions' in metrics:
            draw.text((metric_x, y_offset), f"Regions: {metrics['regions']}",
                     fill=(100, 200, 100), font=font)
            y_offset += 20

        if 'svg_size' in metrics and 'original_size' in metrics:
            reduction = (1 - metrics['svg_size'] / metrics['original_size']) * 100
            color = (100, 200, 100) if reduction > 0 else (200, 100, 100)
            draw.text((metric_x, y_offset), f"Size: {metrics['svg_size']:,}B ({reduction:+.1f}%)",
                     fill=color, font=font)
            y_offset += 20

        if 'time' in metrics:
            draw.text((metric_x, y_offset), f"Time: {metrics['time']:.2f}s",
                     fill=(200, 200, 200), font=font)

    pil_img.save(output_path, dpi=(dpi, dpi))


# Poster-specific visualization functions

def visualize_poster_quantization(
    image: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    output_path: Path,
    dpi: int = 150
):
    """Visualize color quantization stage for poster pipeline.

    Shows the quantized image with palette overlay.
    """
    from apexvec.raster_ingest import linear_to_srgb

    # Convert linear to sRGB for display
    if image.max() <= 1.0:
        image_srgb = linear_to_srgb(image)
    else:
        image_srgb = image / 255.0

    # Create quantized image
    height, width = image.shape[:2]
    quantized = np.zeros((height, width, 3), dtype=np.float32)

    for label_id in range(len(palette)):
        mask = label_map == label_id
        color = np.array(palette[label_id])
        if color.max() > 1.0:
            color = color / 255.0
        quantized[mask] = color

    # Create palette strip
    palette_height = 40
    total_height = height + palette_height + 20
    visualization = np.ones((total_height, width, 3), dtype=np.float32) * 0.15

    # Add quantized image
    visualization[10:10+height, :] = quantized

    # Add palette strip
    palette_y = 10 + height + 10
    color_width = width // max(len(palette), 1)
    for i, color in enumerate(palette):
        if color.max() > 1.0:
            color = color / 255.0
        x_start = i * color_width
        x_end = min((i + 1) * color_width, width)
        visualization[palette_y:palette_y + palette_height, x_start:x_end] = color

    save_image(visualization, output_path, dpi)


def visualize_poster_regions(
    image: np.ndarray,
    regions: list,
    output_path: Path,
    dpi: int = 150
):
    """Visualize extracted regions for poster pipeline.

    Shows regions with distinct colors and boundaries.
    """
    from skimage.segmentation import mark_boundaries

    # Convert linear to sRGB for display
    if image.max() <= 1.0:
        image_srgb = linear_to_srgb(image)
    else:
        image_srgb = image / 255.0

    # Create region label map
    height, width = image.shape[:2]
    label_map = np.zeros((height, width), dtype=int)

    for i, region in enumerate(regions):
        label_map[region.mask] = i + 1

    # Mark boundaries
    visualization = mark_boundaries(image_srgb, label_map, color=(1, 0, 0), mode='thick')

    # Add region count overlay
    img_array = (visualization * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"Regions: {len(regions)}", fill=(255, 255, 255), font=font)

    pil_img.save(output_path, dpi=(dpi, dpi))


def visualize_poster_boundaries(
    image: np.ndarray,
    paths: list,
    output_path: Path,
    dpi: int = 150
):
    """Visualize smoothed boundaries for poster pipeline.

    Shows the original image with smoothed paths overlaid.
    """
    # Convert linear to sRGB for display
    if image.max() <= 1.0:
        image_srgb = linear_to_srgb(image)
    else:
        image_srgb = image / 255.0

    # Create faded background
    faded = image_srgb * 0.4

    # Convert to PIL Image
    height, width = image.shape[:2]
    img_array = (faded * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(pil_img)

    # Draw each path
    for path in paths:
        if not path.curves:
            continue

        # Sample points along bezier curves
        points = []
        for curve in path.curves:
            for t in [0, 0.25, 0.5, 0.75, 1.0]:
                t2 = t * t
                t3 = t2 * t
                mt = 1 - t
                mt2 = mt * mt
                mt3 = mt2 * mt

                x = (mt3 * curve.p0.x +
                     3 * mt2 * t * curve.p1.x +
                     3 * mt * t2 * curve.p2.x +
                     t3 * curve.p3.x)
                y = (mt3 * curve.p0.y +
                     3 * mt2 * t * curve.p1.y +
                     3 * mt * t2 * curve.p2.y +
                     t3 * curve.p3.y)
                # Skip NaN values from spline fitting issues
                if not (np.isnan(x) or np.isnan(y)):
                    points.append((int(x), int(y)))

        if len(points) >= 2:
            draw.line(points, fill=(0, 255, 100), width=2)

    pil_img.save(output_path, dpi=(dpi, dpi))
