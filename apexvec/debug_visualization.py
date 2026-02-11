"""Debug visualization for poster pipeline stages."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def visualize_quantization(
    original: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    output_path: Path
):
    """
    Stage 1: Show original with quantized colors as faint tint.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Quantized
    quantized = palette[label_map]
    axes[1].imshow(quantized)
    axes[1].set_title(f'Quantized ({len(palette)} colors)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_regions(
    original: np.ndarray,
    label_map: np.ndarray,
    output_path: Path
):
    """
    Stage 2: Show region boundaries in random colors.
    """
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(label_map, mode='thick')

    # Color boundaries
    viz = original.copy()
    boundary_color = [255, 0, 0]  # Red boundaries
    viz[boundaries] = boundary_color

    Image.fromarray(viz).save(output_path)


def visualize_shared_boundaries(
    boundaries: Dict[Tuple[int, int], List[Tuple[float, float]]],
    image_shape: Tuple[int, int],
    output_path: Path
):
    """
    Stage 3: Show shared boundary lines in different colors.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Random color for each boundary
    np.random.seed(42)
    for (label_a, label_b), points in boundaries.items():
        color = np.random.rand(3)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, linewidth=1,
                label=f'{label_a}-{label_b}')

    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Invert Y for image coords
    ax.set_aspect('equal')
    ax.set_title('Shared Boundaries')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_smoothed_boundaries(
    original_boundaries: Dict,
    smoothed_boundaries: Dict,
    image_shape: Tuple[int, int],
    output_path: Path
):
    """
    Stage 4: Compare original vs smoothed boundaries.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, boundaries, title in [
        (axes[0], original_boundaries, 'Original Boundaries'),
        (axes[1], smoothed_boundaries, 'Smoothed Boundaries')
    ]:
        for (label_a, label_b), points in boundaries.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, linewidth=1)

        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)
        ax.set_aspect('equal')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_comparison_panel(
    original: np.ndarray,
    quantized: np.ndarray,
    svg_rasterized: np.ndarray,
    output_path: Path
):
    """
    Stage 7: 4-panel comparison: original, quantized, SVG, gap mask.
    """
    # Calculate gap mask
    gap_mask = np.zeros((*original.shape[:2], 3), dtype=np.uint8)

    # Magenta: transparent in SVG but not in original
    # (simplified: check if SVG pixel is black/white vs original)
    gray_svg = np.mean(svg_rasterized, axis=2)
    gap_mask[gray_svg < 10] = [255, 0, 255]  # Magenta for gaps

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(quantized)
    axes[0, 1].set_title('Quantized')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(svg_rasterized)
    axes[1, 0].set_title('SVG Output')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gap_mask)
    axes[1, 1].set_title('Gap Mask (Magenta=Gaps)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
