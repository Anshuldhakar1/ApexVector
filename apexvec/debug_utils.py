"""Debug utilities for region and color auditing."""
import logging
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from apexvec.types import Region, VectorRegion

logger = logging.getLogger(__name__)


def audit_regions_at_extraction(regions: List[Region], phase: str = "extraction") -> dict:
    """
    Audit regions after extraction to identify potential issues.
    
    Args:
        regions: List of regions to audit
        phase: Description of the phase (for logging)
        
    Returns:
        Dictionary with audit statistics
    """
    stats = {
        "total_regions": len(regions),
        "valid_masks": 0,
        "empty_masks": 0,
        "dark_colors": 0,
        "bright_colors": 0,
        "zero_area": 0,
        "total_pixels": 0,
    }
    
    dark_threshold = 60  # RGB values below this are considered "dark"
    
    for i, region in enumerate(regions):
        # Check mask validity
        mask_pixels = np.sum(region.mask)
        stats["total_pixels"] += mask_pixels
        
        if region.mask is None or region.mask.size == 0:
            stats["empty_masks"] += 1
            logger.warning(f"Region {i} (label={region.label}): Empty mask")
            continue
        
        if not np.any(region.mask):
            stats["empty_masks"] += 1
            logger.warning(f"Region {i} (label={region.label}): Mask has no True pixels")
            continue
        
        stats["valid_masks"] += 1
        
        if mask_pixels == 0:
            stats["zero_area"] += 1
        
        # Check color
        if region.mean_color is not None:
            color = region.mean_color
            # Handle both float [0,1] and uint8 [0,255] formats
            if color.max() <= 1.0:
                color_scaled = color * 255
            else:
                color_scaled = color
            
            if np.all(color_scaled < dark_threshold):
                stats["dark_colors"] += 1
                logger.debug(
                    f"Region {i} (label={region.label}): Dark color detected "
                    f"RGB=({color_scaled[0]:.1f}, {color_scaled[1]:.1f}, {color_scaled[2]:.1f}), "
                    f"area={mask_pixels}"
                )
            else:
                stats["bright_colors"] += 1
    
    logger.info(
        f"Region audit ({phase}): {stats['total_regions']} total, "
        f"{stats['valid_masks']} valid masks, "
        f"{stats['empty_masks']} empty, "
        f"{stats['dark_colors']} dark colors, "
        f"{stats['bright_colors']} bright colors"
    )
    
    return stats


def audit_regions_at_svg(
    regions: List[Region],
    vector_regions: List[VectorRegion],
    phase: str = "svg_export"
) -> dict:
    """
    Audit regions at SVG export to track which survive.
    
    Args:
        regions: Original input regions
        vector_regions: Vectorized regions
        phase: Description of the phase
        
    Returns:
        Dictionary with audit statistics
    """
    stats = {
        "input_regions": len(regions),
        "output_regions": len(vector_regions),
        "dropped_regions": len(regions) - len(vector_regions),
        "empty_paths": 0,
        "valid_paths": 0,
    }
    
    # Check which regions have valid paths
    for i, vregion in enumerate(vector_regions):
        if not vregion.path:
            stats["empty_paths"] += 1
            logger.warning(f"VectorRegion {i}: Empty path (kind={vregion.kind})")
        else:
            stats["valid_paths"] += 1
    
    if stats["dropped_regions"] > 0:
        logger.error(
            f"REGION DROP DETECTED ({phase}): "
            f"{stats['input_regions']} input -> {stats['output_regions']} output, "
            f"{stats['dropped_regions']} regions dropped!"
        )
    else:
        logger.info(
            f"Region count check ({phase}): "
            f"{stats['input_regions']} input = {stats['output_regions']} output ✓"
        )
    
    return stats


def create_color_audit_visualization(
    regions: List[Region],
    image_shape: Tuple[int, int],
    output_path: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    Create a visualization showing each region filled with its assigned color.
    
    Args:
        regions: List of regions
        image_shape: (height, width) of output image
        output_path: Optional path to save visualization
        
    Returns:
        RGB image array if successful, None otherwise
    """
    if not HAS_PIL:
        logger.warning("PIL not available, skipping color audit visualization")
        return None
    
    height, width = image_shape[:2]
    viz = np.zeros((height, width, 3), dtype=np.uint8)
    
    for region in regions:
        if region.mask is None or not np.any(region.mask):
            continue
        
        if region.mean_color is None:
            continue
        
        # Convert color to uint8
        color = region.mean_color
        if color.max() <= 1.0:
            color_uint8 = (color * 255).astype(np.uint8)
        else:
            color_uint8 = color.astype(np.uint8)
        
        # Fill mask with color
        viz[region.mask] = color_uint8
    
    if output_path:
        Image.fromarray(viz).save(output_path)
        logger.info(f"Color audit visualization saved to {output_path}")
    
    return viz


def create_missing_pixel_visualization(
    regions: List[Region],
    original_image: np.ndarray,
    output_path: Optional[Path] = None
) -> Optional[np.ndarray]:
    """
    Create visualization showing pixels not covered by any region.
    
    Missing pixels are shown in magenta on top of the original image.
    
    Args:
        regions: List of regions
        original_image: Original input image (for background)
        output_path: Optional path to save visualization
        
    Returns:
        RGB image array if successful, None otherwise
    """
    if not HAS_PIL:
        logger.warning("PIL not available, skipping missing pixel visualization")
        return None
    
    height, width = original_image.shape[:2]
    
    # Composite all region masks
    composite_mask = np.zeros((height, width), dtype=bool)
    for region in regions:
        if region.mask is not None:
            composite_mask |= region.mask
    
    # Find missing pixels
    missing_pixels = ~composite_mask
    missing_count = np.sum(missing_pixels)
    
    # Create visualization
    if original_image.max() <= 1.0:
        viz = (original_image * 255).astype(np.uint8).copy()
    else:
        viz = original_image.astype(np.uint8).copy()
    
    # Ensure RGB
    if viz.ndim == 2:
        viz = np.stack([viz] * 3, axis=-1)
    
    # Overlay missing pixels in magenta
    viz[missing_pixels] = [255, 0, 255]  # Magenta
    
    missing_percent = (missing_count / (height * width)) * 100
    
    if missing_count > 0:
        logger.warning(
            f"Missing pixel visualization: {missing_count} pixels ({missing_percent:.2f}%) not covered by any region"
        )
    else:
        logger.info("Missing pixel visualization: All pixels covered by regions ✓")
    
    if output_path:
        Image.fromarray(viz).save(output_path)
        logger.info(f"Missing pixel visualization saved to {output_path}")
    
    return viz


def log_boundary_extraction(
    region_idx: int,
    region_label: int,
    input_points: int,
    output_curves: int,
    success: bool
):
    """Log boundary extraction results for a single region."""
    if success:
        logger.debug(
            f"Region {region_idx} (label={region_label}): "
            f"Boundary extracted: {input_points} points -> {output_curves} curves"
        )
    else:
        logger.warning(
            f"Region {region_idx} (label={region_label}): "
            f"Boundary extraction FAILED ({input_points} points input)"
        )
