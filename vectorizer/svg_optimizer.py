"""SVG generation and optimization."""
from typing import List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

from vectorizer.types import VectorRegion, RegionKind, GradientType, AdaptiveConfig


def regions_to_svg(
    regions: List[VectorRegion],
    width: int,
    height: int,
    precision: int = 2,
    config: Optional[AdaptiveConfig] = None
) -> str:
    """
    Convert vectorized regions to SVG string.
    
    Args:
        regions: List of vectorized regions
        width: Image width
        height: Image height
        precision: Decimal places for coordinates
        config: Optional configuration for display options
        
    Returns:
        SVG XML string
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Get display options from config or use defaults
    transparent_bg = config.transparent_background if config else True
    
    # Create SVG root element
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('viewBox', f'0 0 {width} {height}')
    
    # Add transparent background if requested
    if not transparent_bg:
        bg_rect = ET.SubElement(svg, 'rect')
        bg_rect.set('x', '0')
        bg_rect.set('y', '0')
        bg_rect.set('width', str(width))
        bg_rect.set('height', str(height))
        bg_rect.set('fill', '#ffffff')
        logger.debug("Added white background")
    else:
        logger.debug("Transparent background enabled - no background rect added")
    
    # Create defs for gradients
    defs = ET.SubElement(svg, 'defs')
    
    # Track gradient IDs
    gradient_id = 0
    
    # Sort regions by area (smallest first for proper layering)
    # Smaller regions should be drawn first, larger regions on top
    # This ensures details are visible and large areas fill properly
    def get_region_area(region):
        if not region.path:
            return 0
        # Calculate bounding box area from path points
        all_x = []
        all_y = []
        for curve in region.path:
            all_x.extend([curve.p0.x, curve.p1.x, curve.p2.x, curve.p3.x])
            all_y.extend([curve.p0.y, curve.p1.y, curve.p2.y, curve.p3.y])
        if all_x and all_y:
            w = max(all_x) - min(all_x)
            h = max(all_y) - min(all_y)
            return w * h
        return 0
    
    # Sort by area (smallest first = draw first = on bottom)
    sorted_regions = sorted(regions, key=get_region_area)
    
    # Add regions (smaller/detailed regions first, larger regions on top)
    regions_with_paths = 0
    regions_without_paths = 0
    
    for region in sorted_regions:
        if not region.path:
            regions_without_paths += 1
            logger.warning(f"Skipping region with empty path (kind={region.kind})")
            continue
        
        regions_with_paths += 1
        
        # Convert path to SVG path data
        path_data = _bezier_to_svg_path(region.path, precision)
        
        # Create path element
        path_elem = ET.SubElement(svg, 'path')
        path_elem.set('d', path_data)
        
        # Set fill-rule to evenodd for proper hole handling
        path_elem.set('fill-rule', 'evenodd')
        
        # Set fill based on region kind
        if region.kind == RegionKind.FLAT and region.fill_color is not None:
            color = _color_to_hex(region.fill_color)
            path_elem.set('fill', color)
        
        elif region.kind == RegionKind.GRADIENT and region.gradient_type is not None:
            # Create gradient definition
            grad_id = f'gradient_{gradient_id}'
            gradient_id += 1
            
            _create_gradient_def(defs, region, grad_id)
            path_elem.set('fill', f'url(#{grad_id})')
        
        elif region.kind == RegionKind.DETAIL and region.mesh_triangles is not None:
            # For detail regions, use average color as fallback
            if region.mesh_colors is not None and len(region.mesh_colors) > 0:
                avg_color = np.mean(region.mesh_colors, axis=0)
                path_elem.set('fill', _color_to_hex(avg_color))
            else:
                path_elem.set('fill', '#808080')
        
        else:
            # Default fill
            path_elem.set('fill', '#808080')
        
        # Set stroke to none (no outline)
        path_elem.set('stroke', 'none')
    
    # Log region statistics
    logger.info(
        f"SVG generation: {regions_with_paths} regions with paths, "
        f"{regions_without_paths} regions skipped (empty paths)"
    )
    
    # Convert to string
    svg_string = ET.tostring(svg, encoding='unicode')
    
    # Pretty print
    dom = minidom.parseString(svg_string)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    
    return '\n'.join(lines)


def _bezier_to_svg_path(bezier_curves, precision: int = 2) -> str:
    """Convert bezier curves to SVG path data string."""
    if not bezier_curves:
        return ''
    
    fmt = f'{{:.{precision}f}}'
    
    # Start at first point
    p0 = bezier_curves[0].p0
    path_data = f'M {fmt.format(p0.x)} {fmt.format(p0.y)}'
    
    # Add each curve
    for curve in bezier_curves:
        # Cubic bezier: C x1 y1, x2 y2, x y
        path_data += (
            f' C {fmt.format(curve.p1.x)} {fmt.format(curve.p1.y)},'
            f' {fmt.format(curve.p2.x)} {fmt.format(curve.p2.y)},'
            f' {fmt.format(curve.p3.x)} {fmt.format(curve.p3.y)}'
        )
    
    # Close path
    path_data += ' Z'
    
    return path_data


def _color_to_hex(color: np.ndarray) -> str:
    """
    Convert RGB color to hex string.
    
    Expects color in sRGB uint8 format [0-255].
    Handles both uint8 and float [0,1] inputs for compatibility.
    """
    if color is None or len(color) < 3:
        return '#808080'  # Default gray
    
    # Detect if color is in float [0, 1] or uint8 [0, 255] format
    if color.max() <= 1.0:
        # Float [0, 1] format - convert to uint8
        rgb = (np.clip(color, 0, 1) * 255).astype(np.uint8)
    else:
        # Already in uint8 format
        rgb = color.astype(np.uint8)
    
    # Ensure we have exactly 3 channels
    rgb = rgb[:3]
    
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


def _create_gradient_def(defs, region: VectorRegion, grad_id: str):
    """Create gradient definition element."""
    if region.gradient_type == GradientType.LINEAR:
        grad = ET.SubElement(defs, 'linearGradient')
        grad.set('id', grad_id)
        
        if region.gradient_start and region.gradient_end:
            grad.set('x1', str(region.gradient_start.x))
            grad.set('y1', str(region.gradient_start.y))
            grad.set('x2', str(region.gradient_end.x))
            grad.set('y2', str(region.gradient_end.y))
        else:
            # Default gradient direction (left to right)
            grad.set('x1', '0%')
            grad.set('y1', '0%')
            grad.set('x2', '100%')
            grad.set('y2', '0%')
    
    elif region.gradient_type == GradientType.RADIAL:
        grad = ET.SubElement(defs, 'radialGradient')
        grad.set('id', grad_id)
        
        if region.gradient_center:
            grad.set('cx', str(region.gradient_center.x))
            grad.set('cy', str(region.gradient_center.y))
        
        if region.gradient_radius:
            grad.set('r', str(region.gradient_radius))
    
    else:
        # Default to linear
        grad = ET.SubElement(defs, 'linearGradient')
        grad.set('id', grad_id)
    
    # Add color stops
    for stop in region.gradient_stops:
        stop_elem = ET.SubElement(grad, 'stop')
        stop_elem.set('offset', f'{stop.offset * 100:.1f}%')
        stop_elem.set('stop-color', _color_to_hex(stop.color))


def optimize_svg(svg_string: str) -> str:
    """
    Optimize SVG by removing unnecessary precision and whitespace.
    
    Args:
        svg_string: Input SVG string
        
    Returns:
        Optimized SVG string
    """
    # Parse SVG
    root = ET.fromstring(svg_string)
    
    # Remove whitespace text nodes
    _remove_whitespace(root)
    
    # Convert back to string
    svg_string = ET.tostring(root, encoding='unicode')
    
    return svg_string


def _remove_whitespace(element):
    """Remove whitespace-only text nodes from XML tree."""
    if element.text and not element.text.strip():
        element.text = None
    
    if element.tail and not element.tail.strip():
        element.tail = None
    
    for child in element:
        _remove_whitespace(child)


def get_svg_size(svg_string: str) -> int:
    """Get size of SVG in bytes."""
    return len(svg_string.encode('utf-8'))
