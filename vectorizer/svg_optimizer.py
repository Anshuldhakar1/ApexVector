"""SVG generation and optimization."""
from typing import List
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

from vectorizer.types import VectorRegion, RegionKind, GradientType


def regions_to_svg(
    regions: List[VectorRegion],
    width: int,
    height: int,
    precision: int = 2,
    compact: bool = True
) -> str:
    """
    Convert vectorized regions to SVG string with optional compaction.
    
    Args:
        regions: List of vectorized regions
        width: Image width
        height: Image height
        precision: Decimal places for coordinates
        compact: Whether to use compact output (minified)
        
    Returns:
        SVG XML string
    """
    if compact:
        return _regions_to_svg_compact(regions, width, height, precision)
    else:
        return _regions_to_svg_pretty(regions, width, height, precision)


def _regions_to_svg_compact(regions, width, height, precision):
    """Generate compact (minified) SVG output with aggressive optimizations."""
    # Group regions by fill color to minimize attribute repetition
    color_groups = {}
    gradients = []
    
    for region in regions:
        if not region.path:
            continue
        
        if region.kind == RegionKind.FLAT and region.fill_color is not None:
            color = _color_to_hex_compact(region.fill_color)
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(region)
        elif region.kind == RegionKind.GRADIENT and region.gradient_type is not None:
            gradients.append(region)
        elif region.kind == RegionKind.DETAIL and region.mesh_triangles is not None:
            # Use average color
            avg_color = _color_to_hex_compact(np.mean(region.mesh_colors, axis=0))
            if avg_color not in color_groups:
                color_groups[avg_color] = []
            color_groups[avg_color].append(region)
        else:
            color = '#808080'
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(region)
    
    # Merge similar colors (within tolerance) to reduce groups
    color_groups = _merge_similar_colors(color_groups, tolerance=0.02)
    
    # Build compact SVG string manually for maximum compression
    # Remove width/height if same as viewBox (redundant)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    
    # Add gradient definitions if needed with shortest IDs
    if gradients:
        parts.append('<defs>')
        for i, region in enumerate(gradients):
            # Use shortest possible IDs: a, b, c, ... z, aa, ab, etc.
            grad_id = _short_id(i)
            parts.append(_create_gradient_def_compact(region, grad_id))
        parts.append('</defs>')
    
    # Output paths grouped by fill color
    for color, group in color_groups.items():
        # Multiple paths with same color - use group
        if len(group) > 1:
            parts.append(f'<g fill="{color}">')
            for region in group:
                path_data = _bezier_to_svg_path(region.path, precision, use_relative=True)
                parts.append(f'<path d="{path_data}"/>')
            parts.append('</g>')
        else:
            # Single path
            region = group[0]
            path_data = _bezier_to_svg_path(region.path, precision, use_relative=True)
            parts.append(f'<path d="{path_data}" fill="{color}"/>')
    
    # Output gradient-filled paths
    for i, region in enumerate(gradients):
        grad_id = _short_id(i)
        path_data = _bezier_to_svg_path(region.path, precision, use_relative=True)
        parts.append(f'<path d="{path_data}" fill="url(#{grad_id})"/>')
    
    parts.append('</svg>')
    
    return ''.join(parts)


def _short_id(n):
    """Generate short ID string: 0->a, 1->b, ..., 25->z, 26->aa, etc."""
    if n < 26:
        return chr(ord('a') + n)
    else:
        return _short_id(n // 26 - 1) + chr(ord('a') + (n % 26))


def _merge_similar_colors(color_groups, tolerance=0.02):
    """Merge color groups with similar colors to reduce repetition."""
    if len(color_groups) <= 1:
        return color_groups
    
    colors = list(color_groups.keys())
    merged = {}
    used = set()
    
    for i, color1 in enumerate(colors):
        if color1 in used:
            continue
        
        # Parse color
        r1, g1, b1 = _hex_to_rgb(color1)
        merged_regions = list(color_groups[color1])
        
        # Find similar colors
        for j in range(i + 1, len(colors)):
            color2 = colors[j]
            if color2 in used:
                continue
            
            r2, g2, b2 = _hex_to_rgb(color2)
            
            # Check if colors are similar
            if (abs(r1 - r2) < tolerance and 
                abs(g1 - g2) < tolerance and 
                abs(b1 - b2) < tolerance):
                merged_regions.extend(color_groups[color2])
                used.add(color2)
        
        # Use first color as representative
        merged[color1] = merged_regions
        used.add(color1)
    
    return merged


def _hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        r = int(hex_color[0] + hex_color[0], 16) / 255.0
        g = int(hex_color[1] + hex_color[1], 16) / 255.0
        b = int(hex_color[2] + hex_color[2], 16) / 255.0
    else:
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def _regions_to_svg_pretty(regions, width, height, precision):
    """Generate pretty-printed SVG output."""
    # Create SVG root element
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('viewBox', f'0 0 {width} {height}')
    
    # Create defs for gradients
    defs = ET.SubElement(svg, 'defs')
    
    # Track gradient IDs
    gradient_id = 0
    
    # Add regions
    for region in regions:
        if not region.path:
            continue
        
        # Convert path to SVG path data
        path_data = _bezier_to_svg_path(region.path, precision, use_relative=False)
        
        # Create path element
        path_elem = ET.SubElement(svg, 'path')
        path_elem.set('d', path_data)
        
        # Set fill based on region kind
        if region.kind == RegionKind.FLAT and region.fill_color is not None:
            color = _color_to_hex_compact(region.fill_color)
            path_elem.set('fill', color)
        
        elif region.kind == RegionKind.GRADIENT and region.gradient_type is not None:
            # Create gradient definition
            grad_id = f'gradient_{gradient_id}'
            gradient_id += 1
            
            _create_gradient_def(defs, region, grad_id)
            path_elem.set('fill', f'url(#{grad_id})')
        
        elif region.kind == RegionKind.DETAIL and region.mesh_triangles is not None:
            # For detail regions, we need to create mesh gradient
            grad_id = f'mesh_{gradient_id}'
            gradient_id += 1
            
            # Use average color as fallback
            avg_color = np.mean(region.mesh_colors, axis=0)
            path_elem.set('fill', _color_to_hex_compact(avg_color))
        
        else:
            # Default fill
            path_elem.set('fill', '#808080')
    
    # Convert to string
    svg_string = ET.tostring(svg, encoding='unicode')
    
    # Pretty print
    dom = minidom.parseString(svg_string)
    pretty_xml = dom.toprettyxml(indent='  ')
    
    # Remove extra blank lines
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    
    return '\n'.join(lines)


def _bezier_to_svg_path(bezier_curves, precision: int = 2, use_relative: bool = True) -> str:
    """Convert bezier curves to SVG path data string with optimization."""
    if not bezier_curves:
        return ''
    
    fmt = f'{{:.{precision}f}}'
    
    # Start at first point
    p0 = bezier_curves[0].p0
    path_data = f'M{fmt.format(p0.x)},{fmt.format(p0.y)}'
    
    # Track current position for relative coordinates
    curr_x, curr_y = p0.x, p0.y
    
    # Add each curve
    for curve in bezier_curves:
        if use_relative:
            # Use relative cubic bezier (c) - coordinates relative to current position
            # Format: c dx1,dy1 dx2,dy2 dx,dy
            dx1 = curve.p1.x - curr_x
            dy1 = curve.p1.y - curr_y
            dx2 = curve.p2.x - curve.p1.x
            dy2 = curve.p2.y - curve.p1.y
            dx = curve.p3.x - curve.p2.x
            dy = curve.p3.y - curve.p2.y
            
            # Build compact relative path
            path_data += f'c{_fmt_compact(dx1)},{_fmt_compact(dy1)} {_fmt_compact(dx2)},{_fmt_compact(dy2)} {_fmt_compact(dx)},{_fmt_compact(dy)}'
            
            # Update current position
            curr_x, curr_y = curve.p3.x, curve.p3.y
        else:
            # Absolute coordinates
            path_data += (
                f'C{fmt.format(curve.p1.x)},{fmt.format(curve.p1.y)} '
                f'{fmt.format(curve.p2.x)},{fmt.format(curve.p2.y)} '
                f'{fmt.format(curve.p3.x)},{fmt.format(curve.p3.y)}'
            )
    
    # Close path
    path_data += 'z'
    
    return path_data


def _fmt_compact(value: float, max_precision: int = 2) -> str:
    """Format number compactly - remove trailing zeros and use optimal precision."""
    # Round to remove floating point noise
    value = round(value, 10)
    
    # Use scientific notation for very small numbers (avoid scientific notation output)
    if abs(value) < 0.0001 and value != 0:
        value = 0.0
    
    # Format with specified precision
    s = f'{value:.{max_precision}f}'
    
    # Remove trailing zeros
    s = s.rstrip('0').rstrip('.')
    
    # Handle negative zero
    if s == '-0':
        s = '0'
    
    return s


def simplify_bezier_curves(bezier_curves, tolerance: float = 0.5):
    """Simplify bezier curves by removing redundant control points.
    
    Uses Ramer-Douglas-Peucker-like algorithm adapted for Bezier curves.
    Returns simplified list of curves.
    """
    if len(bezier_curves) <= 2:
        return bezier_curves
    
    simplified = [bezier_curves[0]]
    
    for i in range(1, len(bezier_curves)):
        curr = bezier_curves[i]
        prev = simplified[-1]
        
        # Check if current curve adds significant detail
        # Approximate by checking control point deviation from straight line
        start = np.array([prev.p0.x, prev.p0.y])
        end = np.array([curr.p3.x, curr.p3.y])
        
        # Mid control points
        cp1 = np.array([prev.p1.x, prev.p1.y])
        cp2 = np.array([prev.p2.x, prev.p2.y])
        cp3 = np.array([curr.p1.x, curr.p1.y])
        cp4 = np.array([curr.p2.x, curr.p2.y])
        
        # Check deviation from straight line
        def point_line_distance(point, line_start, line_end):
            if np.all(line_start == line_end):
                return np.linalg.norm(point - line_start)
            return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        
        # If curves are roughly collinear, try to merge
        max_deviation = max(
            point_line_distance(cp1, start, end),
            point_line_distance(cp2, start, end),
            point_line_distance(cp3, start, end),
            point_line_distance(cp4, start, end)
        )
        
        if max_deviation < tolerance:
            # Merge curves - replace last curve with new combined one
            # Simple approach: just extend to current end point
            from vectorizer.types import BezierCurve
            merged = BezierCurve(
                p0=prev.p0,
                p1=prev.p1,
                p2=curr.p2,
                p3=curr.p3
            )
            simplified[-1] = merged
        else:
            simplified.append(curr)
    
    return simplified


def _color_to_hex(color: np.ndarray) -> str:
    """Convert RGB color to hex string."""
    return _color_to_hex_compact(color)


def _color_to_hex_compact(color: np.ndarray) -> str:
    """Convert RGB color to compact hex string with 3-digit shorthand when possible."""
    # Ensure color is in [0, 1] range
    if hasattr(color, 'max') and color.max() > 1.0:
        color = color / 255.0
    
    # Clamp to [0, 1]
    color = np.clip(color, 0, 1)
    
    # Convert to 0-255
    rgb = (color * 255).astype(int)
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Check if we can use 3-digit shorthand
    if (r % 17 == 0) and (g % 17 == 0) and (b % 17 == 0):
        # Use shorthand: #RGB
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    else:
        # Use full 6-digit: #RRGGBB
        return f'#{r:02x}{g:02x}{b:02x}'


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


def _create_gradient_def_compact(region: VectorRegion, grad_id: str) -> str:
    """Create compact gradient definition string."""
    if region.gradient_type == GradientType.LINEAR:
        if region.gradient_start and region.gradient_end:
            x1, y1 = region.gradient_start.x, region.gradient_start.y
            x2, y2 = region.gradient_end.x, region.gradient_end.y
            grad_def = f'<linearGradient id="{grad_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
        else:
            grad_def = f'<linearGradient id="{grad_id}">'
    elif region.gradient_type == GradientType.RADIAL:
        cx = region.gradient_center.x if region.gradient_center else '50%'
        cy = region.gradient_center.y if region.gradient_center else '50%'
        r = region.gradient_radius if region.gradient_radius else '50%'
        grad_def = f'<radialGradient id="{grad_id}" cx="{cx}" cy="{cy}" r="{r}">'
    else:
        grad_def = f'<linearGradient id="{grad_id}">'
    
    # Add color stops
    stops = []
    for stop in region.gradient_stops:
        offset = f'{stop.offset * 100:g}%'  # Use %g to remove trailing zeros
        color = _color_to_hex_compact(stop.color)
        stops.append(f'<stop offset="{offset}" stop-color="{color}"/>')
    
    return grad_def + ''.join(stops) + '</linearGradient>'


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


def generate_ultra_compressed_svg(regions, width, height):
    """Generate ultra-compressed SVG with maximum size reduction.
    
    WARNING: This may reduce visual quality. Use for maximum compression only.
    
    Applies:
    - 1-pixel coordinate quantization
    - 0 decimal precision (integers only)
    - Aggressive color merging (5% tolerance)
    - Path deduplication
    - Ultra-compact number formatting
    """
    from copy import deepcopy
    
    # Process regions with aggressive quantization
    processed_regions = []
    for region in regions:
        if not region.path:
            continue
        
        # Ultra-aggressive: 1px grid, simplify with tolerance
        simplified = simplify_bezier_curves(region.path, tolerance=1.0)
        quantized = quantize_coordinates(simplified, grid_size=1.0)
        
        new_region = deepcopy(region)
        new_region.path = quantized
        processed_regions.append(new_region)
    
    # Build ultra-compact SVG with 0 precision
    return _build_ultra_compact_svg(processed_regions, width, height)


def _build_ultra_compact_svg(regions, width, height):
    """Build SVG with maximum compression (0 decimal places)."""
    # Group by color with aggressive merging
    color_groups = {}
    for region in regions:
        if not region.path:
            continue
        
        if region.fill_color is not None:
            # Quantize color to reduce palette
            color = _quantize_color(region.fill_color, levels=16)
            color_hex = _color_to_hex_compact(color)
            if color_hex not in color_groups:
                color_groups[color_hex] = []
            color_groups[color_hex].append(region)
    
    # Build SVG
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    
    # Group paths by color
    for color, group in color_groups.items():
        if len(group) > 1:
            parts.append(f'<g fill="{color}">')
            for region in group:
                path_data = _bezier_to_svg_path_ultra(region.path)
                parts.append(f'<path d="{path_data}"/>')
            parts.append('</g>')
        else:
            region = group[0]
            path_data = _bezier_to_svg_path_ultra(region.path)
            parts.append(f'<path d="{path_data}" fill="{color}"/>')
    
    parts.append('</svg>')
    return ''.join(parts)


def _bezier_to_svg_path_ultra(bezier_curves):
    """Convert bezier curves to ultra-compact path (integers only)."""
    if not bezier_curves:
        return ''
    
    # Start at first point (as integers)
    p0 = bezier_curves[0].p0
    path_data = f'M{int(round(p0.x))},{int(round(p0.y))}'
    
    curr_x, curr_y = int(round(p0.x)), int(round(p0.y))
    
    for curve in bezier_curves:
        # Use relative coordinates (all integers)
        dx1 = int(round(curve.p1.x)) - curr_x
        dy1 = int(round(curve.p1.y)) - curr_y
        dx2 = int(round(curve.p2.x)) - int(round(curve.p1.x))
        dy2 = int(round(curve.p2.y)) - int(round(curve.p1.y))
        dx = int(round(curve.p3.x)) - int(round(curve.p2.x))
        dy = int(round(curve.p3.y)) - int(round(curve.p2.y))
        
        path_data += f'c{dx1},{dy1} {dx2},{dy2} {dx},{dy}'
        
        curr_x = int(round(curve.p3.x))
        curr_y = int(round(curve.p3.y))
    
    path_data += 'z'
    return path_data


def _quantize_color(color, levels=16):
    """Quantize color to reduce palette size."""
    if color.max() > 1.0:
        color = color / 255.0
    color = np.clip(color, 0, 1)
    
    # Quantize to specified levels
    step = 1.0 / (levels - 1)
    quantized = np.round(color / step) * step
    return np.clip(quantized, 0, 1)


def deduplicate_paths(regions):
    """Deduplicate identical path data across regions.
    
    Returns dict mapping path signatures to lists of regions with that path.
    """
    path_groups = {}
    
    for region in regions:
        if not region.path:
            continue
        
        # Create path signature from curve data
        sig_parts = []
        for curve in region.path:
            # Quantize to integers for comparison
            sig_parts.extend([
                int(round(curve.p0.x)), int(round(curve.p0.y)),
                int(round(curve.p1.x)), int(round(curve.p1.y)),
                int(round(curve.p2.x)), int(round(curve.p2.y)),
                int(round(curve.p3.x)), int(round(curve.p3.y)),
            ])
        
        sig = tuple(sig_parts)
        
        if sig not in path_groups:
            path_groups[sig] = []
        path_groups[sig].append(region)
    
    return path_groups


def merge_adjacent_paths(regions):
    """Merge adjacent paths with same fill color into single path.
    
    This reduces the number of path elements by combining them.
    """
    if not regions:
        return regions
    
    # Sort regions by color
    def get_color_key(r):
        if r.fill_color is not None:
            c = _quantize_color(r.fill_color, 16)
            return _color_to_hex_compact(c)
        return '#808080'
    
    sorted_regions = sorted(regions, key=get_color_key)
    
    merged = []
    current_color = None
    current_path = []
    
    for region in sorted_regions:
        if not region.path:
            continue
        
        color = get_color_key(region)
        
        if color == current_color:
            # Same color, append path
            current_path.extend(region.path)
        else:
            # Different color, save current and start new
            if current_path:
                from copy import deepcopy
                new_region = deepcopy(sorted_regions[0])  # Template
                new_region.path = current_path
                new_region.fill_color = regions[0].fill_color if regions else None
                merged.append(new_region)
            
            current_color = color
            current_path = list(region.path)
    
    # Don't forget the last group
    if current_path:
        from copy import deepcopy
        new_region = deepcopy(sorted_regions[0])
        new_region.path = current_path
        new_region.fill_color = regions[0].fill_color if regions else None
        merged.append(new_region)
    
    return merged


def generate_merged_svg(regions, width, height):
    """Generate SVG with merged adjacent paths of same color."""
    from copy import deepcopy
    
    # Process regions
    processed = []
    for region in regions:
        if not region.path:
            continue
        simplified = simplify_bezier_curves(region.path, tolerance=1.0)
        quantized = quantize_coordinates(simplified, grid_size=1.0)
        new_region = deepcopy(region)
        new_region.path = quantized
        processed.append(new_region)
    
    # Merge ALL same-color paths globally (not just adjacent)
    merged = merge_all_same_color_paths(processed)
    
    # Build SVG with minimal syntax
    # Remove xmlns (implied in HTML5), use single quotes, no spaces
    parts = [f"<svg viewBox='0 0 {width} {height}'>"]
    
    for region in merged:
        if not region.path:
            continue
        
        if region.fill_color is not None:
            color = _color_to_hex_compact(_quantize_color(region.fill_color, 16))
        else:
            color = '#808080'
        
        # Use ultra-compact path generation
        path_data = _bezier_to_svg_path_minimal(region.path)
        parts.append(f"<path d='{path_data}' fill='{color}'/>")
    
    parts.append('</svg>')
    return ''.join(parts)


def generate_extreme_svg(regions, width, height):
    """Generate extremely compressed SVG with maximum aggression.
    
    WARNING: Significant quality loss likely. Use only when size is critical.
    """
    from copy import deepcopy
    
    # Process with maximum aggression
    processed = []
    for region in regions:
        if not region.path:
            continue
        # Very aggressive simplification
        simplified = simplify_bezier_curves(region.path, tolerance=2.0)
        # 2px quantization
        quantized = quantize_coordinates(simplified, grid_size=2.0)
        new_region = deepcopy(region)
        new_region.path = quantized
        processed.append(new_region)
    
    # Merge with only 8 color levels (very aggressive)
    merged = merge_all_same_color_paths(processed, color_levels=8)
    
    # Build minimal SVG
    parts = [f"<svg viewBox='0 0 {width} {height}'>"]
    
    for region in merged:
        if not region.path:
            continue
        
        if region.fill_color is not None:
            color = _color_to_hex_compact(_quantize_color(region.fill_color, 8))
        else:
            color = '#808080'
        
        path_data = _bezier_to_svg_path_minimal(region.path)
        parts.append(f"<path d='{path_data}' fill='{color}'/>")
    
    parts.append('</svg>')
    return ''.join(parts)


def generate_insane_svg(regions, width, height):
    """Generate INSANE compression SVG with maximum possible reduction.
    
    WARNING: MASSIVE quality loss expected. Only for extreme size constraints.
    Uses 4-level color palette and 4px grid.
    """
    from copy import deepcopy
    
    # Maximum aggression processing
    processed = []
    for region in regions:
        if not region.path:
            continue
        # Extreme simplification
        simplified = simplify_bezier_curves(region.path, tolerance=4.0)
        # 4px quantization - very coarse
        quantized = quantize_coordinates(simplified, grid_size=4.0)
        new_region = deepcopy(region)
        new_region.path = quantized
        processed.append(new_region)
    
    # Merge with only 4 color levels (insane)
    merged = merge_all_same_color_paths(processed, color_levels=4)
    
    # Build minimal SVG
    parts = [f"<svg viewBox='0 0 {width} {height}'>"]
    
    for region in merged:
        if not region.path:
            continue
        
        if region.fill_color is not None:
            color = _color_to_hex_compact(_quantize_color(region.fill_color, 4))
        else:
            color = '#888'
        
        path_data = _bezier_to_svg_path_minimal(region.path)
        parts.append(f"<path d='{path_data}' fill='{color}'/>")
    
    parts.append('</svg>')
    return ''.join(parts)


def merge_all_same_color_paths(regions, color_levels=8):
    """Merge ALL paths with same color globally (not just adjacent).
    
    Args:
        regions: List of regions
        color_levels: Number of color quantization levels (8 = aggressive, 16 = normal)
    """
    # Group by quantized color
    color_groups = {}
    
    for region in regions:
        if not region.path:
            continue
        
        if region.fill_color is not None:
            color = _color_to_hex_compact(_quantize_color(region.fill_color, color_levels))
        else:
            color = '#808080'
        
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].extend(region.path)
    
    # Create merged regions
    merged = []
    for color, path in color_groups.items():
        from vectorizer.types import VectorRegion, RegionKind
        import numpy as np
        
        region = VectorRegion(
            kind=RegionKind.FLAT,
            path=path,
            fill_color=np.array([0.5, 0.5, 0.5])  # Placeholder
        )
        merged.append(region)
    
    return merged


def _bezier_to_svg_path_minimal(bezier_curves):
    """Convert to absolute minimal path data (no spaces, single quotes)."""
    if not bezier_curves:
        return ''
    
    # Use integers only, no spaces, compact format
    p0 = bezier_curves[0].p0
    path_data = f'M{int(round(p0.x))},{int(round(p0.y))}'
    
    curr_x, curr_y = int(round(p0.x)), int(round(p0.y))
    
    for curve in bezier_curves:
        # Relative coordinates
        p1x, p1y = int(round(curve.p1.x)), int(round(curve.p1.y))
        p2x, p2y = int(round(curve.p2.x)), int(round(curve.p2.y))
        p3x, p3y = int(round(curve.p3.x)), int(round(curve.p3.y))
        
        dx1, dy1 = p1x - curr_x, p1y - curr_y
        dx2, dy2 = p2x - p1x, p2y - p1y
        dx, dy = p3x - p2x, p3y - p2y
        
        # No spaces between numbers, comma is separator
        path_data += f'c{dx1},{dy1},{dx2},{dy2},{dx},{dy}'
        
        curr_x, curr_y = p3x, p3y
    
    path_data += 'z'
    return path_data


def generate_symbol_optimized_svg(regions, width, height):
    """Generate SVG with symbol reuse for duplicate paths.
    
    Uses SVG <symbol> and <use> elements to avoid repeating identical path data.
    Best for images with many repeated shapes (patterns, icons, etc.).
    """
    from copy import deepcopy
    
    # Process regions with standard quantization
    processed = []
    for region in regions:
        if not region.path:
            continue
        simplified = simplify_bezier_curves(region.path, tolerance=0.5)
        quantized = quantize_coordinates(simplified, grid_size=1.0)
        new_region = deepcopy(region)
        new_region.path = quantized
        processed.append(new_region)
    
    # Find duplicate paths
    path_groups = deduplicate_paths(processed)
    
    # Separate unique paths from duplicates
    unique_paths = []
    duplicate_groups = {}
    symbol_id = 0
    
    for sig, group in path_groups.items():
        if len(group) > 1:
            # Multiple regions share this path
            sid = _short_id(symbol_id)
            symbol_id += 1
            duplicate_groups[sid] = {
                'path': group[0].path,
                'regions': group
            }
        else:
            unique_paths.append(group[0])
    
    # Build SVG with symbols
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    
    # Define symbols
    if duplicate_groups:
        parts.append('<defs>')
        for sid, data in duplicate_groups.items():
            path_data = _bezier_to_svg_path_ultra(data['path'])
            parts.append(f'<symbol id="{sid}" viewBox="0 0 {width} {height}"><path d="{path_data}"/></symbol>')
        parts.append('</defs>')
    
    # Group by fill color
    color_groups = {}
    for region in unique_paths:
        if region.fill_color is not None:
            color = _color_to_hex_compact(_quantize_color(region.fill_color, 16))
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(region)
    
    # Output unique paths
    for color, group in color_groups.items():
        if len(group) > 1:
            parts.append(f'<g fill="{color}">')
            for region in group:
                path_data = _bezier_to_svg_path_ultra(region.path)
                parts.append(f'<path d="{path_data}"/>')
            parts.append('</g>')
        else:
            path_data = _bezier_to_svg_path_ultra(group[0].path)
            parts.append(f'<path d="{path_data}" fill="{color}"/>')
    
    # Output uses of duplicate paths
    for sid, data in duplicate_groups.items():
        # Group by color
        color_groups = {}
        for region in data['regions']:
            if region.fill_color is not None:
                color = _color_to_hex_compact(_quantize_color(region.fill_color, 16))
                if color not in color_groups:
                    color_groups[color] = []
                color_groups[color].append(region)
        
        for color, group in color_groups.items():
            if len(group) > 1:
                parts.append(f'<g fill="{color}">')
                for _ in group:
                    parts.append(f'<use href="#{sid}"/>')
                parts.append('</g>')
            else:
                parts.append(f'<use href="#{sid}" fill="{color}"/>')
    
    parts.append('</svg>')
    return ''.join(parts)


def quantize_coordinates(bezier_curves, grid_size: float = 0.5):
    """Quantize coordinates to grid to improve compressibility.
    
    Rounds all coordinates to nearest grid point, which:
    1. Reduces precision needed to represent coordinates
    2. Makes values more repetitive (better gzip compression)
    3. Can maintain visual quality if grid is fine enough
    
    Args:
        bezier_curves: List of BezierCurve objects
        grid_size: Size of quantization grid (default 0.5 pixels)
        
    Returns:
        List of quantized BezierCurve objects
    """
    from vectorizer.types import BezierCurve, Point
    
    def quantize(val):
        return round(val / grid_size) * grid_size
    
    quantized = []
    for curve in bezier_curves:
        q_curve = BezierCurve(
            p0=Point(quantize(curve.p0.x), quantize(curve.p0.y)),
            p1=Point(quantize(curve.p1.x), quantize(curve.p1.y)),
            p2=Point(quantize(curve.p2.x), quantize(curve.p2.y)),
            p3=Point(quantize(curve.p3.x), quantize(curve.p3.y))
        )
        quantized.append(q_curve)
    
    return quantized


def optimize_precision_adaptive(bezier_curves, base_precision: int = 2):
    """Adaptively optimize decimal precision for each coordinate.
    
    Uses lower precision for coordinates that are already close to integers.
    
    Returns:
        Optimized path string
    """
    if not bezier_curves:
        return ''
    
    # Collect all coordinate values
    values = []
    for curve in bezier_curves:
        for point in [curve.p0, curve.p1, curve.p2, curve.p3]:
            values.extend([point.x, point.y])
    
    # Analyze decimal parts to find optimal precision
    # If most values are integers or half-integers, we can use lower precision
    decimal_parts = [abs(v - round(v)) for v in values]
    avg_decimal = sum(decimal_parts) / len(decimal_parts) if decimal_parts else 0
    
    # Adjust precision based on average decimal part
    if avg_decimal < 0.05:
        precision = 0  # Mostly integers
    elif avg_decimal < 0.15:
        precision = 1  # Mostly half-integers
    else:
        precision = base_precision
    
    return _bezier_to_svg_path(bezier_curves, precision=precision, use_relative=True)


def generate_optimized_svg(regions, width, height, 
                          quantization_grid: float = 0.5,
                          simplify_tolerance: float = 0.5,
                          base_precision: int = 2) -> str:
    """Generate fully optimized SVG with all compression techniques applied.
    
    Applies:
    1. Path simplification
    2. Coordinate quantization  
    3. Adaptive precision
    4. Relative coordinates
    5. Compact number formatting
    6. Color grouping
    7. Minified XML output
    
    Args:
        regions: List of VectorRegion objects
        width: SVG width
        height: SVG height
        quantization_grid: Grid size for coordinate quantization
        simplify_tolerance: Tolerance for path simplification
        base_precision: Base decimal precision
        
    Returns:
        Optimized SVG string
    """
    from copy import deepcopy
    
    # Process regions
    processed_regions = []
    for region in regions:
        if not region.path:
            continue
        
        # Simplify curves
        simplified = simplify_bezier_curves(region.path, tolerance=simplify_tolerance)
        
        # Quantize coordinates
        quantized = quantize_coordinates(simplified, grid_size=quantization_grid)
        
        # Create new region with optimized path
        new_region = deepcopy(region)
        new_region.path = quantized
        processed_regions.append(new_region)
    
    # Generate compact SVG
    return _regions_to_svg_compact(processed_regions, width, height, base_precision)
