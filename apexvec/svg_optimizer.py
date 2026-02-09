"""SVG generation and optimization with curvature preservation."""
from typing import List, Tuple
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

from apexvec.types import VectorRegion, RegionKind, GradientType, BezierCurve, Point


# ============================================================================
# CURVATURE COMPUTATION
# ============================================================================

def compute_bezier_curvature(curve: BezierCurve, t: float) -> float:
    """
    Compute curvature of a cubic Bezier curve at parameter t.
    
    Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    
    Returns:
        Curvature value (0 for straight line, higher for tighter curves)
    """
    # Cubic Bezier: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    # First derivative
    mt = 1 - t
    
    # B'(t) = 3(1-t)^2 (P1-P0) + 6(1-t)t (P2-P1) + 3t^2 (P3-P2)
    dx = 3 * mt**2 * (curve.p1.x - curve.p0.x) + \
         6 * mt * t * (curve.p2.x - curve.p1.x) + \
         3 * t**2 * (curve.p3.x - curve.p2.x)
    dy = 3 * mt**2 * (curve.p1.y - curve.p0.y) + \
         6 * mt * t * (curve.p2.y - curve.p1.y) + \
         3 * t**2 * (curve.p3.y - curve.p2.y)
    
    # Second derivative
    # B''(t) = 6(1-t) (P2-2P1+P0) + 6t (P3-2P2+P1)
    ddx = 6 * mt * (curve.p2.x - 2*curve.p1.x + curve.p0.x) + \
          6 * t * (curve.p3.x - 2*curve.p2.x + curve.p1.x)
    ddy = 6 * mt * (curve.p2.y - 2*curve.p1.y + curve.p0.y) + \
          6 * t * (curve.p3.y - 2*curve.p2.y + curve.p1.y)
    
    # Curvature formula
    denom = (dx**2 + dy**2)**1.5
    if denom < 1e-10:
        return 0.0
    
    curvature = abs(dx * ddy - dy * ddx) / denom
    return curvature


def compute_curve_max_curvature(curve: BezierCurve, num_samples: int = 10) -> float:
    """Compute maximum curvature along a Bezier curve."""
    max_curv = 0.0
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0
        curv = compute_bezier_curvature(curve, t)
        max_curv = max(max_curv, curv)
    return max_curv


def check_g2_continuity(curve1: BezierCurve, curve2: BezierCurve, tolerance: float = 0.1) -> bool:
    """
    Check if two Bezier curves have G² continuity at their join.
    
    G² continuity requires:
    1. Position continuity (end of curve1 == start of curve2)
    2. Tangent continuity (first derivatives match direction)
    3. Curvature continuity (curvatures match)
    
    Returns:
        True if curves are G² continuous
    """
    # Check position continuity
    pos_diff = np.sqrt((curve1.p3.x - curve2.p0.x)**2 + (curve1.p3.y - curve2.p0.y)**2)
    if pos_diff > 0.1:
        return False
    
    # Compute end tangent of curve1 (at t=1)
    # B'(1) = 3(P3 - P2)
    tangent1_x = 3 * (curve1.p3.x - curve1.p2.x)
    tangent1_y = 3 * (curve1.p3.y - curve1.p2.y)
    
    # Compute start tangent of curve2 (at t=0)
    # B'(0) = 3(P1 - P0)
    tangent2_x = 3 * (curve2.p1.x - curve2.p0.x)
    tangent2_y = 3 * (curve2.p1.y - curve2.p0.y)
    
    # Check tangent direction
    len1 = np.sqrt(tangent1_x**2 + tangent1_y**2)
    len2 = np.sqrt(tangent2_x**2 + tangent2_y**2)
    
    if len1 < 1e-10 or len2 < 1e-10:
        return True  # Zero tangent, can't check direction
    
    # Normalize and check dot product
    tx1, ty1 = tangent1_x / len1, tangent1_y / len1
    tx2, ty2 = tangent2_x / len2, tangent2_y / len2
    
    dot = tx1 * tx2 + ty1 * ty2
    if dot < 0.95:  # Not collinear (allow 18° deviation)
        return False
    
    # Check curvature continuity
    curv1 = compute_bezier_curvature(curve1, 1.0)
    curv2 = compute_bezier_curvature(curve2, 0.0)
    
    if curv1 > 0.01 or curv2 > 0.01:  # Only check if significant curvature
        curv_diff = abs(curv1 - curv2) / max(curv1, curv2, 1e-10)
        if curv_diff > tolerance:
            return False
    
    return True


# ============================================================================
# CURVE-AWARE QUANTIZATION
# ============================================================================

def quantize_curvature_aware(bezier_curves: List[BezierCurve], base_grid: float = 0.25) -> List[BezierCurve]:
    """
    Quantize coordinates with adaptive grid based on curvature.
    
    Finer quantization where curvature is high to preserve smooth curves,
    coarser where nearly straight.
    
    Args:
        bezier_curves: List of BezierCurve objects
        base_grid: Base grid size (default 0.25 for high quality)
        
    Returns:
        List of quantized BezierCurve objects
    """
    quantized = []
    
    for curve in bezier_curves:
        # Compute maximum curvature
        max_curv = compute_curve_max_curvature(curve, num_samples=5)
        
        # Adaptive grid: finer where curved
        if max_curv > 0.1:  # High curvature
            grid = base_grid * 0.25  # 0.0625px precision (4x finer)
        elif max_curv > 0.01:  # Medium curvature
            grid = base_grid  # 0.25px (standard)
        else:  # Nearly straight
            grid = base_grid * 2.0  # 0.5px (can be coarser)
        
        def quantize(val):
            return round(val / grid) * grid
        
        q_curve = BezierCurve(
            p0=Point(quantize(curve.p0.x), quantize(curve.p0.y)),
            p1=Point(quantize(curve.p1.x), quantize(curve.p1.y)),
            p2=Point(quantize(curve.p2.x), quantize(curve.p2.y)),
            p3=Point(quantize(curve.p3.x), quantize(curve.p3.y))
        )
        quantized.append(q_curve)
    
    return quantized


def quantize_coordinates(bezier_curves: List[BezierCurve], grid_size: float = 0.5) -> List[BezierCurve]:
    """
    Quantize coordinates to grid (legacy function - prefer curvature-aware).
    
    Kept for backward compatibility but now uses curvature-aware internally.
    """
    return quantize_curvature_aware(bezier_curves, base_grid=grid_size)


# ============================================================================
# G²-PRESERVING SIMPLIFICATION
# ============================================================================

def compute_curvature_error(original_curve: BezierCurve, simplified_curve: BezierCurve) -> float:
    """
    Compute curvature error between original and simplified curve.
    
    Returns maximum curvature difference at sample points.
    """
    errors = []
    for i in range(5):
        t = i / 4
        curv_orig = compute_bezier_curvature(original_curve, t)
        curv_simp = compute_bezier_curvature(simplified_curve, t)
        errors.append(abs(curv_orig - curv_simp))
    return max(errors) if errors else 0.0


def simplify_g2_preserving(bezier_curves: List[BezierCurve], tolerance: float = 0.5) -> List[BezierCurve]:
    """
    Simplify bezier curves while preserving G² continuity.
    
    Only simplifies if:
    1. Curves meet G² continuity at join
    2. Curvature error is within tolerance
    3. Geometric deviation is small
    
    Args:
        bezier_curves: List of BezierCurve objects
        tolerance: Maximum allowed curvature error
        
    Returns:
        Simplified list of curves
    """
    if len(bezier_curves) <= 2:
        return bezier_curves
    
    simplified = [bezier_curves[0]]
    
    for i in range(1, len(bezier_curves)):
        curr = bezier_curves[i]
        prev = simplified[-1]
        
        # Check G² continuity first
        if not check_g2_continuity(prev, curr, tolerance=0.2):
            # Not G² continuous, keep separate
            simplified.append(curr)
            continue
        
        # Try to merge curves
        start = np.array([prev.p0.x, prev.p0.y])
        end = np.array([curr.p3.x, curr.p3.y])
        
        # Control points
        cp1 = np.array([prev.p1.x, prev.p1.y])
        cp2 = np.array([prev.p2.x, prev.p2.y])
        cp3 = np.array([curr.p1.x, curr.p1.y])
        cp4 = np.array([curr.p2.x, curr.p2.y])
        
        # Check geometric deviation from straight line
        def point_line_distance(point, line_start, line_end):
            if np.all(line_start == line_end):
                return np.linalg.norm(point - line_start)
            return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)
        
        max_deviation = max(
            point_line_distance(cp1, start, end),
            point_line_distance(cp2, start, end),
            point_line_distance(cp3, start, end),
            point_line_distance(cp4, start, end)
        )
        
        # Create merged curve
        merged = BezierCurve(
            p0=prev.p0,
            p1=prev.p1,
            p2=curr.p2,
            p3=curr.p3
        )
        
        # Check curvature preservation
        curvature_err = compute_curvature_error(prev, merged)
        curvature_err += compute_curvature_error(curr, merged)
        
        if max_deviation < tolerance and curvature_err < tolerance * 0.5:
            # Accept merge
            simplified[-1] = merged
        else:
            simplified.append(curr)
    
    return simplified


def simplify_bezier_curves(bezier_curves: List[BezierCurve], tolerance: float = 0.5) -> List[BezierCurve]:
    """
    Simplify bezier curves (legacy - now uses G²-preserving version).
    """
    return simplify_g2_preserving(bezier_curves, tolerance=tolerance)


# ============================================================================
# GRADIENT-AWARE COLOR OPTIMIZATION
# ============================================================================

def optimize_region_colors(regions: List[VectorRegion], flat_levels: int = 256) -> List[VectorRegion]:
    """
    Optimize colors with gradient awareness.
    
    Preserves full precision for gradient regions while allowing quantization
    for flat regions.
    
    Args:
        regions: List of VectorRegion objects
        flat_levels: Number of quantization levels for flat regions (256 = full)
        
    Returns:
        Regions with optimized colors
    """
    from copy import deepcopy
    
    optimized = []
    for region in regions:
        new_region = deepcopy(region)
        
        if region.kind == RegionKind.GRADIENT:
            # Keep full precision for gradient regions
            # Don't quantize gradient stops
            pass
        elif region.kind == RegionKind.FLAT and region.fill_color is not None:
            # Can quantize flat regions
            new_region.fill_color = _quantize_color_safe(region.fill_color, levels=flat_levels)
        elif region.kind == RegionKind.DETAIL and region.mesh_colors is not None:
            # Keep detail region colors but can quantize slightly
            new_region.mesh_colors = np.array([
                _quantize_color_safe(color, levels=flat_levels)
                for color in region.mesh_colors
            ])
        
        optimized.append(new_region)
    
    return optimized


def _quantize_color_safe(color: np.ndarray, levels: int = 256) -> np.ndarray:
    """Quantize color safely, preserving full range for levels=256."""
    if levels >= 256:
        return color  # No quantization
    
    if color.max() > 1.0:
        color = color / 255.0
    color = np.clip(color, 0, 1)
    
    step = 1.0 / (levels - 1)
    quantized = np.round(color / step) * step
    return np.clip(quantized, 0, 1)


# ============================================================================
# OPTIMIZATION PRESETS
# ============================================================================

class OptimizationPreset:
    """Configuration for different optimization levels."""
    
    def __init__(self, name: str, quantization: float, simplification: float, 
                 color_levels: int, use_curvature_aware: bool = True):
        self.name = name
        self.quantization = quantization  # Base grid size
        self.simplification = simplification  # Tolerance
        self.color_levels = color_levels
        self.use_curvature_aware = use_curvature_aware


# Preset configurations
PRESETS = {
    'lossless': OptimizationPreset(
        name='lossless',
        quantization=0.0,  # No quantization
        simplification=0.0,  # No simplification
        color_levels=256,  # Full color
        use_curvature_aware=False
    ),
    'standard': OptimizationPreset(
        name='standard',
        quantization=0.25,  # Fine grid
        simplification=0.5,  # Moderate simplification
        color_levels=256,  # Full color
        use_curvature_aware=True
    ),
    'compact': OptimizationPreset(
        name='compact',
        quantization=0.5,  # Standard grid
        simplification=1.0,  # More simplification
        color_levels=128,  # Reduced colors
        use_curvature_aware=True
    ),
    'thumbnail': OptimizationPreset(
        name='thumbnail',
        quantization=1.0,  # Coarse grid
        simplification=2.0,  # Aggressive simplification
        color_levels=64,  # Few colors
        use_curvature_aware=True
    )
}


def apply_optimization_preset(regions: List[VectorRegion], preset: OptimizationPreset) -> List[VectorRegion]:
    """Apply an optimization preset to regions."""
    from copy import deepcopy
    
    processed = []
    
    for region in regions:
        if not region.path:
            continue
        
        new_region = deepcopy(region)
        curves = region.path
        
        # Apply simplification if tolerance > 0
        if preset.simplification > 0:
            curves = simplify_g2_preserving(curves, tolerance=preset.simplification)
        
        # Apply quantization if grid > 0
        if preset.quantization > 0:
            if preset.use_curvature_aware:
                curves = quantize_curvature_aware(curves, base_grid=preset.quantization)
            else:
                curves = quantize_coordinates(curves, grid_size=preset.quantization)
        
        new_region.path = curves
        processed.append(new_region)
    
    # Apply color optimization
    if preset.color_levels < 256:
        processed = optimize_region_colors(processed, flat_levels=preset.color_levels)
    
    return processed


# ============================================================================
# SVG GENERATION
# ============================================================================

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
    """Generate compact (minified) SVG output with safe optimizations."""
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
    
    # Build compact SVG string manually
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">']
    
    # Add gradient definitions if needed with shortest IDs
    if gradients:
        parts.append('<defs>')
        for i, region in enumerate(gradients):
            grad_id = _short_id(i)
            parts.append(_create_gradient_def_compact(region, grad_id))
        parts.append('</defs>')
    
    # Output paths grouped by fill color
    for color, group in color_groups.items():
        if len(group) > 1:
            parts.append(f'<g fill="{color}">')
            for region in group:
                path_data = _bezier_to_svg_path(region.path, precision, use_relative=True)
                parts.append(f'<path d="{path_data}"/>')
            parts.append('</g>')
        else:
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


def _color_to_hex_compact(color: np.ndarray) -> str:
    """Convert RGB color to compact hex string with 3-digit shorthand when possible."""
    if color.max() > 1.0:
        color = color / 255.0
    color = np.clip(color, 0, 1)
    rgb = (color * 255).astype(int)
    r, g, b = rgb[0], rgb[1], rgb[2]
    
    if (r % 17 == 0) and (g % 17 == 0) and (b % 17 == 0):
        return f'#{r//17:x}{g//17:x}{b//17:x}'
    else:
        return f'#{r:02x}{g:02x}{b:02x}'


def _bezier_to_svg_path(bezier_curves, precision: int = 2, use_relative: bool = True) -> str:
    """Convert bezier curves to SVG path data string."""
    if not bezier_curves:
        return ''
    
    fmt = f'{{:.{precision}f}}'
    
    p0 = bezier_curves[0].p0
    path_data = f'M{fmt.format(p0.x)},{fmt.format(p0.y)}'
    
    curr_x, curr_y = p0.x, p0.y
    
    for curve in bezier_curves:
        if use_relative:
            dx1 = curve.p1.x - curr_x
            dy1 = curve.p1.y - curr_y
            dx2 = curve.p2.x - curve.p1.x
            dy2 = curve.p2.y - curve.p1.y
            dx = curve.p3.x - curve.p2.x
            dy = curve.p3.y - curve.p2.y
            
            path_data += f'c{_fmt_compact(dx1)},{_fmt_compact(dy1)} {_fmt_compact(dx2)},{_fmt_compact(dy2)} {_fmt_compact(dx)},{_fmt_compact(dy)}'
            
            curr_x, curr_y = curve.p3.x, curve.p3.y
        else:
            path_data += (
                f'C{fmt.format(curve.p1.x)},{fmt.format(curve.p1.y)} '
                f'{fmt.format(curve.p2.x)},{fmt.format(curve.p2.y)} '
                f'{fmt.format(curve.p3.x)},{fmt.format(curve.p3.y)}'
            )
    
    path_data += 'z'
    return path_data


def _fmt_compact(value: float, max_precision: int = 2) -> str:
    """Format number compactly - remove trailing zeros."""
    value = round(value, 10)
    
    if abs(value) < 0.0001 and value != 0:
        value = 0.0
    
    s = f'{value:.{max_precision}f}'
    s = s.rstrip('0').rstrip('.')
    
    if s == '-0':
        s = '0'
    
    return s


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
    
    stops = []
    for stop in region.gradient_stops:
        offset = f'{stop.offset * 100:g}%'
        color = _color_to_hex_compact(stop.color)
        stops.append(f'<stop offset="{offset}" stop-color="{color}"/>')
    
    return grad_def + ''.join(stops) + '</linearGradient>'


def _regions_to_svg_pretty(regions, width, height, precision):
    """Generate pretty-printed SVG output."""
    svg = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('width', str(width))
    svg.set('height', str(height))
    svg.set('viewBox', f'0 0 {width} {height}')
    
    defs = ET.SubElement(svg, 'defs')
    gradient_id = 0
    
    for region in regions:
        if not region.path:
            continue
        
        path_data = _bezier_to_svg_path(region.path, precision, use_relative=False)
        path_elem = ET.SubElement(svg, 'path')
        path_elem.set('d', path_data)
        
        if region.kind == RegionKind.FLAT and region.fill_color is not None:
            color = _color_to_hex_compact(region.fill_color)
            path_elem.set('fill', color)
        elif region.kind == RegionKind.GRADIENT and region.gradient_type is not None:
            grad_id = f'gradient_{gradient_id}'
            gradient_id += 1
            _create_gradient_def(defs, region, grad_id)
            path_elem.set('fill', f'url(#{grad_id})')
        elif region.kind == RegionKind.DETAIL and region.mesh_triangles is not None:
            avg_color = np.mean(region.mesh_colors, axis=0)
            path_elem.set('fill', _color_to_hex_compact(avg_color))
        else:
            path_elem.set('fill', '#808080')
    
    svg_string = ET.tostring(svg, encoding='unicode')
    dom = minidom.parseString(svg_string)
    pretty_xml = dom.toprettyxml(indent='  ')
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    
    return '\n'.join(lines)


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
    elif region.gradient_type == GradientType.RADIAL:
        grad = ET.SubElement(defs, 'radialGradient')
        grad.set('id', grad_id)
        if region.gradient_center:
            grad.set('cx', str(region.gradient_center.x))
            grad.set('cy', str(region.gradient_center.y))
        if region.gradient_radius:
            grad.set('r', str(region.gradient_radius))
    else:
        grad = ET.SubElement(defs, 'linearGradient')
        grad.set('id', grad_id)
    
    for stop in region.gradient_stops:
        stop_elem = ET.SubElement(grad, 'stop')
        stop_elem.set('offset', f'{stop.offset * 100:.1f}%')
        stop_elem.set('stop-color', _color_to_hex_compact(stop.color))


# ============================================================================
# LEGACY FUNCTIONS (DEPRECATED - KEEP FOR COMPATIBILITY)
# ============================================================================

def generate_ultra_compressed_svg(regions, width, height):
    """DEPRECATED: Use generate_optimized_svg with 'compact' preset instead."""
    return generate_optimized_svg(regions, width, height, preset_name='compact')


def generate_extreme_svg(regions, width, height):
    """DEPRECATED: Use generate_optimized_svg with 'thumbnail' preset instead."""
    return generate_optimized_svg(regions, width, height, preset_name='thumbnail')


def generate_merged_svg(regions, width, height):
    """DEPRECATED: Use standard optimization instead."""
    return generate_optimized_svg(regions, width, height, preset_name='standard')


def generate_symbol_optimized_svg(regions, width, height):
    """DEPRECATED: Symbol optimization disabled for quality preservation."""
    return generate_optimized_svg(regions, width, height, preset_name='standard')


def generate_insane_svg(regions, width, height):
    """REMOVED: This mode caused unacceptable quality loss. Use 'thumbnail' preset."""
    raise NotImplementedError(
        "generate_insane_svg has been removed due to quality issues. "
        "Use generate_optimized_svg with preset='thumbnail' for maximum compression."
    )


def generate_monochrome_svg(regions, width, height):
    """REMOVED: This mode destroyed all color information."""
    raise NotImplementedError(
        "generate_monochrome_svg has been removed due to quality issues. "
        "Use external tools for monochrome conversion."
    )


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def generate_optimized_svg(
    regions,
    width,
    height,
    preset_name: str = 'standard',
    quantization_grid: float = None,
    simplify_tolerance: float = None,
    base_precision: int = 2
) -> str:
    """
    Generate optimized SVG with curvature-preserving compression.
    
    Args:
        regions: List of VectorRegion objects
        width: SVG width
        height: SVG height
        preset_name: One of 'lossless', 'standard', 'compact', 'thumbnail'
        quantization_grid: Override preset quantization (optional)
        simplify_tolerance: Override preset simplification (optional)
        base_precision: Decimal precision for coordinates (default 2)
        
    Returns:
        Optimized SVG string
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Use: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    # Allow parameter overrides
    if quantization_grid is not None:
        preset = OptimizationPreset(
            name=preset.name,
            quantization=quantization_grid,
            simplification=preset.simplification,
            color_levels=preset.color_levels,
            use_curvature_aware=preset.use_curvature_aware
        )
    
    if simplify_tolerance is not None:
        preset = OptimizationPreset(
            name=preset.name,
            quantization=preset.quantization,
            simplification=simplify_tolerance,
            color_levels=preset.color_levels,
            use_curvature_aware=preset.use_curvature_aware
        )
    
    # Apply optimizations
    processed = apply_optimization_preset(regions, preset)
    
    # Generate SVG
    return _regions_to_svg_compact(processed, width, height, base_precision)


def get_svg_size(svg_string: str) -> int:
    """Get size of SVG in bytes."""
    return len(svg_string.encode('utf-8'))


def optimize_svg(svg_string: str) -> str:
    """Optimize SVG by removing unnecessary whitespace."""
    root = ET.fromstring(svg_string)
    _remove_whitespace(root)
    return ET.tostring(root, encoding='unicode')


def _remove_whitespace(element):
    """Remove whitespace-only text nodes from XML tree."""
    if element.text and not element.text.strip():
        element.text = None
    if element.tail and not element.tail.strip():
        element.tail = None
    for child in element:
        _remove_whitespace(child)
