"""SVG export for poster-style vectorization."""
from typing import List, Tuple, Optional
import numpy as np

from apexvec.types import BezierPath, BezierCurve, Region, ApexConfig


def format_color(rgb: np.ndarray) -> str:
    """
    Format RGB color as hex string.
    
    Uses #RGB shorthand when possible.
    
    Args:
        rgb: RGB array with values in [0, 1]
        
    Returns:
        Hex color string
    """
    # Convert to 0-255 range
    r, g, b = [int(min(255, max(0, c * 255))) for c in rgb]
    
    # Check if we can use shorthand #RGB
    if (r % 17 == 0) and (g % 17 == 0) and (b % 17 == 0):
        # Can use shorthand
        return f"#{r//17:x}{g//17:x}{b//17:x}"
    else:
        # Full hex
        return f"#{r:02x}{g:02x}{b:02x}"


def format_number(x: float, precision: int) -> str:
    """
    Format number with given precision.
    
    Args:
        x: Number to format
        precision: Decimal places
        
    Returns:
        Formatted string
    """
    formatted = f"{x:.{precision}f}"
    # Remove trailing zeros and decimal point if not needed
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def bezier_to_path_command(
    curve: BezierCurve,
    prev_point: Optional[Tuple[float, float]] = None,
    precision: int = 2
) -> str:
    """
    Convert BezierCurve to SVG path command.
    
    Uses relative cubic bezier 'c' command.
    
    Args:
        curve: Bezier curve
        prev_point: Previous point (for relative coordinates)
        precision: Decimal precision
        
    Returns:
        SVG path command string
    """
    fmt = lambda x: format_number(x, precision)
    
    if prev_point is None:
        # First segment, use absolute M command
        cmds = [f"M{fmt(curve.p0.x)},{fmt(curve.p0.y)}"]
        prev = (curve.p0.x, curve.p0.y)
    else:
        cmds = []
        prev = prev_point
    
    # Relative control points and end point
    dx1 = curve.p1.x - prev[0]
    dy1 = curve.p1.y - prev[1]
    dx2 = curve.p2.x - prev[0]
    dy2 = curve.p2.y - prev[1]
    dx3 = curve.p3.x - prev[0]
    dy3 = curve.p3.y - prev[1]
    
    # Relative cubic bezier command
    cmds.append(
        f"c{fmt(dx1)},{fmt(dy1)} {fmt(dx2)},{fmt(dy2)} {fmt(dx3)},{fmt(dy3)}"
    )
    
    return ' '.join(cmds)


def bezier_path_to_svg(
    path: BezierPath,
    fill_color: np.ndarray,
    precision: int = 2
) -> str:
    """
    Convert BezierPath to SVG path element.
    
    Args:
        path: Bezier path
        fill_color: RGB fill color
        precision: Decimal precision
        
    Returns:
        SVG path element string
    """
    if not path.curves:
        return ""
    
    # Build path commands
    commands = []
    prev_point = None
    
    for curve in path.curves:
        cmd = bezier_to_path_command(curve, prev_point, precision)
        commands.append(cmd)
        prev_point = (curve.p3.x, curve.p3.y)
    
    # Close path if needed
    if path.is_closed:
        commands.append("Z")
    
    path_data = ' '.join(commands)
    color = format_color(fill_color)
    
    return f'<path d="{path_data}" fill="{color}"/>'


def generate_svg(
    paths: List[BezierPath],
    regions: List[Region],
    width: int,
    height: int,
    config: ApexConfig
) -> str:
    """
    Generate SVG from regions and paths.
    
    Args:
        paths: List of BezierPath objects
        regions: List of Region objects (for colors)
        width: Image width
        height: Image height
        config: Configuration
        
    Returns:
        Complete SVG string
    """
    # Build SVG content
    path_elements = []
    
    for path, region in zip(paths, regions):
        if not path.curves:
            continue
            
        fill_color = region.mean_color
        if fill_color is None:
            fill_color = np.array([0.5, 0.5, 0.5])
        
        path_elem = bezier_path_to_svg(path, fill_color, config.precision)
        if path_elem:
            path_elements.append(path_elem)
    
    # Assemble SVG
    svg_content = '\n  '.join(path_elements)
    
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  {svg_content}
</svg>'''
    
    return svg


def save_svg(
    svg_string: str,
    output_path: str
) -> None:
    """
    Save SVG string to file.
    
    Args:
        svg_string: SVG content
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_string)
