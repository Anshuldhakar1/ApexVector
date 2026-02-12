"""SVG to PNG rasterizer for testing/debugging only.

This module is NOT part of the user-facing pipeline.
It's used internally for visualizing and comparing outputs.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

def svg_to_png(
    svg_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    scale: float = 1.0,
    size: Optional[Tuple[int, int]] = None
) -> Path:
    """Convert SVG file to PNG.
    
    This is a testing utility for visualizing outputs.
    Not part of the user-facing pipeline.
    
    Args:
        svg_path: Path to input SVG file
        output_path: Path for output PNG (default: same name with .png)
        scale: Scale factor for output (default: 1.0)
        size: Optional (width, height) to override SVG dimensions
        
    Returns:
        Path to output PNG file
        
    Raises:
        ImportError: If no SVG rendering backend is available
    """
    svg_path = Path(svg_path)
    
    if output_path is None:
        output_path = svg_path.with_suffix(".png")
    else:
        output_path = Path(output_path)
    
    # Try cairosvg first (best quality)
    try:
        import cairosvg
        _render_with_cairosvg(svg_path, output_path, scale, size)
        return output_path
    except ImportError:
        pass
    
    # Fallback to matplotlib
    try:
        _render_with_matplotlib(svg_path, output_path, size)
        return output_path
    except Exception as e:
        raise ImportError(
            f"Could not render SVG: {e}. "
            "Install cairosvg for best results: pip install cairosvg"
        )


def _render_with_cairosvg(
    svg_path: Path,
    output_path: Path,
    scale: float,
    size: Optional[Tuple[int, int]]
) -> None:
    """Render SVG using cairosvg."""
    import cairosvg
    
    svg_content = svg_path.read_text()
    
    if size:
        width, height = size
        png_data = cairosvg.svg2png(
            bytestring=svg_content,
            output_width=width,
            output_height=height
        )
    else:
        png_data = cairosvg.svg2png(
            bytestring=svg_content,
            scale=scale
        )
    
    output_path.write_bytes(png_data)


def _render_with_matplotlib(
    svg_path: Path,
    output_path: Path,
    size: Optional[Tuple[int, int]]
) -> None:
    """Render SVG using matplotlib (fallback)."""
    import matplotlib.pyplot as plt
    from xml.etree import ElementTree as ET
    import numpy as np
    
    # Parse SVG
    tree = ET.parse(svg_path)
    root = tree.getroot()
    
    # Get SVG dimensions
    svg_ns = {"svg": "http://www.w3.org/2000/svg"}
    
    width = root.get("width", "100")
    height = root.get("height", "100")
    viewbox = root.get("viewBox", f"0 0 {width} {height}")
    
    # Parse viewBox for dimensions
    try:
        _, _, vb_width, vb_height = map(float, viewbox.split())
    except (ValueError, AttributeError):
        vb_width = float(width.replace("px", ""))
        vb_height = float(height.replace("px", ""))
    
    # Create figure
    if size:
        fig_width, fig_height = size
        dpi = 100
        figsize = (fig_width / dpi, fig_height / dpi)
    else:
        figsize = (vb_width / 100, vb_height / 100)
        dpi = 100
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, vb_width)
    ax.set_ylim(vb_height, 0)  # Flip Y for SVG coordinates
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Find all path elements
    paths = root.findall(".//svg:path", svg_ns) or root.findall(".//{http://www.w3.org/2000/svg}path")
    if not paths:
        paths = root.findall(".//path")  # Try without namespace
    
    for path_elem in paths:
        d = path_elem.get("d", "")
        fill = path_elem.get("fill", "black")
        
        # Parse simple paths (M, L, C, Z commands)
        points = _parse_svg_path(d)
        if points and len(points) >= 3:
            # Close the path
            polygon = plt.Polygon(points, closed=True, facecolor=fill, edgecolor='none')
            ax.add_patch(polygon)
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def _parse_svg_path(d: str) -> list:
    """Parse SVG path data into list of (x, y) points.
    
    Handles basic M, L, C, Z commands.
    """
    import re
    
    points = []
    current_pos = (0, 0)
    
    # Tokenize path commands
    tokens = re.findall(r'([MLCZmlcz])|(-?\d+\.?\d*)', d)
    
    i = 0
    cmd = None
    while i < len(tokens):
        token_type, value = tokens[i]
        
        if token_type:  # Command
            cmd = token_type.upper()
            i += 1
            
            if cmd == 'Z':
                if points:
                    points.append(points[0])  # Close path
                break
            continue
        
        # Coordinates
        if cmd == 'M':
            x = float(value)
            i += 1
            if i < len(tokens):
                _, y_val = tokens[i]
                y = float(y_val)
                current_pos = (x, y)
                points.append(current_pos)
                cmd = 'L'  # Subsequent coords are implicit line-to
                i += 1
                
        elif cmd == 'L':
            x = float(value)
            i += 1
            if i < len(tokens):
                _, y_val = tokens[i]
                y = float(y_val)
                current_pos = (x, y)
                points.append(current_pos)
                i += 1
                
        elif cmd == 'C':
            # Cubic bezier - skip control points, take end point
            # C x1 y1 x2 y2 x y
            try:
                x = float(value)
                coords = [x]
                i += 1
                for _ in range(5):
                    if i < len(tokens):
                        _, coord_val = tokens[i]
                        coords.append(float(coord_val))
                        i += 1
                if len(coords) >= 6:
                    current_pos = (coords[4], coords[5])
                    points.append(current_pos)
            except (ValueError, IndexError):
                i += 1
        else:
            i += 1
    
    return points


def convert_folder_svgs(folder: Union[str, Path], pattern: str = "*.svg") -> list:
    """Convert all SVGs in a folder to PNG.
    
    Args:
        folder: Path to folder containing SVGs
        pattern: Glob pattern for SVG files
        
    Returns:
        List of paths to generated PNG files
    """
    folder = Path(folder)
    png_files = []
    
    for svg_file in folder.glob(pattern):
        try:
            png_path = svg_to_png(svg_file)
            png_files.append(png_path)
            print(f"Converted: {svg_file.name} -> {png_path.name}")
        except Exception as e:
            print(f"Failed to convert {svg_file}: {e}")
    
    return png_files


if __name__ == "__main__":
    # CLI usage for testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python svg_to_png.py <input.svg> [output.png]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = svg_to_png(input_file, output_file)
    print(f"Created: {result}")
