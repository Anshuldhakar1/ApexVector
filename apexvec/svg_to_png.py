"""SVG to PNG conversion utility."""
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)


def svg_to_png(
    svg_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0
) -> Optional[Path]:
    """
    Convert SVG to PNG.
    
    Args:
        svg_path: Path to SVG file or SVG string
        output_path: Output PNG path (default: svg_path with .png extension)
        width: Output width in pixels
        height: Output height in pixels
        scale: Scale factor (if width/height not specified)
        
    Returns:
        Path to output PNG file or None if conversion failed
    """
    svg_path = Path(svg_path)
    
    if output_path is None:
        output_path = svg_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    # Try converters in order of preference
    converters = [
        ("cairosvg", _convert_with_cairosvg),
        ("pyqt5", _convert_with_pyqt5),
        ("drawsvg", _convert_with_drawsvg),
        ("svglib", _convert_with_svglib),
        ("wand", _convert_with_wand),
        ("inkscape", _convert_with_inkscape),
        ("rsvg-convert", _convert_with_rsvg),
    ]
    
    for name, converter in converters:
        try:
            converter(svg_path, output_path, width, height, scale)
            logger.info(f"Converted SVG to PNG using {name}: {output_path}")
            return output_path
        except Exception as e:
            logger.debug(f"{name} conversion failed: {e}")
            continue
    
    logger.error(
        "No SVG to PNG converter available.\n"
        "Install one of:\n"
        "  - cairosvg: pip install cairosvg (requires Cairo system library)\n"
        "  - svglib: pip install svglib reportlab (pure Python)\n"
        "  - wand: pip install wand (requires ImageMagick)\n"
        "  - Or install Inkscape/rsvg-convert externally"
    )
    return None


def _convert_with_cairosvg(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using cairosvg (requires Cairo library)."""
    import cairosvg
    import re
    
    # Read SVG content
    svg_content = svg_path.read_text(encoding='utf-8')
    
    # Parse width/height from SVG if not provided
    if width is None and height is None:
        width_match = re.search(r'width="(\d+)"', svg_content)
        height_match = re.search(r'height="(\d+)"', svg_content)
        
        if width_match and height_match:
            width = int(int(width_match.group(1)) * scale)
            height = int(int(height_match.group(1)) * scale)
    
    # Convert
    if width and height:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=str(output_path),
            output_width=width,
            output_height=height
        )
    else:
        cairosvg.svg2png(
            bytestring=svg_content.encode('utf-8'),
            write_to=str(output_path),
            scale=scale
        )


def _convert_with_pyqt5(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using PyQt5 (good cross-platform support, no extra system deps)."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QImage, QPainter
    from PyQt5.QtSvg import QSvgRenderer
    import re
    
    # Read SVG content to get dimensions
    svg_content = svg_path.read_text(encoding='utf-8')
    
    # Parse width/height from SVG
    width_match = re.search(r'width="(\d+)"', svg_content)
    height_match = re.search(r'height="(\d+)"', svg_content)
    
    if width_match and height_match:
        orig_width = int(width_match.group(1))
        orig_height = int(height_match.group(1))
    else:
        orig_width = 512
        orig_height = 512
    
    # Calculate output dimensions
    if width and height:
        out_width, out_height = width, height
    elif width:
        out_width = width
        out_height = int(orig_height * (width / orig_width))
    elif height:
        out_height = height
        out_width = int(orig_width * (height / orig_height))
    else:
        out_width = int(orig_width * scale)
        out_height = int(orig_height * scale)
    
    # Create SVG renderer
    renderer = QSvgRenderer(str(svg_path))
    
    # Create image and paint SVG onto it
    image = QImage(out_width, out_height, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    
    # Scale to fit
    renderer.render(painter)
    painter.end()
    
    # Save as PNG
    image.save(str(output_path), "PNG")


def _convert_with_drawsvg(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using drawsvg (renders to PNG via skia if available)."""
    import drawsvg as draw
    import re
    
    # Read SVG content
    svg_content = svg_path.read_text(encoding='utf-8')
    
    # Parse SVG to get dimensions
    width_match = re.search(r'width="(\d+)"', svg_content)
    height_match = re.search(r'height="(\d+)"', svg_content)
    
    if width_match and height_match:
        orig_width = int(width_match.group(1))
        orig_height = int(height_match.group(1))
    else:
        orig_width = 512
        orig_height = 512
    
    # Calculate output dimensions
    if width and height:
        out_width, out_height = width, height
    elif width:
        out_width = width
        out_height = int(orig_height * (width / orig_width))
    elif height:
        out_height = height
        out_width = int(orig_width * (height / orig_height))
    else:
        out_width = int(orig_width * scale)
        out_height = int(orig_height * scale)
    
    # Try using drawsvg's raster module if available (uses skia)
    try:
        from drawsvg import raster
        png_bytes = raster.rasterize(svg_content, output_width=out_width, output_height=out_height)
        with open(output_path, 'wb') as f:
            f.write(png_bytes)
        return
    except ImportError:
        pass  # skia not available
    except Exception as e:
        logger.debug(f"drawsvg raster failed: {e}")
    
    # Fallback: Use PIL to render by creating an image and using the SVG as a template
    # This won't render complex SVGs well but handles simple ones
    raise RuntimeError("drawsvg requires skia for rasterization")


def _convert_with_svglib(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using svglib + reportlab (pure Python, no system deps)."""
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    
    # Parse the SVG
    drawing = svg2rlg(str(svg_path))
    
    if drawing is None:
        raise RuntimeError("Failed to parse SVG with svglib")
    
    # Calculate dimensions
    orig_width = drawing.width
    orig_height = drawing.height
    
    if width and height:
        drawing.width = width
        drawing.height = height
        drawing.scale(width / orig_width, height / orig_height)
    elif width:
        ratio = width / orig_width
        drawing.width = width
        drawing.height = orig_height * ratio
        drawing.scale(ratio, ratio)
    elif height:
        ratio = height / orig_height
        drawing.width = orig_width * ratio
        drawing.height = height
        drawing.scale(ratio, ratio)
    elif scale != 1.0:
        drawing.width = orig_width * scale
        drawing.height = orig_height * scale
        drawing.scale(scale, scale)
    
    # Render to PNG
    renderPM.drawToFile(drawing, str(output_path), fmt="PNG")


def _convert_with_wand(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using Wand (ImageMagick binding)."""
    from wand.image import Image
    from wand.display import display
    
    with Image(filename=str(svg_path), format='svg') as img:
        if width and height:
            img.resize(width, height)
        elif width:
            ratio = width / img.width
            img.resize(width, int(img.height * ratio))
        elif height:
            ratio = height / img.height
            img.resize(int(img.width * ratio), height)
        elif scale != 1.0:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img.resize(new_width, new_height)
        
        img.format = 'png'
        img.save(filename=str(output_path))


def _convert_with_inkscape(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using Inkscape (external tool)."""
    import subprocess
    
    cmd = ['inkscape', str(svg_path), '--export-type=png', '--export-filename=' + str(output_path)]
    
    if width:
        cmd.extend(['--export-width', str(width)])
    if height:
        cmd.extend(['--export-height', str(height)])
    
    subprocess.run(cmd, check=True, capture_output=True)


def _convert_with_rsvg(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using rsvg-convert (external tool)."""
    import subprocess
    
    cmd = ['rsvg-convert', '-o', str(output_path)]
    
    if width:
        cmd.extend(['-w', str(width)])
    if height:
        cmd.extend(['-h', str(height)])
    
    cmd.append(str(svg_path))
    subprocess.run(cmd, check=True, capture_output=True)
