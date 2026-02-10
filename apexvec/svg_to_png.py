"""SVG to PNG conversion utility using PyQt5."""
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
    Convert SVG to PNG using PyQt5.
    
    Args:
        svg_path: Path to SVG file
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
    
    try:
        _convert_with_pyqt5(svg_path, output_path, width, height, scale)
        logger.info(f"Converted SVG to PNG: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"SVG to PNG conversion failed: {e}")
        return None


def _convert_with_pyqt5(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using PyQt5."""
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
