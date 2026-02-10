"""SVG to PNG conversion utility."""
import logging
from pathlib import Path
from typing import Union, Optional

try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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
    
    # Try cairosvg first (best quality)
    if HAS_CAIROSVG:
        try:
            _convert_with_cairosvg(svg_path, output_path, width, height, scale)
            logger.info(f"Converted SVG to PNG using cairosvg: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"cairosvg conversion failed: {e}")
    
    # Fallback to PIL + rsvg (if available)
    if HAS_PIL:
        try:
            _convert_with_pil(svg_path, output_path, width, height, scale)
            logger.info(f"Converted SVG to PNG using PIL: {output_path}")
            return output_path
        except Exception as e:
            logger.warning(f"PIL conversion failed: {e}")
    
    logger.error("No SVG to PNG converter available. Install cairosvg: pip install cairosvg")
    return None


def _convert_with_cairosvg(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using cairosvg."""
    # Read SVG content
    svg_content = svg_path.read_text(encoding='utf-8')
    
    # Parse width/height from SVG if not provided
    if width is None and height is None:
        import re
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


def _convert_with_pil(
    svg_path: Path,
    output_path: Path,
    width: Optional[int],
    height: Optional[int],
    scale: float
):
    """Convert using PIL and librsvg (Linux) or Inkscape."""
    import subprocess
    import tempfile
    
    # Try using Inkscape
    try:
        cmd = ['inkscape', str(svg_path), '--export-type=png', '--export-filename=' + str(output_path)]
        
        if width:
            cmd.extend(['--export-width', str(width)])
        if height:
            cmd.extend(['--export-height', str(height)])
        
        subprocess.run(cmd, check=True, capture_output=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try using rsvg-convert
    try:
        cmd = ['rsvg-convert', '-o', str(output_path)]
        
        if width:
            cmd.extend(['-w', str(width)])
        if height:
            cmd.extend(['-h', str(height)])
        
        cmd.append(str(svg_path))
        subprocess.run(cmd, check=True, capture_output=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    raise RuntimeError("No SVG converter available (tried Inkscape, rsvg-convert)")
