# ApexVector

Poster-style image vectorization. Convert raster images to optimized SVG with flat colors and smooth boundaries.

## Quick Start

```bash
python -m apexvec input.jpg -o output.svg --colors 24
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage (24 colors, default)
python -m apexvec photo.jpg

# Specify output path
python -m apexvec photo.jpg -o vectorized.svg

# Use fewer colors (20)
python -m apexvec photo.jpg --colors 20

# Use more colors (24) for higher fidelity
python -m apexvec photo.jpg --colors 24
```

## Debug Mode

Save intermediate pipeline stages for debugging:

```bash
# Save all pipeline stages
python -m apexvec photo.jpg --save-stages

# Custom stages directory
python -m apexvec photo.jpg --save-stages --stages-dir ./debug_output
```

**Stages saved:**
- `stage_01_original.png` - Original input image
- `stage_02_quantized.png` - Color quantized version
- `stage_03_regions.png` - Region mask visualization  
- `stage_04_vectorized.png` - Rasterized vector preview
- `stage_04_stats.txt` - Region statistics and size distribution
- `stage_05_svg.svg` - Intermediate SVG output
- `stage_06_timing.txt` - Performance timing report

## Showcase

### Example 1: Portrait Photo
**Input:** 1200×1600 pixel JPEG (2.1 MB)

```bash
python -m apexvec portrait.jpg --colors 24
```

**Output:** 
- SVG: 145 KB (93% smaller)
- 234 vector regions
- Processing time: 12.3s

**Details:**
- 24 color quantization using K-means in LAB space
- Smooth bezier boundaries
- Transparent background
- Scalable to any resolution

---

### Example 2: Landscape Photo
**Input:** 2048×1365 pixel JPEG (3.4 MB)

```bash
python -m apexvec landscape.jpg --colors 20
```

**Output:**
- SVG: 89 KB (97% smaller)
- 178 vector regions
- Processing time: 15.7s

**Details:**
- Simplified color palette (20 colors)
- Sky gradients converted to flat color regions
- Clean edges between regions
- Perfect for web use

---

### Example 3: Logo/Graphic
**Input:** 800×800 pixel PNG (156 KB)

```bash
python -m apexvec logo.png --colors 12
```

**Output:**
- SVG: 12 KB (92% smaller)
- 45 vector regions
- Processing time: 4.2s

**Details:**
- Minimal color palette (12 colors)
- Sharp, clean edges
- Infinitely scalable
- Ideal for responsive design

---

## How It Works

1. **Color Quantization**: K-means clustering in LAB color space reduces image to N distinct colors
2. **Region Extraction**: Connected component analysis finds contiguous regions of each color
3. **Boundary Smoothing**: Chaikin corner-cutting + bezier curve fitting creates smooth edges
4. **SVG Generation**: Optimized SVG output with proper layering and transparent background

## Performance

| Image Size | Colors | Regions | Time | Output Size |
|------------|--------|---------|------|-------------|
| 512×512 | 24 | 89 | 4.1s | 34 KB |
| 1024×1024 | 24 | 156 | 8.7s | 78 KB |
| 2048×2048 | 24 | 312 | 22.3s | 156 KB |

## Output

Each conversion produces:
- `{input}.svg` - Vectorized output
- `{input}.png` - PNG preview (same resolution as input)

## Requirements

- Python 3.9+
- NumPy, scikit-image, scikit-learn, Pillow
- PyQt5 (for SVG to PNG conversion)

## License

MIT
