# ApexVector

Python image vectorization: raster → optimized SVG via adaptive region classification and multi-scale quantization. It converts raster images (JPG/PNG) into scalable vector graphics (SVG) with a focus on preserving geometric and perceptual quality using advanced curve fitting and aggressive compression.

![Poster-style Vectorization Example](showcase_poster.svg)

*Example: Poster-style vectorization with 12 colors and smooth spline boundaries*

## Installation

ApexVector requires Python 3.9+ and the following dependencies:

```bash
# Clone the repository
git clone https://github.com/Anshuldhakar1/ApexVector.git
cd ApexVector

# Install dependencies
pip install -r requirements.txt
```

## How to Run

The program is run via the command line interface using the `apexvec` module.

### Basic Usage

To vectorize an input image with default settings:

```bash
python -m apexvec input.png -o output.svg
```

### Advanced Modes

Use flags to adjust performance or quality trade-offs.

#### Speed vs Quality Modes

| Flag | Mode | Description |
|------|------|-------------|
| `--speed` | Fast | Prioritizes segmentation and faster vector fitting. |
| `--quality` | Quality | Prioritizes more complex segmentation and higher precision curve fitting. |

**Example:** Run in quality mode
```bash
python -m apexvec input.png -o output.svg --quality
```

#### Optimization Presets

| Flag | Preset | Description |
|------|--------|-------------|
| `--preset lossless` | Archival | No compression, maximum quality |
| `--preset standard` | Balanced | Good balance of size and quality (default) |
| `--preset compact` | Compact | Smaller file size with good quality |
| `--preset thumbnail` | Aggressive | Maximum compression, smallest file size |

**Example:** Run with compact preset
```bash
python -m apexvec input.png -o output.svg --preset compact
```

### Poster-Style Vectorization

Creates artistic poster-style vectorization with flat colors and smooth boundaries:

```bash
# Default: 12 colors
python -m apexvec input.png -o output.svg --poster

# Custom color count (8-16 recommended)
python -m apexvec input.png -o output.svg --poster --colors 8
```

The poster pipeline:
1. Quantizes image to N colors using K-means in LAB space
2. Extracts connected components as regions
3. Merges small regions (< 0.1% area) to nearest color
4. Fits periodic cubic splines to boundaries
5. Exports as SVG with relative cubic Bézier commands

### Contrast-Aware Vectorization (NEW)

Uses multi-scale quantization to preserve details at different scales:

```bash
# Default: 3 scales (6, 10, 18 colors)
python -m apexvec input.png -o output.svg --contrast

# Custom scales
python -m apexvec input.png -o output.svg --contrast --scales 4,8,16,24
```

The contrast-aware pipeline:
1. **Multi-Scale Quantization**: Creates 3+ quantization levels
   - Coarse (4-6 colors): Large regions like sky, ground
   - Medium (8-12 colors): Mid-size features like mountains
   - Fine (16-24 colors): Small details like trees, textures
2. **Hierarchical Merging**: Intelligently merges from coarse to fine
3. **Boundary Smoothing**: Applies spline smoothing to region boundaries
4. **SVG Export**: Generates compact SVG output

### Validation

Validate output quality with perceptual metrics:

```bash
python -m apexvec input.png -o output.svg --validate
```

This outputs:
- **SSIM**: Structural similarity index (> 0.75 recommended)
- **Delta E**: Perceptual color difference (< 15 recommended)
- **SVG size**: File size in bytes
- **Size reduction**: Compression percentage

### Debug Mode

Save intermediate pipeline stages for debugging:

```bash
# Standard pipeline (6 stages)
python -m apexvec input.png -o output.svg --save-stages ./debug_output

# Poster pipeline (6 stages)
python -m apexvec input.png -o output.svg --poster --save-stages ./debug_output
```

This saves visualization images for each pipeline stage:

**Standard Pipeline Stages:**
- `01_ingest.png` - Original input after EXIF correction
- `02_regions.png` - SLIC segmentation with region boundaries
- `03_classified.png` - Regions color-coded by type (flat/gradient/edge/detail)
- `04_vectorized.png` - Vector paths overlaid on original
- `05_merged.png` - Final merged topology with fills
- `06_final.svg` - Generated SVG output
- `06_comparison.png` - Side-by-side comparison with metrics (SSIM, ΔE, timing)

**Poster Pipeline Stages:**
- `01_ingest.png` - Original input image
- `02_quantization.png` - Color-quantized image with palette strip
- `03_regions.png` - Extracted regions with boundaries and count overlay
- `04_boundaries.png` - Smoothed spline boundaries overlaid on image
- `05_final.svg` - Generated SVG output
- `06_comparison.png` - Side-by-side comparison with metrics (regions, file size, timing)

## Complete CLI Reference

```
usage: apexvec [-h] [--output OUTPUT] [--speed] [--quality] [--segments SEGMENTS]
               [--validate] [--save-stages SAVE_STAGES] [--preset PRESET]
               [--poster] [--colors COLORS] [--contrast] [--scales SCALES]
               input

Convert raster images to optimized SVG

positional arguments:
  input                 Input image path

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output SVG path (default: input.svg)
  --speed               Fast mode - lower quality but faster
  --quality             Quality mode - higher quality but slower
  --segments SEGMENTS   Number of SLIC segments (default: 400)
  --validate            Validate output quality
  --save-stages SAVE_STAGES
                        Directory to save pipeline stage debug images
  --preset PRESET       Optimization preset: lossless, standard, compact, thumbnail
  --poster              Poster-style mode - flat colors with smooth boundaries
  --colors COLORS       Number of colors for poster mode (default: 12)
  --contrast            Contrast-aware mode - multi-scale quantization
  --scales SCALES       Color scales for contrast mode (default: 6,10,18)
```

## Testing

Run the unit tests to verify installation:

```bash
pytest -x                    # Run all tests (stops on first failure)
pytest -v -x path/to/test.py # Verbose for a single file
```

## Development Branches

### Feature: Contrast-Aware (Current Branch)

The `feature/contrast-aware` branch implements multi-scale quantization for better detail preservation.

**Key Features:**
- Multi-scale K-means quantization at 3+ levels
- Hierarchical region merging from coarse to fine
- Delta E-based contrast thresholding
- Configurable scale parameters

### Feature: Curvature-Preserving

The `feature/curvature-preserving` branch is under active development to implement Clothoid-based G² curve synthesis for superior curvature preservation.

To switch to the development branch:

```bash
git checkout feature/curvature-preserving
```

### Feature: Poster-Style

The `feature/poster-style-vectorization` branch contains the poster-style vectorization pipeline with flat colors and smooth boundaries.

```bash
git checkout feature/poster-style-vectorization
```

## Architecture

ApexVector uses a modular pipeline architecture:

### Core Modules

| Module | Function |
|--------|----------|
| `types.py` | Dataclasses for regions, configurations, and types |
| `raster_ingest.py` | Image loading with EXIF orientation and color space conversion |
| `multi_scale_quantizer.py` | K-means quantization at multiple scales |
| `hierarchical_merger.py` | Intelligent region merging from coarse to fine |
| `boundary_smoother.py` | Spline-based boundary smoothing |
| `svg_export.py` | Compact SVG generation with relative Bézier commands |
| `pipeline.py` | Standard unified pipeline orchestration |
| `poster_pipeline.py` | Poster-style pipeline orchestration |
| `contrast_pipeline.py` | Contrast-aware pipeline orchestration |
| `cli.py` | Command-line interface |

### Pipeline Flow

```
Input Image
    ↓
Raster Ingest (EXIF fix, linear RGB)
    ↓
Multi-Scale Quantization (3 levels)
    ↓
Hierarchical Region Merge (coarse → fine)
    ↓
Boundary Smoothing (spline fitting)
    ↓
SVG Export (compact Bézier paths)
    ↓
Output SVG
```

## Performance Metrics

| Metric | Threshold | Typical |
|--------|-----------|---------|
| SSIM | > 0.75 | 0.82 |
| Mean ΔE | < 15 | 8.5 |
| SVG size | < input | 45-70% smaller |
| Speed (512×512) | < 10s | 5-8s |

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
