# ApexVector

Python image vectorization: raster → optimized SVG via adaptive region classification. Converts raster images (JPG/PNG) into scalable vector graphics (SVG) with a focus on preserving geometric and perceptual quality using shared boundary extraction and Gaussian smoothing.

![Poster-style Vectorization Example](showcase_poster.svg)

*Example: Poster-style vectorization with 12 colors and smooth boundaries*

## Installation

ApexVector requires Python 3.9+ and the following dependencies:

```bash
# Clone the repository
git clone https://github.com/Anshuldhakar1/ApexVector.git
cd ApexVector

# Create virtual environment and install dependencies
uv venv venv
uv pip install -r requirements.txt
```

## How to Run

The program is run via the command line interface using the `apexvec` module.

### Basic Usage

To vectorize an input image with default settings:

```bash
python -m apexvec input.png -o output.svg
```

### Poster-Style Vectorization

Creates artistic poster-style vectorization with flat colors and smooth boundaries using shared boundary extraction:

```bash
# Default: 12 colors
python -m apexvec input.png -o output.svg --poster

# Custom color count (8-24 recommended)
python -m apexvec input.png -o output.svg --poster --colors 24

# With debug stage outputs
python -m apexvec input.png -o output.svg --poster --colors 24 --save-stages debug/
```

The poster pipeline:
1. Quantizes image to N colors using K-means in LAB space
2. Extracts connected components as regions
3. Extracts shared boundaries between adjacent regions
4. Applies Gaussian smoothing to shared edges
5. Reconstructs regions from smoothed boundaries
6. Exports as SVG with solid fills (no gaps)

### Advanced Options

| Flag | Description |
|---|---|
| `--speed` | Fast mode with looser tolerances |
| `--quality` | Quality mode with tighter tolerances |
| `--colors N` | Number of colors for poster mode (default: 12) |
| `--save-stages DIR` | Save debug visualizations for each stage |

## Architecture

### Core Modules

- `poster_first_pipeline.py` - Main poster-style pipeline with shared boundaries
- `color_quantizer.py` - K-means quantization in LAB color space
- `boundary_smoother.py` - Gaussian smoothing of shared edges
- `svg_export.py` - SVG generation with proper fill rules

### Key Features

- **Shared Boundary Extraction**: One edge per adjacent region pair prevents gaps
- **Gap-Free Output**: Solid fills with no transparency between regions
- **Color Fidelity**: Uses quantized palette colors directly (not mean colors)
- **Scalable**: Handles 4K images with 24 colors

## Usage Examples

```bash
# Vectorize Snorlax image with 24 colors
python -m apexvec test_images/img0.jpg -o snorlax.svg --poster --colors 24

# Vectorize 4K image
python -m apexvec test_images/4K_image.jpg -o output.svg --poster --colors 24
```

## License

MIT License - See LICENSE file for details
