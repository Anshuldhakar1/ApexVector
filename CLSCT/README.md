# apx-clsct

Color Layer Separation + Contour Tracing Vectorization

A Python module for converting raster images to SVG using color quantization, layer extraction, contour detection, and curve smoothing.

## Installation

```bash
cd CLSCT
pip install -e .
```

## CLI Commands

### Basic Vectorization

```bash
# Vectorize with 24 colors (default)
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg

# Vectorize with custom color count
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --colors 12
```

### With Debug Output

```bash
# Save intermediate stage visualizations
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --debug
```

### Smoothing Options

```bash
# No smoothing (default) - best for sharp edges
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg

# Gaussian smoothing
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth gaussian --sigma 1.5

# B-spline smoothing
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth bspline
```

### Advanced Options

```bash
# Adjust simplification (lower = more points)
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --epsilon 0.005

# Filter small mask regions (noise removal)
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --min-area 50

# Filter small contours after detection
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --min-contour-area 100.0
```

## Python API

```python
from apx_clsct.pipeline import Pipeline, PipelineConfig

# Create pipeline with custom config
config = PipelineConfig(
    n_colors=10,
    smooth_method="gaussian",
    smooth_sigma=1.0,
    epsilon_factor=0.01
)

pipeline = Pipeline(config)
svg = pipeline.process("input.jpg", "output.svg")
```

## Pipeline Stages

1. **Color Quantization** - Reduce image to limited palette using K-means
2. **Layer Extraction** - Create binary masks for each color
3. **Contour Detection** - Find boundaries using edge detection
4. **Curve Simplification** - Douglas-Peucker algorithm
5. **Curve Smoothing** - Bézier/B-spline fitting
6. **SVG Generation** - Export as SVG paths

## Testing

```bash
cd CLSCT
python -m pytest tests/
```

Test images are available in `../test_images/`.

### SVG to PNG Rasterizer (Development Utility)

For visualizing outputs during development, a test utility converts SVG outputs to PNG:

```python
from tests.utils.svg_to_png import svg_to_png, convert_folder_svgs

# Convert single SVG
png_path = svg_to_png("output.svg", "output.png")

# Convert all SVGs in folder
png_files = convert_folder_svgs("output_folder", "*.svg")
```

**Note:** This utility is NOT part of the user-facing pipeline. Install dependencies:
```bash
pip install matplotlib cairosvg
```
