# apx-clsct

Color Layer Separation + Contour Tracing Vectorization

A Python module for converting raster images to SVG using color quantization, layer extraction, contour detection, and curve smoothing.

## Installation

```bash
cd CLSCT
pip install -e .
```

## Pipeline Modes

The module supports two vectorization pipelines:

- **CLSCT** (default): General-purpose vectorization with configurable smoothing
- **POSTER**: Sharp, geometric vector art with no smoothing - ideal for logos and character art

## CLI Commands

### CLSCT Mode (Default - General Purpose)

```bash
# Vectorize with 24 colors (default)
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg

# Vectorize with custom color count
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --colors 32
```

### POSTER Mode (Sharp, Geometric Vector Art)

```bash
# Basic poster mode - sharp edges, 32 colors
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --mode poster

# Higher detail preservation
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --mode poster --colors 48

# Minimal simplification for maximum detail
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --mode poster --epsilon 0.001
```

### With Debug Output

```bash
# Save intermediate stage visualizations
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --debug

# Debug poster pipeline
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --mode poster --debug
```

### Smoothing Options (CLSCT mode only)

```bash
# No smoothing (default) - best for sharp edges
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg

# Smart smoothing - preserves corners, smooths long curves
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth smart

# Gaussian smoothing
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth gaussian --sigma 1.5

# B-spline smoothing
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth bspline
```

**Note:** Smoothing is NOT available in POSTER mode - it's forced to "none" for crisp edges.

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

### CLSCT Pipeline (General Purpose)

```python
from apx_clsct.pipeline import Pipeline, PipelineConfig

# Create pipeline with custom config
config = PipelineConfig(
    n_colors=24,
    smooth_method="gaussian",
    smooth_sigma=1.0,
    epsilon_factor=0.005
)

pipeline = Pipeline(config)
svg = pipeline.process("input.jpg", "output.svg")
```

### POSTER Pipeline (Sharp, Geometric)

```python
from apx_clsct.pipeline import PosterPipeline, PosterPipelineConfig

# Create poster pipeline - optimized for sharp edges
config = PosterPipelineConfig(
    n_colors=32,           # Higher for gradients
    epsilon_factor=0.002,  # Minimal simplification
    min_contour_area=20.0  # Keep small features
)

pipeline = PosterPipeline(config)
svg = pipeline.process("input.jpg", "output.svg")
```

## Pipeline Stages

1. **Color Quantization** - Reduce image to limited palette using K-means
2. **Layer Extraction** - Create binary masks for each color
3. **Contour Detection** - Find boundaries using edge detection
4. **Curve Simplification** - Douglas-Peucker algorithm
5. **Curve Smoothing** - Bézier/B-spline fitting (CLSCT mode only)
6. **SVG Generation** - Export as SVG paths

### Mode Differences

| Feature | CLSCT | POSTER |
|---------|-------|--------|
| Default Colors | 24 | 32 |
| Smoothing | Configurable | None (forced) |
| Dilation | 1 iteration | 0 (none) |
| Min Contour Area | 50.0 | 20.0 |
| Epsilon Factor | 0.005 | 0.002 |
| Best For | Photos, illustrations | Logos, character art |

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
