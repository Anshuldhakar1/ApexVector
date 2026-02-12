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
# Vectorize with 8 colors (default)
apx-clsct -i ../test_images/img0.jpg -o output.svg

# Vectorize with custom color count
apx-clsct -i ../test_images/img0.jpg -o output.svg --colors 12
```

### With Debug Output

```bash
# Save intermediate stage visualizations
apx-clsct -i ../test_images/img0.jpg -o output.svg --debug
```

### Smoothing Options

```bash
# Gaussian smoothing (default)
apx-clsct -i ../test_images/img0.jpg -o output.svg --smooth gaussian --sigma 1.5

# B-spline smoothing
apx-clsct -i ../test_images/img0.jpg -o output.svg --smooth bspline

# No smoothing (use line segments)
apx-clsct -i ../test_images/img0.jpg -o output.svg --smooth none
```

### Advanced Options

```bash
# Adjust simplification (lower = more points)
apx-clsct -i ../test_images/img0.jpg -o output.svg --epsilon 0.005

# Filter small regions
apx-clsct -i ../test_images/img0.jpg -o output.svg --min-area 50
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
