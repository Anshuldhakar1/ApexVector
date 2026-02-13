This will be a python module "apx-clsct".

## Pipeline Modes

The module supports multiple vectorization pipelines, each optimized for different use cases:

### Mode 1: `clsct` (Default)
**Color Layer Separation + Contour Tracing**
- General-purpose vectorization
- Configurable smoothing options
- Balanced defaults for most use cases
- Good for: photos, illustrations, mixed content

### Mode 2: `poster`
**Poster-Style Vector Art**
- Sharp, geometric edges (no smoothing allowed)
- High color count for gradient separation
- Crisp boundaries, no dilation
- Optimized for: logos, character art, bold graphics, screen-print aesthetic

---

## CLI Parameters

### Required
- `--input, -i` : Input image file path (JPG, PNG)
- `--output, -o` : Output SVG file path (optional, defaults to input name with .svg extension)

### Pipeline Selection
- `--mode, -m` : Pipeline to use (default: clsct)
  - `clsct` - General-purpose (original pipeline)
  - `poster` - Poster-style sharp vector art (new pipeline)

### Optional Configuration (CLSCT mode)
- `--colors, -c` : Number of colors for quantization (default: 24)
- `--epsilon, -e` : Simplification factor (default: 0.005)
- `--min-area` : Minimum contour area to keep (default: 50)
- `--smooth` : Smoothing method (default: none)
  - Options: none, smart, gaussian, bspline
- `--dilate` : Mask dilation iterations (default: 1)

### Optional Configuration (POSTER mode)
- `--colors, -c` : Number of colors for quantization (default: 32, min: 24)
- `--epsilon, -e` : Simplification factor (default: 0.002, range: 0.001-0.003)
- `--min-area` : Minimum contour area to keep (default: 20)
- `--smooth` : **NOT AVAILABLE** (forced to 'none' for poster aesthetic)
- `--dilate` : **NOT AVAILABLE** (forced to 0 for crisp boundaries)

### Debug & Development
- `--debug` : Output intermediate pipeline stages using matplotlib
  - Shows: quantized colors, masks, contours, simplified paths

---

## Example Usage

### CLSCT Mode (Original)
```bash
# Basic usage
python -m apx_clsct -i image.jpg -o output.svg

# With smoothing
python -m apx_clsct -i image.jpg -o output.svg --smooth smart

# Custom colors and simplification
python -m apx_clsct -i image.jpg -o output.svg --colors 32 --epsilon 0.003
```

### POSTER Mode (New)
```bash
# Basic poster mode (sharp edges, high color count)
python -m apx_clsct -i logo.jpg -o output.svg --mode poster

# Higher detail preservation
python -m apx_clsct -i character.jpg -o output.svg --mode poster --colors 48 --epsilon 0.001

# Debug poster pipeline
python -m apx_clsct -i image.jpg -o output.svg --mode poster --debug
```

---

## Pipeline Implementation Pattern

Both pipelines use **dependency injection** pattern:
- Shared core modules: `quantize.py`, `extract.py`, `contour.py`, `simplify.py`, `smooth.py`, `svg.py`
- Separate config classes: `PipelineConfig` (CLSCT) and `PosterPipelineConfig` (POSTER)
- Mode selection at CLI level determines which config/pipeline is instantiated
- POSTER mode enforces restrictions (no smoothing, no dilation) at config validation level

This allows:
- Adding new pipelines without breaking existing code
- Each pipeline optimized for specific aesthetic
- Shared modules = less code duplication
- Clear separation of concerns