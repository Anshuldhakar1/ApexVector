# apx-clsct Project Context (mind.md)

## Project Overview
**apx-clsct** = Color Layer Separation + Contour Tracing
A Python module for converting raster images (JPG/PNG) to optimized SVG using:
1. Color quantization (K-means)
2. Layer extraction (binary masks per color)
3. Contour detection (OpenCV)
4. Simplification (Douglas-Peucker)
5. Smoothing (optional)
6. SVG generation with hole support

## Root Directory
`D:\Github Cloning\ApexVector\CLSCT` (treat as project root)

## Key Files & Their Purposes

### Configuration & Planning
- `plan.md` - Project requirements and iterative building guidelines
- `core.md` - Technical implementation details
- `architecture.md` - User-facing usage documentation
- `mind.md` - This file (current understanding/state)

### Source Code (`apx_clsct/`)
- `pipeline.py` - Main orchestrator with PipelineConfig dataclass
- `quantize.py` - K-means color quantization
- `extract.py` - Binary mask extraction per color
- `contour.py` - OpenCV contour detection
- `simplify.py` - Douglas-Peucker simplification
- `smooth.py` - Gaussian, B-spline, and smart smoothing
- `svg.py` - SVG path generation with compound path support
- `cli.py` - Command-line interface
- `types.py` - Type definitions

### Tests (`tests/`)
- `test_pipeline.py` - Integration tests
- `test_real_images.py` - Real image testing (the important ones)
- `test_quantize.py`, `test_contour.py`, etc. - Unit tests
- `utils/svg_to_png.py` - Development utility for visual verification

### Build Configuration
- `pyproject.toml` - Package metadata and dependencies
- `requirements.txt` - Dependencies
- `README.md` - User documentation

## Current Default Configuration

```python
PipelineConfig(
    n_colors=24,                    # Default colors
    min_area=50,                    # Filter small mask regions
    dilate_iterations=1,            # Prevent gaps between regions
    contour_method="simple",
    min_contour_area=50.0,          # Filter tiny contours
    epsilon_factor=0.005,           # Shape preservation
    smooth_method="none",           # Default: sharp edges
    smooth_sigma=1.0,
    smoothness=3.0,
    use_bezier=True
)
```

## Smoothing Options
1. **none** (default) - Sharp edges, best for most cases
2. **smart** - Preserves corners, smooths long curves only
3. **gaussian** - Global smoothing (can blur details)
4. **bspline** - Polynomial curve fitting

## Key Issues Fixed

### 1. White Gaps/Fragmentation ✓
**Problem:** White holes inside Snorlax's body, fragmented regions
**Solution:** 
- Contour hierarchy detection (parent = outer, child = hole)
- Compound paths with `fill-rule="evenodd"`
- Holes reversed in direction for proper winding

### 2. Jagged Edges ✓
**Problem:** Sharp angles on curves, pixelated appearance
**Solution:**
- Smart smoothing that preserves corners (angle < 45°)
- Only smooths long curves (>100 points)
- Gentle gaussian (sigma=0.5) on curve segments

### 3. Gaps Between Regions ✓
**Problem:** Micro-gaps between neighboring color regions
**Solution:**
- 1px mask dilation by default
- Optional 2px for cleaner merging
- Better region connectivity

## Testing Approach

### Requirements (from plan.md)
- Always test with **24 colors**
- Use **real images** from `../test_images/`
- Create **separate folders** per image in `tests/output/`
- **Convert SVG to PNG** for visual inspection
- Analyze mistakes visually and propose fixes

### Test Images Available
- `img0.jpg` - Snorlax (431x431)
- `img1.jpg` - Character with sunglasses (680x680)
- `img2.jpg` - Abstract black/yellow design (768x768)
- `img3.jpg`, `img4.jpg`, `img5.jpg` - Additional test images

### Current Test Results
- **41 tests passing**
- All 3 test images processed successfully
- Comparison outputs in `output/comparison/`

## Important Commands

### Run Pipeline
```bash
cd CLSCT

# Default (24 colors, no smoothing, transparent bg)
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg

# With smart smoothing
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --smooth smart

# Custom colors
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --colors 32

# With debug stages
python -m apx_clsct -i ../test_images/img0.jpg -o output.svg --debug
```

### Testing
```bash
# Run all tests
python -m pytest tests/ --ignore=tests/test_rasterizer.py

# Run specific test file
python -m pytest tests/test_real_images.py -v

# Run with visual output
python -c "
from apx_clsct.pipeline import Pipeline, PipelineConfig
pipeline = Pipeline(PipelineConfig(n_colors=24))
svg = pipeline.process('../test_images/img0.jpg', 'output.svg')
"
```

### Git Workflow
- Always commit after meaningful changes
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- Update README with each commit

## Current State

### What's Working
✅ Hole handling - no more white gaps
✅ Sharp edge preservation with no smoothing
✅ Smart smoothing option available
✅ Transparent SVG backgrounds
✅ Auto-creation of output directories
✅ 41/41 tests passing
✅ Comparison outputs generated for 3 test images

### Comparison Results Location
`output/comparison/`
- `img0_v1_fixed.png` - Basic hole fix
- `img0_v2_smart.png` - With smart smoothing
- `img0_v3_dilated.png` - With extra dilation
- (Same pattern for img1, img2)

### Next Steps (if needed)
1. Review comparison outputs visually
2. Identify remaining issues
3. Propose and implement fixes
4. Iterate until satisfied

## Notes

- **Cairo not available** - Using matplotlib fallback for PNG conversion
- **Windows paths** - Using `\` separator in output
- **No linter/typechecker** - Clean code by convention only
- **Transparent background** - Required, non-negotiable (fill="none")
- **24 colors default** - Per plan.md requirements

## Architecture Understanding

The pipeline processes images through these stages:
1. **Load** → RGB array
2. **Quantize** → K-means to N colors
3. **Extract** → Binary mask per color
4. **Clean** → Remove noise, dilate slightly
5. **Find Contours** → With hierarchy (outer + holes)
6. **Group** → Match holes to parent contours
7. **Simplify** → Douglas-Peucker
8. **Smooth** → Optional (none/smart/gaussian/bspline)
9. **Generate SVG** → Compound paths with evenodd fill
10. **Save** → Auto-create dirs if needed

Each color layer becomes one or more SVG paths. Holes are drawn as reverse-wound sub-paths within their parent.
