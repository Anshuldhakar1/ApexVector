# ApexVector — Poster-Style Vectorization Plan

## Pipeline

```
Input → EXIF fix → Color Quantization (K-means, 8-16 colors) → 
Connected Components → Small Region Merge → 
Spline Boundary Smoothing → SVG Export
```

## Modules

| # | Module | Function |
|---|--------|----------|
| 1 | `types.py` | `Region`, `BezierPath`, `ApexConfig` |
| 2 | `raster_ingest.py` | EXIF transpose, linear RGB |
| 3 | `color_quantizer.py` | K-means in LAB space, output label map + palette |
| 4 | `region_extractor.py` | Connected components per label, merge regions < 0.1% area |
| 5 | `boundary_smoother.py` | Cubic spline fit to contours, periodic boundary conditions |
| 6 | `svg_export.py` | Solid fills, relative cubic Bézier commands |
| 7 | `pipeline.py` | Orchestrate with `--save-stages` toggle |
| 8 | `cli.py` | `apexvector input.png -o out.svg --colors 12` |

## Key Implementation

**Color Quantization**
- K-means++ in CIELAB, `n_colors=12` default
- `n_init=10` for stability

**Region Extraction**
- `scipy.ndimage.label` per color
- Merge small regions to nearest color by ΔE2000

**Boundary Smoothing**
- Parametric cubic spline `x(t), y(t)` with `bc_type='periodic'`
- Resample to `max(4, len(contour)//10)` points
- Convert to cubic Bézier via chord-length parameterization

**SVG Export**
- Solid fills only
- Relative `c` commands
- `#RGB` shorthand when possible

## Config

```python
@dataclass
class ApexConfig:
    n_colors: int = 12
    min_region_area_ratio: float = 0.001
    spline_smoothness: float = 0.5
    max_regions: int = 20
```

## Test

All images in `./testing/`. Verify: region count ≤ 20, smooth boundaries, no rotation, valid SVG.