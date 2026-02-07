I'll condense this to the essential technical instructions for a coding agent, focusing only on what needs to be built and how to test it.

---

# Vectorizer v2 (ApexVector) — Technical Plan

## Architecture Overview

Replace two rigid pipelines with one adaptive pipeline that classifies image regions and applies optimal strategies per region.

```
Input → Ingest → SLIC Segmentation → Classify → Route to Strategy → Merge → Optimize SVG
```

## Module Build Order

| # | Module | Key Function | Test Against |
|---|--------|--------------|--------------|
| 1 | `types.py` | `Region`, `VectorRegion`, `AdaptiveConfig` dataclasses | Instantiate all, validate shapes |
| 2 | `compute_backend.py` | `slic_superpixels`, `delta_e_2000`, `fit_bezier`, `delaunay` | Load image from `./testing/`, verify outputs |
| 3 | `raster_ingest.py` | `ingest(path) → IngestResult` | All images in `./testing/`, verify linear/sRGB arrays |
| 4 | `perceptual_loss.py` | `perceptual_loss()`, `rasterize_svg()` | Image vs self = loss 0; vs blur = loss > 0 |
| 5 | `region_decomposer.py` | `decompose() → list[Region]` | Masks cover image, disjoint, neighbors symmetric |
| 6 | `region_classifier.py` | `classify() → regions with kind` | At least one of each kind exists across test images |
| 7 | `strategies/flat.py` | `vectorize_flat() → VectorRegion` | SOLID fill, closed path |
| 8 | `strategies/gradient.py` | `vectorize_gradient()` | LINEAR/RADIAL/MESH gradient, stops sorted |
| 9 | `strategies/edge.py` | `vectorize_edge()` | More segments than flat, precise boundary |
| 10 | `strategies/detail.py` | `vectorize_detail()` | Mesh within triangle limit |
| 11 | `strategies/router.py` | `vectorize_all_regions()` | Parallel = sequential output |
| 12 | `topology_merger.py` | `merge_topology()` | Fewer regions out, no duplicate adjacent colors |
| 13 | `svg_optimizer.py` | `regions_to_svg() → str` | Valid XML, smaller than input |
| 14 | `pipeline.py` | `UnifiedPipeline.process()` | SSIM > 0.75, ΔE < 15, runs < 10s for 512² |
| 15 | `cli.py` | `python -m vectorizer input.png -o out.svg` | Exit 0, valid output, `--speed` faster than `--quality` |

## Key Implementation Details

- **Color space**: Work in linear RGB, convert to sRGB only for output
- **Bézier fitting**: Implement Schneider's algorithm, split at 60° corners
- **SLIC**: Use `skimage.segmentation.slic`, merge hierarchically by ΔE < threshold
- **Region kinds**: FLAT (uniform), GRADIENT (consistent direction), EDGE (high edge density), DETAIL (complex)
- **Parallel**: `ProcessPoolExecutor` with worker-local `ComputeBackend`
- **Fallback**: Any strategy failure → `flat` strategy

## Dependencies

```toml
numpy, opencv-python, scikit-image, scipy, Pillow, shapely
optional: cupy-cuda12x, cairosvg, triangle
```

## Validation

Run `pytest tests/ -x` after each module. Final check: process all images in `./testing/`, verify SSIM > 0.75, mean ΔE < 15, SVG < original size, no NaN in output.