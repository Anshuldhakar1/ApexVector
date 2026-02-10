# Poster Pipeline Implementation

## Overview

The poster-style vectorization pipeline converts raster images into artistic poster-style SVGs with flat colors and smooth spline boundaries. It was designed to complement the standard adaptive pipeline with a focus on artistic vectorization rather than photographic fidelity.

## Pipeline Architecture

The poster pipeline consists of six sequential stages, each with corresponding visualization outputs when the debug mode is enabled.

### Stage 1: Image Ingestion

**File:** `apexvec/raster_ingest.py`

The pipeline begins by loading the input image through the shared ingestion module. This stage handles EXIF orientation correction, color space conversion from sRGB to linear RGB for processing, and produces an sRGB version for visualization. The ingested image serves as the foundation for all subsequent quantization and boundary extraction operations.

### Stage 2: Color Quantization

**File:** `apexvec/color_quantizer.py`

Color quantization reduces the image to a specified number of colors using K-means clustering in LAB color space. LAB was chosen over RGB because it provides perceptually uniform color distances, resulting in more visually pleasing posterization. The quantizer produces both a label map (assigning each pixel to a color cluster) and a palette containing the representative colors. The number of colors is configurable, with 12 being the default and 8-16 recommended for typical poster effects.

### Stage 3: Region Extraction

**File:** `apexvec/region_extractor.py`

Connected component analysis transforms the quantized label map into distinct regions. Each region represents a contiguous area of similar color. The extractor computes region metadata including pixel masks, bounding boxes, centroids, and mean colors. Small regions below a configurable area threshold (default 0.1% of image area) are merged with their nearest color neighbor to reduce noise and SVG complexity.

### Stage 4: Boundary Smoothing

**File:** `apexvec/boundary_smoother.py`

Region boundaries undergo spline-based smoothing to eliminate quantization artifacts and create aesthetically pleasing curves. Each boundary is extracted as a contour and fitted with periodic cubic splines. The smoothness parameter controls the trade-off between fidelity to the original boundary and curve smoothness. Higher values produce smoother, more stylized boundaries while lower values preserve more detail. The output is a collection of Bézier paths suitable for SVG export.

### Stage 5: SVG Generation

**File:** `apexvec/svg_export.py`

The smoothed boundaries and region colors are compiled into a compact SVG file. The export uses relative cubic Bézier commands for efficient path encoding. Each region becomes a closed path filled with its representative color. The SVG is optimized for size while maintaining visual quality, using coordinate precision appropriate for the image resolution.

### Stage 6: Output and Validation

**File:** `apexvec/poster_pipeline.py` (main orchestration)

The final stage saves the SVG output and optionally generates a side-by-side comparison visualization. The pipeline reports statistics including processing time, region count, file sizes, and size reduction percentage. Unlike the standard pipeline, poster mode focuses on artistic merit rather than perceptual metrics like SSIM, as the intentional color reduction necessarily deviates from the original.

## Configuration System

**File:** `apexvec/types.py`

Poster pipeline configuration is managed through the `ApexConfig` dataclass, which includes parameters for color count, region merging thresholds, spline smoothness, and debug output settings. The configuration system was unified to support the same `save_stages` interface as the standard pipeline, enabling consistent debugging across both modes.

## Visualization System

**File:** `apexvec/viz_utils.py`

The pipeline generates six debug visualizations when `save_stages` is enabled:

1. **Ingest visualization:** Original input image after EXIF correction
2. **Quantization visualization:** Color-reduced image with palette strip showing the extracted colors
3. **Region visualization:** Original image with region boundaries overlaid and region count
4. **Boundary visualization:** Faded background with smoothed spline boundaries highlighted
5. **Final SVG:** Generated output file
6. **Comparison visualization:** Side-by-side view of original and vectorized output with metrics overlay

The visualization system uses matplotlib and PIL to create informative debug images that help diagnose issues at each pipeline stage.

## Command-Line Interface

**File:** `apexvec/cli.py`

The poster pipeline is invoked via the `--poster` flag with optional `--colors` parameter. The CLI handles configuration setup and delegates to the `PosterPipeline` class. Debug output is controlled via `--save-stages` followed by a directory path.

## Integration with Standard Pipeline

The poster pipeline shares infrastructure with the standard adaptive pipeline where appropriate:

- Common image ingestion
- Shared SVG export utilities
- Unified configuration pattern
- Consistent CLI interface
- Parallel batch processing support

This design ensures maintenance improvements benefit both pipelines while allowing each to optimize for its specific use case.

## Performance Characteristics

The poster pipeline typically processes images in 10-20 seconds depending on image size and complexity. The most time-consuming operations are color quantization (K-means clustering) and boundary smoothing (spline fitting). The pipeline produces SVGs with significantly fewer regions than the standard pipeline (typically 20-100 vs 100-500), resulting in smaller file sizes and faster rendering.

## Known Limitations

- Boundary smoothing may produce warnings when spline fitting struggles with tight tolerances
- Small color regions can create fragmented output if the area threshold is too low
- The pipeline is designed for artistic posterization, not photographic accuracy
- Region count can exceed the configured maximum if the image has many distinct color areas

## Future Enhancements

Potential improvements include adaptive color count selection based on image content, better handling of gradient regions, and integration with the contrast-aware pipeline for multi-scale posterization.
