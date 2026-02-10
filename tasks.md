Of course. Your idea to handle contrast by slicing the image, processing the layers independently, and then superimposing them is excellent. It's a "divide and conquer" strategy that allows the pipeline to focus on color relationships within a specific tonal range, preventing high-contrast boundaries from distorting the color quantization.

Here is the technical plan for this new, layered pipeline.

# ApexVector — Layered Contrast Pipeline Plan

## Core Concept

Deconstruct the input image into multiple layers based on luminance (shadows, midtones, highlights). Run the existing "poster-style" vectorization pipeline on each layer independently. Finally, composite the resulting vector layers into a single SVG, preserving the original image's full dynamic range and sharp contrast boundaries.

## Layered Pipeline Flow

```
Input Image → Luminance Slicing (e.g., Shadows, Midtones, Highlights) →
    [
        Layer 1 (Shadows) → Posterization Pipeline → SVG Paths 1
        Layer 2 (Midtones) → Posterization Pipeline → SVG Paths 2
        Layer 3 (Highlights) → Posterization Pipeline → SVG Paths 3
    ] (in parallel)
→ SVG Compositor → Final Layered SVG Output
```

## Modules & Steps

| # | Module | Function |
|---|---|---|
| 1 | `types.py` | Add `LayeredPipelineConfig`. Retain existing types. |
| 2 | `raster_ingest.py` | No changes needed. EXIF transpose and color space conversion remain. |
| 3 | **`luminance_slicer.py` (New)** | **Slices image into RGBA layers based on CIELAB 'L' channel thresholds.** |
| 4 | `poster_pipeline.py` | The previous pipeline is now a sub-routine that processes one layer. |
| 5 | **`svg_compositor.py` (New)** | **Merges SVG paths from multiple layers into a single SVG file.** |
| 6 | `pipeline.py` (Updated) | Main entry point. Orchestrates slicing, parallel processing, and compositing. |
| 7 | `cli.py` (Updated) | Add CLI flags to enable and configure layering. |

---

## Key Implementation Details

### 1. `luminance_slicer.py` (New Module)

This module creates the layers. It's the most critical new component.

**Function:** `slice_by_luminance(image, thresholds, feather_width)`

1.  **Convert to CIELAB:** Convert the input sRGB image to CIELAB color space. The 'L' channel represents perceptual lightness.
2.  **Define Slices:** Use the `thresholds` array to define the L\* value ranges for each layer. For example, `thresholds = [33, 66]` creates three layers:
    *   Shadows: L\* in `[0, 33)`
    *   Midtones: L\* in `[33, 66)`
    *   Highlights: L\* in `[66, 100]`
3.  **Create Feathered Masks:** For each slice, create a boolean mask. **Critically, apply a Gaussian blur (e.g., `sigma=feather_width`) to this mask.** This creates soft edges, preventing hard seams in the final composite.
4.  **Generate RGBA Layers:** Apply each feathered mask to the original RGBA image's alpha channel. The output is a list of RGBA images, where each image only contains the pixels for its designated tonal range.

### 2. `pipeline.py` (Orchestration Update)

The main `process` function will be updated to manage the layered workflow.

1.  **Check for Layering:** An `if config.use_layering:` block will control the workflow.
2.  **Slice:** Call `luminance_slicer.slice_by_luminance()`.
3.  **Parallel Processing:**
    *   Use a `concurrent.futures.ProcessPoolExecutor`.
    *   For each sliced RGBA image, submit a job to run the `poster_pipeline.process()` sub-routine. This happens in parallel for maximum speed.
4.  **Composite:** Collect the results (lists of vector paths) from each parallel job. Pass these results to the `svg_compositor`.

### 3. `svg_compositor.py` (New Module)

This module builds the final SVG.

**Function:** `compose_layers(layer_results, width, height)`

1.  **Create Root SVG:** Start with a clean `<svg>` element with the correct `viewBox`.
2.  **Create Layer Groups:** For each layer's vector paths, create a `<g>` (group) element. Assign a descriptive ID, e.g., `<g id="layer-shadows">`.
3.  **Add Paths:** Convert the vector regions from each result into SVG `<path>` strings and place them inside their corresponding group.
4.  **Stack Correctly:** Append the `<g>` elements to the root SVG in the correct order: shadows first, then midtones, then highlights. This ensures that highlights are rendered on top of shadows.

---

## Configurable Parameters

Add these to `ApexConfig` to control the new behavior.

```python
@dataclass
class ApexConfig:
    # ... existing poster-style options ...
    use_layering: bool = True  # Master switch for this feature
    luminance_thresholds: list[int] = field(default_factory=lambda: [33, 66])
    feather_width: int = 2  # In pixels. Crucial for soft blending.
```

## Test Validation

After implementation, the agent must verify:
1.  **Improved Contrast:** High-contrast edges (like the treeline against the sky) are sharp and well-defined.
2.  **No Seams:** There are no visible lines or artifacts at the boundaries where luminance layers meet, thanks to the feathered masks.
3.  **Correct Stacking:** The final SVG layers are in the correct z-order.
4.  **Performance:** The parallel processing should keep the total time reasonable, despite running the core pipeline multiple times.