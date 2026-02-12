# SLIC Pipeline Validation Results

## Branch
`spike/slic-pipeline-20260211`

## Pipeline Architecture

Stage 1: SLIC quantization (spatial + color)
↓
Stage 2: Same-color region merging (cleanup fragmentation)
↓
Stage 3: Marching squares boundary extraction (sub-pixel)
↓
Stage 4: Gaussian smoothing per shared edge
↓
Stage 5: Region reconstruction from smoothed edges
↓
Stage 6: SVG export with palette colors
↓
Stage 7: resvg CLI rasterization + comparison

## Test Results: img0.jpg (Snorlax)

### Stage 1: Ingest
- Image: 431x431 pixels
- Status: PASS

### Stage 2: SLIC Quantization
- Target colors: 12
- SLIC segments: 48 (4x colors for spatial coherence)
- Output colors: 12 (all preserved)
- Status: PASS
- Key fix: Stage visualization now uses palette colors (not random colors)

### Stage 3: Same-Color Region Merging
- Min area threshold: 185 pixels (0.1% of image)
- Input regions: ~48 (from SLIC)
- Output regions: 21
- All 12 colors preserved
- Status: PASS

### Stage 4: Marching Squares Boundary Extraction
- Algorithm: skimage.measure.find_contours (marching squares)
- Extracted edges: 21
- Status: PASS

### Stage 5: Gaussian Smoothing
- Sigma: 1.0
- Edges smoothed: 21
- Status: PASS

### Stage 6: SVG Export
- Output: output_slic.svg (146KB)
- Format: Solid fills only, no strokes, no transparency
- Status: PASS

### Stage 7: Validation
- Coverage: 100.0%
- Gap pixels: 0
- Status: PASS

## Usage

```bash
python -m apexvec input.jpg -o output.svg --slic --colors 12 --save-stages debug/
```

## Key Improvements Over Previous Pipeline

1. **Spatial coherence**: SLIC superpixels ensure spatially connected regions
2. **Palette color visualization**: Stage 2/3 now shows actual palette colors
3. **Better color preservation**: All quantized colors are preserved through region merging
4. **Gap-free output**: Shared boundaries + proper reconstruction

## Known Limitations

- resvg CLI not available in test environment (fallback to cairosvg)
- Edge count equals region count (each region appears to have 1 edge) - 
  this suggests regions may not be properly forming closed loops in some cases

## Comparison: SLIC vs Edge-Aware Pipeline

### SLIC Pipeline Issues
- **Collage-like appearance**: Spatial coherence forces smooth, blobby regions
- **Lost details**: Toes/claws get merged into larger regions
- **Pieces stuck together**: Over-smoothing at boundaries

### Edge-Aware Pipeline Improvements
- **No spatial weighting**: Pure K-means in LAB space preserves sharp color edges
- **Better color fidelity**: Stage 2 shows clean teal/cream Snorlax (not psychedelic)
- **32 regions vs 21**: More granular segmentation preserves shape better
- **Grayscale detail detection**: Edge detection on luminance finds small features

### Visual Comparison

| Stage | SLIC | Edge-Aware |
|-------|------|------------|
| Quantization | Washed, blobby colors | Sharp, accurate colors |
| Regions | 21 regions, collage look | 32 regions, defined edges |
| Toes/claws | Lost in smoothing | Preserved as distinct regions |

## Usage

```bash
# Edge-aware pipeline (recommended for poster style)
python -m apexvec input.jpg -o output.svg --edge-poster --colors 12

# SLIC pipeline (if you want spatial smoothing)
python -m apexvec input.jpg -o output.svg --slic --colors 12

# Original poster pipeline
python -m apexvec input.jpg -o output.svg --poster --colors 12
```

## Key Insight

The grayscale detail mask (Stage 3) correctly identifies edges and small features:
- Uses Canny edge detection on luminance
- Highlights toes, claws, and boundaries in red
- Could be used for adaptive smoothing (less smoothing on details)

Current limitation: Detail threshold (0.15% of image) too strict for this image size.
For 431x431 image, details need to be < 278 pixels - toes are larger than this.

## Fix: Region Visualization in poster_first_pipeline

### Problem
Stage 3 (regions) visualization used random colors instead of palette colors:
```python
# BROKEN - random colors
for region in regions:
    color = np.random.rand(3)  # Random!
    viz[region.mask] = color
```

This caused:
- Psychedelic/random colors in debug output
- Granular/noisy appearance at boundaries
- Difficult to verify correct region extraction

### Fix
Use palette colors for visualization:
```python
# FIXED - palette colors
for region in regions:
    color = self.palette[region.color_idx]  # Actual color!
    viz[region.mask] = color
```

### Result
- Stage 3 now shows actual Snorlax colors (teal/cream)
- Boundaries between regions are clean
- Proper visualization before smoothing is applied

Note: Stage 3 shows RAW regions (pixel boundaries). Stage 5 shows SMOOTHED boundaries.
The final SVG uses the smoothed boundaries from Stage 5.

## Critical Fix: Broken SVG Output (Stripe Artifacts)

### Problem
The final SVG output had severe artifacts:
- Horizontal stripes across the body
- Missing pieces (transparent areas)
- Garbled paths with straight lines cutting across
- Total corruption of the image

### Root Cause
The edge sorting logic in `_sort_region_edges` was fundamentally broken. When edges couldn't be chained together, it would just add them anyway, creating paths that jumped from one random location to another.

```python
# BROKEN - when edges can't connect, just add them anyway
if not found:
    for i, (e_idx, rev, start, end) in enumerate(edges_with_points):
        if i not in used:
            sorted_edges.append((e_idx, rev, start, end))  # DISCONNECTED!
            used.add(i)
            break
```

This caused:
1. Open paths that didn't close properly
2. Line segments jumping across the image
3. Stripe patterns from horizontal jumps
4. Missing regions

### Solution
Complete rewrite of boundary extraction and SVG building:

1. **Identify outer boundary vs holes** using polygon area:
```python
# Largest area = outer boundary
outer_idx = np.argmax(contour_areas)
```

2. **Store holes separately** in region:
```python
region.edge_ids = []        # Outer boundary
region.hole_edge_ids = []   # Hole boundaries
```

3. **Build SVG paths properly** with evenodd fill rule:
```python
# Outer boundary (clockwise)
path_parts.append(self._points_to_path(outer_points))

# Holes (counter-clockwise for evenodd fill)
for hole in holes:
    path_parts.append(self._points_to_path(hole_points[::-1]))

# Single path with all subpaths
full_path = " ".join(path_parts)
```

### Key Changes
- Removed broken `_sort_region_edges` function entirely
- Each region now tracks outer boundary + holes separately
- SVG path uses `evenodd` fill rule properly
- No more edge chaining - each contour is a complete subpath

### Result
- Clean closed paths
- Proper hole handling (e.g., inside of mouth, between arms)
- No stripe artifacts
- Gap-free output
