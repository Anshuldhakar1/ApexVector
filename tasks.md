This output reveals critical issues. Let me analyze what's wrong and create a targeted fix plan.

---

## Problem Diagnosis

| Issue | Evidence | Root Cause |
|-------|----------|------------|
| **Gaps/Transparency** | Red showing through (background), white fringes around regions | Per-region smoothing without shared boundaries; regions shrink independently |
| **Color Bleed/Desaturation** | Teal → washed blue-gray, cream → beige | Mean color computed from original pixels, not palette; or LAB→RGB conversion error |
| **Fragmented Regions** | Speckled texture, broken claws | Small regions merged incorrectly or boundary smoothing creating islands |
| **Lost Dark Colors** | No deep shadows, flat appearance | Dark regions dropped or merged into midtones |

The 60% that's working: quantization found roughly correct color clusters. The 40% failing: boundary handling and color fidelity.

---

## Revised Pipeline Plan: "Poster-First" Architecture

### Core Principle
**Quantized colors are law.** The palette from Stage 2 is immutable. Every pixel must map to exactly one palette entry. Boundaries are derived from the label map, then smoothed, but the fill colors never deviate from the palette.

---

## Stage-by-Stage Specification

### Stage 0: Ingest
- Load image, convert to linear RGB for processing
- Keep sRGB reference for final comparison

### Stage 1: Color Quantization (Locked Palette)
- K-means in LAB space, **but with spatial coherence weighting**
- Output: `label_map` (H×W of uint8 indices), `palette` (N×3 uint8 sRGB)
- **Invariant:** `np.unique(label_map)` must equal `range(len(palette))` — no dropped colors

### Stage 2: Region Extraction (Topology-Aware)
- Connected components per label
- **Critical:** Small regions merge only to **same-label neighbors** (spatial cleanup), never across colors
- Build adjacency graph: which region IDs touch which

### Stage 3: Shared Boundary Extraction
- For every pair of adjacent regions, extract the exact pixel boundary between them
- Store as polyline in **sub-pixel coordinates** (center of edge between pixels)
- **No smoothing yet**

### Stage 4: Gaussian Smoothing (Per-Shared-Edge)
- Apply Gaussian filter to each shared boundary polyline
- **Same curve used by both adjacent regions** (reversed direction)
- Parameter: `sigma` in pixels (1.0–2.0 typical)

### Stage 5: Region Reconstruction from Smoothed Edges
- For each region, collect its incident smoothed boundaries
- Sort into closed cycles (outer boundary + holes)
- **Fill color:** `palette[region.label_id]` — never recomputed

### Stage 6: SVG Export
- One `<path>` per region with `fill="#rrggbb"` from palette
- `fill-rule="evenodd"` for holes
- **No stroke.** No gradients. No opacity.

### Stage 7: Rasterization & Comparison
- Render SVG to PNG at original resolution
- Generate side-by-side: original | quantized | SVG output | difference mask

---

## Debug Visualization System

### Per-Stage Overlays (enabled by `--debug-stages`)

| Stage | Overlay | Purpose |
|-------|---------|---------|
| 1 Quantization | Original + color labels as faint tint | Verify color clusters match perception |
| 2 Regions | Original + region boundaries in random colors | Check region fragmentation |
| 3 Shared Boundaries | Black canvas + boundary lines in region colors | Verify topology extraction |
| 4 Smoothed Boundaries | Same as 3 but with smoothed curves | Validate smoothing doesn't cross |
| 5 Reconstructed | Filled regions with boundaries highlighted | Preview final appearance |
| 6 Final | SVG rasterized | Output validation |
| 7 Comparison | 4-panel: original, quantized, SVG, gap mask | End-to-end verification |

### Gap Mask Visualization
- Pixels that are transparent in SVG but non-transparent in original → **magenta**
- Pixels that are filled in SVG but transparent in original → **yellow** (shouldn't happen)
- Overlap where colors differ significantly → **red overlay**

---

## SVG-to-PNG Rasterizer Requirements

### Option A: CairoSVG (preferred)
- Pure Python, pip-installable
- Accurate geometry, no browser dependencies

### Option B: Inkscape CLI
- Higher quality, slower
- Requires Inkscape installed

### Option C: rsvg-convert (librsvg)
- Fast, accurate
- Requires system package

**Fallback chain:** Try CairoSVG → rsvg-convert → Inkscape → error with instructions.

---

## Self-Correcting Parameters

| Symptom | Diagnostic | Fix |
|---------|-----------|-----|
| White fringes (gaps) | Gap mask shows magenta | Reduce `sigma` or add dilation post-smoothing |
| Melted/blobby regions | Boundaries visually over-smoothed | Reduce `sigma` or preserve sharp corners via angle detection |
| Speckled texture | Stage 2 overlay shows many tiny regions | Increase spatial coherence in quantization, or raise minimum region area |
| Washed colors | Stage 7 shows palette ≠ output colors | Enforce `mean_color = palette[label_id]` in region struct |
| Lost shadows | Stage 1 overlay shows dark colors, Stage 6 doesn't | Check Stage 2 merger isn't dropping small dark regions |

---

## Implementation Order (Risk-Ordered)

1. **Fix color fidelity first** (easiest, highest impact)
   - Change `region.mean_color` assignment to use `palette` directly
   - Verify with Stage 7 comparison

2. **Implement shared boundary extraction** (architectural fix for gaps)
   - Replace per-region contour extraction with adjacency-aware edge tracing
   - Validate with Stage 3 overlay

3. **Add Gaussian smoothing per-edge** (aesthetic improvement)
   - Smooth once, use twice (both directions)
   - Validate with Stage 4 overlay + gap mask

4. **Debug visualization system** (enables rapid iteration)
   - Implement all stage overlays
   - Final validation with 4-panel comparison

---

## Success Criteria for `img0.jpg` (Snorlax test)

- [ ] No magenta pixels in gap mask (zero unintended transparency)
- [ ] Teal color in output matches input teal (within 5 RGB units)
- [ ] Cream belly color matches input (within 5 RGB units)
- [ ] Claws rendered as distinct regions, not fragmented
- [ ] Boundaries visibly smoother than raw pixel edges
- [ ] No white halo around figure



Clearup the directory after the task is completed. remove artifacts from differentt branches, and redudent folders and output files.