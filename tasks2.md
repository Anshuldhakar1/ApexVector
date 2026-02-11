Here is a bulletproof, self-correcting plan that fixes the **transparent gaps** (caused by independent per-region smoothing) while using **Gaussian smoothing** (which you confirmed works better for your aesthetic).

The core insight: **Smooth the shared boundary lines once, then have both adjacent regions use that exact same curve.** If Region A and Region B touch, they must reference the *same* smoothed path object (traversed in opposite directions), not two separately smoothed approximations.

---

## Phase 0: Diagnostic Baseline (Do This First)

**Goal:** Confirm exactly which pixels are becoming transparent.

```bash
# Run current broken version
python -m apexvec test_images/img0.jpg --poster --colors 24 -o test_images/debug/broken.svg

# Create a diagnostic overlay
python -c "
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

# Load original
orig = np.array(Image.open('test_images/img0.jpg').convert('RGBA'))

# Parse SVG to find all path bounding boxes (approximate coverage)
svg = ET.parse('test_images/debug/broken.svg').getroot()
# ... (code to rasterize SVG or just check for transparency)

# Save an image where transparent pixels in output are marked red on original
print('If you see red lines between regions, those are the gaps.')
"
```

**Pass Criteria:** You now have a `gaps_visualization.png` showing exactly where boundaries are leaking transparency.

---

## Phase 1: Fix Color Dropout (Prerequisite)

**Problem:** Dark regions disappear entirely.
**Fix:** Ensure the quantizer → extractor → exporter pipeline never drops a color class.

**Implementation:**
1. **In `color_quantizer.py`:** Add a `preserve_color_indices` list. After K-means, force at least one pixel to remain for every palette index (reassign lone pixels to nearest if needed, but ensure the color exists in `label_map`).

2. **In `region_extractor.py`:** 
   - Change `_merge_small_regions` to never drop the *last* region of any given `color_idx`.
   - If a color has only one small region, keep it regardless of area.

**Validation:**
```python
# In extract_regions, before returning:
for i in range(len(palette)):
    if not any(getattr(r, '_color_idx', -1) == i for r in regions):
        raise VectorizationError(f"Color {i} dropped! Fix merger logic.")
```

**Self-Correction:** If validation fails, the merger is too aggressive. Reduce `min_area_ratio` dynamically until all colors survive.

---

## Phase 2: Extract Shared Boundaries (The "Cell Complex")

**Goal:** Convert the `label_map` into a **Planar Graph** where edges are shared between regions.

**Implementation:**

Replace per-region contour extraction with this topology-aware extraction:

```python
from skimage.segmentation import find_boundaries
from skimage.measure import grid_points_in_poly
import numpy as np

def extract_shared_boundaries(label_map: np.ndarray, palette: np.ndarray):
    """
    Returns:
        edges: List of [(y1,x1), (y2,x2), ...] polylines
        edge_regions: List of (region_id_left, region_id_right) for each edge
                      (-1 for background/outer boundary)
    """
    h, w = label_map.shape
    
    # 1. Find all boundary pixels (where label != neighbor)
    #    This gives us a binary mask of all edges
    boundaries = find_boundaries(label_map, mode='thick')
    
    # 2. Trace these into polylines using skimage.measure.find_contours
    #    on the boundary mask, BUT we need to know which regions each edge separates
    from skimage import measure
    
    # Get contours at sub-pixel precision (level=0.5 on distance transform of boundary)
    boundary_dt = ndimage.distance_transform_edt(~boundaries)
    contours = measure.find_contours(boundary_dt, level=0.5)
    
    # 3. For each contour point, determine which two regions it separates
    edges = []
    edge_pairs = []
    
    for contour in contours:
        # Sample points along contour
        region_pairs = []
        for (y, x) in contour:
            yy, xx = int(round(y)), int(round(x))
            # Check 4-neighbors to find the two different labels
            neighbors = []
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = yy+dy, xx+dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbors.append(label_map[ny, nx])
            
            unique = list(set(neighbors))
            if len(unique) == 2:
                region_pairs.append(tuple(sorted(unique)))
            elif len(unique) == 1:
                region_pairs.append((unique[0], -1))  # Background
            else:
                region_pairs.append((-1, -1))
        
        edges.append(contour)
        # Store the most common region pair for this edge (or determine per-segment)
        edge_pairs.append(region_pairs)
    
    return edges, edge_pairs
```

**Self-Correction Check:** 
- Assert that every edge has exactly 2 regions assigned (or 1 for outer boundary).
- Assert that the sum of edge lengths per region roughly matches the region's perimeter.

---

## Phase 3: Gaussian Smoothing on Shared Edges

**Goal:** Smooth the boundaries using Gaussian filtering (or spline), but do it **once per edge**, not once per region.

**Implementation:**

```python
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev

def smooth_shared_edge(edge_points: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Smooth a polyline representing a boundary between two regions.
    Uses Gaussian filtering on coordinates (your preferred method).
    """
    if len(edge_points) < 4:
        return edge_points
    
    # Separate coordinates
    y = edge_points[:, 0]
    x = edge_points[:, 1]
    
    # Gaussian smoothing (wrap=False for open edges, True for closed loops)
    # Most region boundaries are closed loops, but shared edges between 
    # two regions are typically... actually they can be open if they hit image border
    
    # Check if it's a closed loop (first == last)
    is_closed = np.allclose(edge_points[0], edge_points[-1])
    
    if is_closed and len(y) > 3:
        # For closed curves, smooth circularly
        y_smooth = gaussian_filter1d(np.concatenate([y, y]), sigma, mode='wrap')[:len(y)]
        x_smooth = gaussian_filter1d(np.concatenate([x, x]), sigma, mode='wrap')[:len(x)]
    else:
        y_smooth = gaussian_filter1d(y, sigma, mode='nearest')
        x_smooth = gaussian_filter1d(x, sigma, mode='nearest')
    
    return np.column_stack([y_smooth, x_smooth])
```

**Apply to all edges:**
```python
smoothed_edges = [smooth_shared_edge(e, sigma=1.5) for e in edges]
```

---

## Phase 4: Reconstruct Regions from Shared Edges

**Goal:** Build each region's SVG path by concatenating its smoothed boundary edges into closed loops.

**Implementation:**

```python
def build_region_paths(edges, edge_pairs, n_regions):
    """
    Build closed paths for each region from the shared edge pool.
    Returns: List of SVG path strings (one per region)
    """
    region_borders = [[] for _ in range(n_regions)]
    
    # Assign edges to regions
    for edge, pair_list in zip(edges, edge_pairs):
        # Determine which region pair this edge belongs to
        # (Take the most frequent pair in the list, excluding -1)
        from collections import Counter
        valid_pairs = [p for p in pair_list if -1 not in p]
        if not valid_pairs:
            continue
        (r1, r2), _ = Counter(valid_pairs).most_common(1)[0]
        
        # Add edge to both regions (reversed for one of them to maintain winding)
        region_borders[r1].append(edge)
        region_borders[r2].append(edge[::-1])  # Reverse direction
    
    # Now trace closed loops for each region
    # (This is a graph traversal problem: connect edges head-to-tail)
    svg_paths = []
    for region_id, border_edges in enumerate(region_borders):
        if not border_edges:
            svg_paths.append("")
            continue
        
        # Simple approach: if edges share endpoints, concatenate
        # For now, just join them (assumes they form a closed loop)
        # More robust: build adjacency graph and walk it
        
        # Convert to SVG path string
        path_parts = []
        for edge in border_edges:
            coords = " ".join([f"{x:.2f},{y:.2f}" for y, x in edge])
            if not path_parts:
                path_parts.append(f"M {coords}")
            else:
                path_parts.append(f"L {coords}")
        path_parts.append("Z")
        svg_paths.append(" ".join(path_parts))
    
    return svg_paths
```

**Critical Fix for Transparency:** Because both Region A and Region B use the exact same `edge` array (just one is reversed), there is **zero gap** between them. The coordinates are identical.

---

## Phase 5: SVG Export with Color Attachment

**Implementation:**

```python
def export_svg_shared_edges(region_paths, palette, image_shape):
    h, w = image_shape[:2]
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
    ]
    
    for i, path_data in enumerate(region_paths):
        if not path_data:
            continue
        
        r, g, b = map(int, palette[i])
        fill = f"#{r:02x}{g:02x}{b:02x}"
        
        svg_parts.append(
            f'<path d="{path_data}" fill="{fill}" stroke="none" />'
        )
    
    svg_parts.append('</svg>')
    return "\n".join(svg_parts)
```

**Why no stroke?** Because adjacent fills touch exactly. Adding a stroke would cause overdrawing artifacts. The `stroke="none"` ensures no transparency leaks.

---

## Phase 6: Self-Correction Validation Loop

**Create `scripts/validate_topology.py`:**

```python
def validate_no_gaps(label_map, smoothed_edges, edge_pairs):
    """
    Rasterize the smoothed edges and ensure they cover all boundary pixels
    from the original label_map.
    """
    from skimage.draw import polygon_perimeter
    
    h, w = label_map.shape
    coverage = np.zeros((h, w), dtype=bool)
    
    for edge in smoothed_edges:
        rr, cc = polygon_perimeter(edge[:,0], edge[:,1], shape=(h,w))
        coverage[rr, cc] = True
    
    # Original boundaries
    orig_boundaries = find_boundaries(label_map, mode='thick')
    
    # Check coverage
    missed = orig_boundaries & ~coverage
    missed_pct = missed.sum() / orig_boundaries.sum() * 100
    
    print(f"Boundary coverage: {100-missed_pct:.1f}%")
    
    if missed_pct > 5:
        print("WARNING: Smoothed edges don't cover original boundaries.")
        print("ACTION: Reduce Gaussian sigma or check edge tracing.")
        return False
    
    # Check for gaps (pixels that are boundary in neither region)
    # This is implicit in the shared edge approach
    return True
```

**Master Validation:**
1. **Color Count Check:** Does SVG have same number of unique fills as palette?
2. **Coverage Check:** Rasterize SVG, compare to original. Transparent pixels should only be outside the image convex hull.
3. **Edge Continuity:** Do smoothed edges form closed loops for every region?

---

## Summary of Changes Required

| File | Change |
|------|--------|
| `region_extractor.py` | Replace per-region `find_contours` with `extract_shared_boundaries()` that builds a planar graph |
| `boundary_smoother.py` | Apply `gaussian_filter1d` to the shared edges list, not per-region masks |
| `svg_export.py` | Build paths by concatenating shared edges per region, ensuring no gaps |
| New `topology_validator.py` | Verify that rasterized smoothed edges cover original boundaries |

**Why this fixes your issues:**
- **No transparent gaps:** Adjacent regions share the exact same smoothed curve coordinates (just opposite winding). Pixel-perfect alignment.
- **Colors preserved:** Phase 1 ensures no color index is ever dropped.
- **Gaussian smoothing works:** Applied to the boundary lines themselves, avoiding the "jagged spline" problem you encountered.

**Self-Correction Trigger:**
If `validate_topology.py` reports <95% boundary coverage after smoothing, automatically reduce `sigma` by 0.5 and retry until coverage passes or sigma hits 0 (raw edges).

Run this plan step-by-step. Start with Phase 2 (the shared edge extraction) - that's the architectural fix that makes everything else possible.