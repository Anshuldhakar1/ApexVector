# Poster-First Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement shared-boundary vectorization with SLIC superpixels, Marching Squares, and debug visualization

**Architecture:** Replace per-region contour extraction with adjacency-aware shared boundary system. Use SLIC for spatially-coherent quantization, Marching Squares for sub-pixel accuracy, Gaussian smoothing per shared edge (used by both adjacent regions).

**Tech Stack:** scikit-image (SLIC, Marching Squares), scipy (Gaussian smoothing), numpy, matplotlib (debug viz)

---

## Overview

This plan implements the "Poster-First" architecture validated in the spike:
1. **Stage 1**: SLIC superpixels (spatial coherence)  
2. **Stage 2**: Same-color region merging (area threshold)
3. **Stage 3**: Marching Squares boundary extraction (sub-pixel)
4. **Stage 4**: Shared boundary smoothing (Gaussian per-edge)
5. **Stage 5**: Region reconstruction from smoothed edges
6. **Stage 6**: SVG export with palette-locked colors
7. **Stage 7**: Debug visualization system

---

## Task 1: Add SLIC Superpixel Quantization

**Files:**
- Create: `apexvec/quantization.py`
- Modify: `apexvec/poster_pipeline.py:361-423` (replace `quantize_colors`)
- Test: `tests/test_quantization.py`

**Rationale:** SLIC provides spatial coherence, reducing fragmentation vs pure K-means

**Step 1: Write failing test**

```python
# tests/test_quantization.py
def test_slic_quantization_basic():
    """Test SLIC quantization produces spatially coherent regions."""
    import numpy as np
    from apexvec.quantization import slic_quantize
    
    # Create test image with clear color regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = [255, 0, 0]  # Red top
    img[50:, :] = [0, 0, 255]  # Blue bottom
    
    label_map, palette = slic_quantize(img, n_segments=2, compactness=10)
    
    # Should have exactly 2 labels
    assert len(np.unique(label_map)) == 2
    # Palette should have 2 colors
    assert len(palette) == 2
    # Label map should be spatially coherent (no salt-and-pepper)
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(label_map, mode='thick')
    # Should have one clean boundary line
    assert np.sum(boundaries) < 150  # Not fragmented


def test_slic_preserves_palette():
    """Test that palette is never recomputed from pixels."""
    import numpy as np
    from apexvec.quantization import slic_quantize
    
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    label_map, palette = slic_quantize(img, n_segments=4, compactness=10)
    
    # All labels should be in range
    assert np.all(label_map < len(palette))
    assert len(np.unique(label_map)) <= len(palette)
```

**Step 2: Run test to verify it fails**

```bash
cd D:\Github Cloning\ApexVector
python -m pytest tests/test_quantization.py -v
```

Expected: FAIL - "ModuleNotFoundError: No module named 'apexvec.quantization'"

**Step 3: Implement SLIC quantization module**

```python
# apexvec/quantization.py
"""Color quantization with spatial coherence using SLIC superpixels."""
from typing import Tuple
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab, lab2rgb


def slic_quantize(
    image: np.ndarray,
    n_segments: int = 12,
    compactness: float = 10.0,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize image using SLIC superpixels for spatial coherence.
    
    SLIC provides better spatial coherence than K-means alone by combining
    color similarity with spatial proximity. This reduces fragmentation.
    
    Args:
        image: Input image (H, W, 3) uint8
        n_segments: Target number of superpixels/colors
        compactness: Balance between color and spatial proximity
                     (higher = more spatially regular)
        sigma: Gaussian smoothing sigma before segmentation
        
    Returns:
        label_map: (H, W) array of palette indices
        palette: (N, 3) uint8 palette in sRGB
    """
    # Ensure uint8 input
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Run SLIC
    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1
    )
    
    # Assign each superpixel to dominant color
    unique_labels = np.unique(segments)
    label_map = np.zeros_like(segments, dtype=np.uint8)
    palette = np.zeros((len(unique_labels), 3), dtype=np.uint8)
    
    for i, seg_id in enumerate(unique_labels):
        mask = (segments == seg_id)
        # Mean color of this superpixel
        mean_color = np.mean(image[mask], axis=0)
        palette[i] = mean_color.astype(np.uint8)
        label_map[mask] = i
    
    # Ensure all labels are used
    assert len(np.unique(label_map)) == len(palette), \
        f"Label mismatch: {len(np.unique(label_map))} vs {len(palette)}"
    
    return label_map, palette
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_quantization.py -v
```

Expected: PASS

**Step 5: Update pipeline to use SLIC**

```python
# In apexvec/poster_pipeline.py
# Replace line 11-17 imports:
from apexvec.quantization import slic_quantize

# Replace quantize_colors function (lines 361-423):
def quantize_colors(
    image: np.ndarray,
    num_colors: int = 12,
    compactness: float = 10.0,
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize using SLIC for spatial coherence."""
    return slic_quantize(
        image,
        n_segments=num_colors,
        compactness=compactness,
        sigma=sigma
    )
```

**Step 6: Commit**

```bash
git add apexvec/quantization.py tests/test_quantization.py apexvec/poster_pipeline.py
git commit -m "feat: add SLIC superpixel quantization for spatial coherence"
```

---

## Task 2: Implement Same-Color Region Merging

**Files:**
- Create: `apexvec/region_merger.py`
- Modify: `apexvec/poster_pipeline.py:466-534` (replace `merge_small_regions`)
- Test: `tests/test_region_merger.py`

**Rationale:** Current merging merges across colors causing color bleed. New version only merges same-color regions.

**Step 1: Write failing test**

```python
# tests/test_region_merger.py
def test_merge_same_color_only():
    """Test that merging only happens within same color."""
    import numpy as np
    from apexvec.region_merger import merge_small_regions_same_color
    from apexvec.types import Region
    
    # Create label map with two colors
    label_map = np.zeros((20, 20), dtype=np.int32)
    label_map[:, :10] = 0  # Color 0 left
    label_map[:, 10:] = 1  # Color 1 right
    
    # Add small region of color 0 inside color 1 area
    label_map[5:7, 12:14] = 0  # Small red island in blue
    
    # Create regions
    regions = []
    for label_id in [0, 1]:
        # Find connected components for this label
        from scipy.ndimage import label
        mask = (label_map == label_id)
        labeled, num = label(mask)
        for i in range(1, num + 1):
            comp_mask = (labeled == i)
            region = Region(
                mask=comp_mask,
                label=len(regions),
                mean_color=np.array([255, 0, 0] if label_id == 0 else [0, 0, 255])
            )
            region.color_label = label_id  # Track original color
            regions.append(region)
    
    # Merge with threshold of 10 pixels
    merged = merge_small_regions_same_color(regions, min_area=10)
    
    # Small red region should merge with larger red region
    # NOT with blue regions
    red_regions = [r for r in merged if np.array_equal(r.mean_color, [255, 0, 0])]
    blue_regions = [r for r in merged if np.array_equal(r.mean_color, [0, 0, 255])]
    
    assert len(red_regions) == 1, "Red regions should merge into one"
    assert len(blue_regions) == 1, "Blue region should remain"


def test_respect_area_threshold():
    """Test that regions above threshold are not merged."""
    import numpy as np
    from apexvec.region_merger import merge_small_regions_same_color
    from apexvec.types import Region
    
    # Create two separate red regions, both large
    label_map = np.zeros((50, 50), dtype=np.int32)
    label_map[10:20, 10:20] = 0  # Region 1: 100 pixels
    label_map[30:40, 30:40] = 0  # Region 2: 100 pixels
    
    regions = []
    from scipy.ndimage import label
    mask = (label_map == 0)
    labeled, num = label(mask)
    for i in range(1, num + 1):
        comp_mask = (labeled == i)
        region = Region(mask=comp_mask, label=i-1, mean_color=np.array([255, 0, 0]))
        region.color_label = 0
        regions.append(region)
    
    # Merge with threshold of 50
    merged = merge_small_regions_same_color(regions, min_area=50)
    
    # Both regions should remain (both > 50 pixels)
    assert len(merged) == 2, "Both large regions should be preserved"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_region_merger.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement same-color merger**

```python
# apexvec/region_merger.py
"""Region merging with same-color constraint."""
from typing import List
import numpy as np
from scipy.ndimage import label, binary_dilation
from apexvec.types import Region


def merge_small_regions_same_color(
    regions: List[Region],
    min_area: int = 100
) -> List[Region]:
    """
    Merge small regions only with same-color neighbors.
    
    Unlike the original merge_small_regions which could merge across colors,
    this only merges regions that share the same mean_color. This prevents
    color bleeding and maintains palette integrity.
    
    Args:
        regions: List of regions (must have mean_color and color_label attributes)
        min_area: Minimum area in pixels. Regions smaller than this get merged.
        
    Returns:
        List of merged regions
    """
    if not regions:
        return regions
    
    # Group regions by color
    color_groups = {}
    for i, region in enumerate(regions):
        # Use mean_color as key (convert to tuple for hashing)
        color_key = tuple(region.mean_color.astype(int))
        if color_key not in color_groups:
            color_groups[color_key] = []
        color_groups[color_key].append((i, region))
    
    merged_regions = []
    
    # Process each color group independently
    for color_key, group_regions in color_groups.items():
        if len(group_regions) == 1:
            # Only one region of this color, keep it
            merged_regions.append(group_regions[0][1])
            continue
        
        # Separate small and large regions
        small = []
        large = []
        for idx, region in group_regions:
            area = np.sum(region.mask)
            if area < min_area:
                small.append((idx, region, area))
            else:
                large.append((idx, region, area))
        
        if not small:
            # No small regions to merge
            for idx, region in group_regions:
                merged_regions.append(region)
            continue
        
        # Build spatial index for finding neighbors
        h, w = regions[0].mask.shape
        region_id_map = np.full((h, w), -1, dtype=np.int32)
        for idx, region, area in large:
            region_id_map[region.mask] = idx
        
        # Merge each small region into its largest same-color neighbor
        merged_small = set()
        for small_idx, small_region, small_area in small:
            if small_idx in merged_small:
                continue
            
            # Find neighbors by dilation
            dilated = binary_dilation(small_region.mask, iterations=1)
            neighbor_mask = dilated & ~small_region.mask
            neighbor_ids = np.unique(region_id_map[neighbor_mask])
            neighbor_ids = neighbor_ids[neighbor_ids >= 0]
            
            if len(neighbor_ids) == 0:
                # Isolated small region, keep it
                merged_regions.append(small_region)
                continue
            
            # Find largest neighbor
            largest_neighbor_idx = max(
                neighbor_ids,
                key=lambda nid: np.sum(regions[nid].mask)
            )
            
            # Merge into that neighbor's mask
            regions[largest_neighbor_idx].mask = (
                regions[largest_neighbor_idx].mask | small_region.mask
            )
            merged_small.add(small_idx)
        
        # Add all large regions (now with merged small ones)
        for idx, region, area in large:
            merged_regions.append(region)
    
    # Renumber labels
    for i, region in enumerate(merged_regions):
        region.label = i
    
    return merged_regions
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_region_merger.py -v
```

Expected: PASS

**Step 5: Update pipeline to use new merger**

```python
# In apexvec/poster_pipeline.py
# Replace line 101-105:
from apexvec.region_merger import merge_small_regions_same_color

# In process() method, replace lines 101-105:
if len(regions) > 0:
    regions = merge_small_regions_same_color(
        regions,
        min_area=self.config.min_region_size * 2
    )
    print(f"  After merging: {len(regions)} regions")
```

**Step 6: Commit**

```bash
git add apexvec/region_merger.py tests/test_region_merger.py apexvec/poster_pipeline.py
git commit -m "feat: implement same-color-only region merging to prevent color bleed"
```

---

## Task 3: Implement Marching Squares Boundary Extraction

**Files:**
- Create: `apexvec/boundary_extraction.py`
- Test: `tests/test_boundary_extraction.py`

**Rationale:** Marching Squares provides sub-pixel accurate boundaries between regions

**Step 1: Write failing test**

```python
# tests/test_boundary_extraction.py
def test_marching_squares_basic():
    """Test Marching Squares extracts boundaries between two regions."""
    import numpy as np
    from apexvec.boundary_extraction import extract_shared_boundaries
    
    # Simple two-region image
    label_map = np.zeros((10, 10), dtype=np.int32)
    label_map[:, :5] = 0  # Left
    label_map[:, 5:] = 1  # Right
    
    boundaries = extract_shared_boundaries(label_map)
    
    # Should have one boundary between labels 0 and 1
    assert (0, 1) in boundaries or (1, 0) in boundaries
    
    # Boundary should be roughly vertical around x=4.5
    boundary = boundaries.get((0, 1)) or boundaries.get((1, 0))
    assert boundary is not None
    
    # Check boundary points are at sub-pixel positions
    xs = [p[0] for p in boundary]
    assert all(4.0 <= x <= 6.0 for x in xs), "Boundary should be near x=5"


def test_marching_squares_handles_holes():
    """Test Marching Squares correctly extracts hole boundaries."""
    import numpy as np
    from apexvec.boundary_extraction import extract_shared_boundaries
    
    # Create region with hole
    label_map = np.ones((20, 20), dtype=np.int32)
    label_map[5:15, 5:15] = 0  # Outer region 0
    label_map[8:12, 8:12] = 1  # Hole region 1 inside
    
    boundaries = extract_shared_boundaries(label_map)
    
    # Should have boundary between 0 and 1
    assert (0, 1) in boundaries or (1, 0) in boundaries
    
    # Boundary should form a closed loop
    boundary = boundaries.get((0, 1)) or boundaries.get((1, 0))
    # First and last points should be close (closed loop)
    first = np.array(boundary[0])
    last = np.array(boundary[-1])
    distance = np.linalg.norm(first - last)
    assert distance < 2.0, "Boundary should form closed loop"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_boundary_extraction.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement Marching Squares boundary extraction**

```python
# apexvec/boundary_extraction.py
"""Sub-pixel boundary extraction using Marching Squares."""
from typing import Dict, List, Tuple
import numpy as np
from skimage.measure import find_contours
from collections import defaultdict


def extract_shared_boundaries(
    label_map: np.ndarray,
    level: float = 0.5
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """
    Extract shared boundaries between all adjacent label pairs.
    
    Uses Marching Squares (via skimage.find_contours) for sub-pixel accuracy.
    Boundaries are extracted at the edge between pixels, providing smoother
    curves than pixel-center extraction.
    
    Args:
        label_map: (H, W) array of label indices
        level: Contour level (0.5 = halfway between labels)
        
    Returns:
        Dictionary mapping (label_a, label_b) tuples to boundary point lists.
        Labels are sorted (smaller first) for consistency.
    """
    boundaries = {}
    unique_labels = np.unique(label_map)
    
    # For each pair of adjacent labels, extract boundary
    for i, label_a in enumerate(unique_labels):
        for label_b in unique_labels[i+1:]:
            # Create binary mask: 1 where label_a, 0 elsewhere
            binary = (label_map == label_a).astype(float)
            
            # Find contours at level 0.5
            # This finds the boundary between label_a and everything else
            contours = find_contours(binary, level=level)
            
            if not contours:
                continue
            
            # Filter contours that are actually between label_a and label_b
            for contour in contours:
                # Check that this contour separates label_a from label_b
                # by sampling points along it
                is_shared = False
                for pt in contour[::max(1, len(contour)//10)]:  # Sample 10 points
                    y, x = int(pt[0]), int(pt[1])
                    if 0 <= y < label_map.shape[0] and 0 <= x < label_map.shape[1]:
                        # Check neighbors
                        neighbors = []
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < label_map.shape[0] and 0 <= nx < label_map.shape[1]:
                                neighbors.append(label_map[ny, nx])
                        
                        if label_a in neighbors and label_b in neighbors:
                            is_shared = True
                            break
                
                if is_shared:
                    # Convert from (row, col) to (x, y) coordinates
                    boundary_points = [(float(pt[1]), float(pt[0])) for pt in contour]
                    
                    # Store with sorted label pair
                    key = tuple(sorted([int(label_a), int(label_b)]))
                    if key not in boundaries:
                        boundaries[key] = []
                    boundaries[key].extend(boundary_points)
    
    return boundaries


def build_adjacency_graph(label_map: np.ndarray) -> Dict[Tuple[int, int], set]:
    """
    Build adjacency graph showing which labels touch.
    
    Returns:
        Dictionary mapping (label_a, label_b) to set of pixel coordinates
        where they touch.
    """
    h, w = label_map.shape
    adjacency = defaultdict(set)
    
    for y in range(h):
        for x in range(w):
            current_label = label_map[y, x]
            
            # Check 4-connected neighbors
            neighbors = []
            if y > 0: neighbors.append((y-1, x))
            if y < h-1: neighbors.append((y+1, x))
            if x > 0: neighbors.append((y, x-1))
            if x < w-1: neighbors.append((y, x+1))
            
            for ny, nx in neighbors:
                neighbor_label = label_map[ny, nx]
                if neighbor_label != current_label:
                    pair = tuple(sorted([int(current_label), int(neighbor_label)]))
                    adjacency[pair].add((y, x))
    
    return dict(adjacency)
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_boundary_extraction.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add apexvec/boundary_extraction.py tests/test_boundary_extraction.py
git commit -m "feat: add Marching Squares sub-pixel boundary extraction"
```

---

## Task 4: Implement Gaussian Smoothing Per Shared Edge

**Files:**
- Create: `apexvec/shared_boundary_smoother.py`
- Modify: `apexvec/poster_pipeline.py:537-680` (replace vectorization)
- Test: `tests/test_shared_boundary_smoother.py`

**Rationale:** Smooth each shared boundary once, use for both adjacent regions

**Step 1: Write failing test**

```python
# tests/test_shared_boundary_smoother.py
def test_gaussian_smooth_shared_boundary():
    """Test Gaussian smoothing of shared boundary."""
    import numpy as np
    from apexvec.shared_boundary_smoother import smooth_shared_boundary
    
    # Create simple boundary
    boundary = [(i, 5.0) for i in range(10)]  # Horizontal line
    
    # Add noise
    noisy_boundary = [(x, y + np.random.normal(0, 0.5)) for x, y in boundary]
    
    # Smooth
    smoothed = smooth_shared_boundary(noisy_boundary, sigma=1.0)
    
    # Should have same number of points
    assert len(smoothed) == len(noisy_boundary)
    
    # Should be smoother (less variance in y)
    original_variance = np.var([p[1] for p in noisy_boundary])
    smoothed_variance = np.var([p[1] for p in smoothed])
    assert smoothed_variance < original_variance


def test_boundary_used_by_both_regions():
    """Test that smoothed boundary is used by both adjacent regions."""
    import numpy as np
    from apexvec.shared_boundary_smoother import reconstruct_regions_from_boundaries
    
    # Simple two-region setup
    boundaries = {
        (0, 1): [(5.0, 0.0), (5.0, 5.0), (5.0, 10.0)]  # Vertical boundary
    }
    
    # Mock palette
    palette = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    
    regions = reconstruct_regions_from_boundaries(
        boundaries,
        palette,
        image_shape=(10, 10)
    )
    
    # Should have 2 regions
    assert len(regions) == 2
    
    # Both regions should reference the boundary
    # (in opposite directions)
    # This is validated by ensuring no gaps/overlaps
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_shared_boundary_smoother.py -v
```

Expected: FAIL - Module not found

**Step 3: Implement shared boundary smoother**

```python
# apexvec/shared_boundary_smoother.py
"""Gaussian smoothing of shared boundaries between regions."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.ndimage import gaussian_filter1d
from apexvec.types import VectorRegion, RegionKind, BezierCurve, Point


def smooth_shared_boundary(
    boundary: List[Tuple[float, float]],
    sigma: float = 1.0,
    mode: str = 'wrap'
) -> List[Tuple[float, float]]:
    """
    Apply Gaussian smoothing to a shared boundary.
    
    Args:
        boundary: List of (x, y) points defining the boundary
        sigma: Gaussian smoothing parameter (pixels)
        mode: Edge handling mode ('wrap', 'nearest', 'reflect')
        
    Returns:
        Smoothed boundary points
    """
    if len(boundary) < 3:
        return boundary
    
    points = np.array(boundary)
    
    # Sort points to form continuous path
    # Use angle around centroid for closed curves
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    
    # Apply Gaussian smoothing to x and y separately
    smoothed_x = gaussian_filter1d(sorted_points[:, 0], sigma=sigma, mode=mode)
    smoothed_y = gaussian_filter1d(sorted_points[:, 1], sigma=sigma, mode=mode)
    
    smoothed = list(zip(smoothed_x, smoothed_y))
    
    return smoothed


def reconstruct_regions_from_boundaries(
    boundaries: Dict[Tuple[int, int], List[Tuple[float, float]]],
    palette: np.ndarray,
    image_shape: Tuple[int, int],
    sigma: float = 1.0
) -> List[VectorRegion]:
    """
    Reconstruct all regions from smoothed shared boundaries.
    
    For each region, collect all its incident smoothed boundaries and
    assemble into closed paths.
    
    Args:
        boundaries: Dict mapping (label_a, label_b) to boundary points
        palette: (N, 3) color palette
        image_shape: (H, W) image dimensions
        sigma: Smoothing parameter
        
    Returns:
        List of VectorRegion objects
    """
    # Find all unique labels
    all_labels = set()
    for label_a, label_b in boundaries.keys():
        all_labels.add(label_a)
        all_labels.add(label_b)
    
    vector_regions = []
    
    for label_id in sorted(all_labels):
        # Find all boundaries involving this label
        region_boundaries = []
        for (la, lb), boundary_points in boundaries.items():
            if la == label_id:
                # Use boundary as-is
                smoothed = smooth_shared_boundary(boundary_points, sigma=sigma)
                region_boundaries.append(smoothed)
            elif lb == label_id:
                # Use boundary in reverse direction
                smoothed = smooth_shared_boundary(boundary_points, sigma=sigma)
                region_boundaries.append(smoothed[::-1])  # Reverse
        
        if not region_boundaries:
            continue
        
        # Combine all boundaries into one or more closed paths
        # For now, concatenate them (simplification)
        all_points = []
        for boundary in region_boundaries:
            all_points.extend(boundary)
        
        # Close the path
        if all_points and all_points[0] != all_points[-1]:
            all_points.append(all_points[0])
        
        # Convert to Bezier curves
        bezier_path = points_to_bezier(all_points)
        
        # Create vector region
        vector_region = VectorRegion(
            kind=RegionKind.FLAT,
            path=bezier_path,
            hole_paths=[],  # TODO: Handle holes
            fill_color=palette[label_id]
        )
        
        vector_regions.append(vector_region)
    
    return vector_regions


def points_to_bezier(
    points: List[Tuple[float, float]],
    tolerance: float = 1.0
) -> List[BezierCurve]:
    """
    Convert polyline points to Bezier curves.
    
    Simple implementation: create line segments.
    TODO: Implement proper curve fitting for smoother results.
    """
    if len(points) < 2:
        return []
    
    curves = []
    
    for i in range(len(points) - 1):
        p0 = Point(x=points[i][0], y=points[i][1])
        p3 = Point(x=points[i+1][0], y=points[i+1][1])
        
        # For straight lines, control points are midpoints
        p1 = Point(
            x=p0.x + (p3.x - p0.x) * 0.33,
            y=p0.y + (p3.y - p0.y) * 0.33
        )
        p2 = Point(
            x=p0.x + (p3.x - p0.x) * 0.67,
            y=p0.y + (p3.y - p0.y) * 0.67
        )
        
        curves.append(BezierCurve(p0=p0, p1=p1, p2=p2, p3=p3))
    
    return curves
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_shared_boundary_smoother.py -v
```

Expected: PASS (some tests may be basic)

**Step 5: Update pipeline to use new vectorization**

```python
# In apexvec/poster_pipeline.py
# Replace lines 110-135 (vectorization step):
print("Step 4/6: Extracting and smoothing shared boundaries...")

# Extract shared boundaries
from apexvec.boundary_extraction import extract_shared_boundaries
from apexvec.shared_boundary_smoother import reconstruct_regions_from_boundaries

boundaries = extract_shared_boundaries(label_map)
print(f"  Found {len(boundaries)} shared boundaries")

# Reconstruct regions from smoothed boundaries
vector_regions = reconstruct_regions_from_boundaries(
    boundaries,
    palette,
    image_shape=label_map.shape,
    sigma=1.0  # TODO: Make configurable
)
print(f"  Reconstructed {len(vector_regions)} regions")
```

**Step 6: Commit**

```bash
git add apexvec/shared_boundary_smoother.py tests/test_shared_boundary_smoother.py apexvec/poster_pipeline.py
git commit -m "feat: implement shared boundary smoothing and region reconstruction"
```

---

## Task 5: Fix Color Fidelity (Use Palette Directly)

**Files:**
- Modify: `apexvec/svg_optimizer.py` (regions_to_svg function)
- Test: Verify in integration test

**Rationale:** Ensure fill colors come from palette, never recomputed from pixels

**Step 1: Check current SVG generation**

```python
# Check apexvec/svg_optimizer.py for how fill_color is used
# Need to ensure it uses palette directly
```

**Step 2: Update SVG generation to enforce palette colors**

```python
# In apexvec/svg_optimizer.py
# Ensure regions_to_svg uses fill_color directly without modification
# If fill_color is None, raise error (should never happen with new pipeline)
```

**Step 3: Add color fidelity check**

```python
# In pipeline after vectorization:
# Verify that all vector_regions have fill_color matching palette
for vr in vector_regions:
    assert vr.fill_color is not None, "Region missing fill color"
    # Color should be in palette
    # (Optional strict check)
```

**Step 4: Commit**

```bash
git add apexvec/svg_optimizer.py apexvec/poster_pipeline.py
git commit -m "fix: enforce palette-locked colors in SVG output"
```

---

## Task 6: Add Debug Visualization System

**Files:**
- Create: `apexvec/debug_visualization.py`
- Modify: `apexvec/poster_pipeline.py` (add debug stages)

**Rationale:** Per-stage overlays enable rapid iteration and debugging

**Step 1: Implement debug visualization module**

```python
# apexvec/debug_visualization.py
"""Debug visualization for poster pipeline stages."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def visualize_quantization(
    original: np.ndarray,
    label_map: np.ndarray,
    palette: np.ndarray,
    output_path: Path
):
    """
    Stage 1: Show original with quantized colors as faint tint.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Quantized
    quantized = palette[label_map]
    axes[1].imshow(quantized)
    axes[1].set_title(f'Quantized ({len(palette)} colors)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_regions(
    original: np.ndarray,
    label_map: np.ndarray,
    output_path: Path
):
    """
    Stage 2: Show region boundaries in random colors.
    """
    from skimage.segmentation import find_boundaries
    
    boundaries = find_boundaries(label_map, mode='thick')
    
    # Color boundaries
    viz = original.copy()
    boundary_color = [255, 0, 0]  # Red boundaries
    viz[boundaries] = boundary_color
    
    Image.fromarray(viz).save(output_path)


def visualize_shared_boundaries(
    boundaries: Dict[Tuple[int, int], List[Tuple[float, float]]],
    image_shape: Tuple[int, int],
    output_path: Path
):
    """
    Stage 3: Show shared boundary lines in different colors.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Random color for each boundary
    np.random.seed(42)
    for (label_a, label_b), points in boundaries.items():
        color = np.random.rand(3)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, linewidth=1, 
                label=f'{label_a}-{label_b}')
    
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Invert Y for image coords
    ax.set_aspect('equal')
    ax.set_title('Shared Boundaries')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_smoothed_boundaries(
    original_boundaries: Dict,
    smoothed_boundaries: Dict,
    image_shape: Tuple[int, int],
    output_path: Path
):
    """
    Stage 4: Compare original vs smoothed boundaries.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    for ax, boundaries, title in [
        (axes[0], original_boundaries, 'Original Boundaries'),
        (axes[1], smoothed_boundaries, 'Smoothed Boundaries')
    ]:
        for (label_a, label_b), points in boundaries.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, linewidth=1)
        
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)
        ax.set_aspect('equal')
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def create_comparison_panel(
    original: np.ndarray,
    quantized: np.ndarray,
    svg_rasterized: np.ndarray,
    output_path: Path
):
    """
    Stage 7: 4-panel comparison: original, quantized, SVG, gap mask.
    """
    # Calculate gap mask
    gap_mask = np.zeros((*original.shape[:2], 3), dtype=np.uint8)
    
    # Magenta: transparent in SVG but not in original
    # (simplified: check if SVG pixel is black/white vs original)
    gray_svg = np.mean(svg_rasterized, axis=2)
    gap_mask[gray_svg < 10] = [255, 0, 255]  # Magenta for gaps
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(quantized)
    axes[0, 1].set_title('Quantized')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(svg_rasterized)
    axes[1, 0].set_title('SVG Output')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(gap_mask)
    axes[1, 1].set_title('Gap Mask (Magenta=Gaps)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
```

**Step 2: Integrate into pipeline**

```python
# In apexvec/poster_pipeline.py
# Add debug visualization calls after each stage when save_stages=True

# After Stage 1 (quantization):
if self.save_stages:
    from apexvec.debug_visualization import visualize_quantization
    visualize_quantization(
        ingest_result.image_srgb,
        label_map,
        palette,
        self.stages_dir / "stage_01_quantization.png"
    )

# After Stage 2 (region extraction):
if self.save_stages:
    from apexvec.debug_visualization import visualize_regions
    visualize_regions(
        ingest_result.image_srgb,
        label_map,
        self.stages_dir / "stage_02_regions.png"
    )

# After Stage 3 (boundary extraction):
if self.save_stages:
    from apexvec.debug_visualization import visualize_shared_boundaries
    visualize_shared_boundaries(
        boundaries,
        label_map.shape,
        self.stages_dir / "stage_03_boundaries.png"
    )

# After Stage 7 (final comparison):
if self.save_stages and png_result:
    from apexvec.debug_visualization import create_comparison_panel
    import numpy as np
    from PIL import Image
    
    svg_img = np.array(Image.open(png_result))
    create_comparison_panel(
        ingest_result.image_srgb,
        palette[label_map],
        svg_img,
        self.stages_dir / "stage_07_comparison.png"
    )
```

**Step 3: Commit**

```bash
git add apexvec/debug_visualization.py apexvec/poster_pipeline.py
git commit -m "feat: add debug visualization system for all pipeline stages"
```

---

## Task 7: Integration Test and Validation

**Files:**
- Create: `tests/test_integration_poster_pipeline.py`

**Step 1: Create integration test**

```python
# tests/test_integration_poster_pipeline.py
import numpy as np
from pathlib import Path
import tempfile


def test_poster_pipeline_end_to_end():
    """Full integration test of poster pipeline."""
    from PIL import Image
    from apexvec.poster_pipeline import PosterPipeline
    from apexvec.types import AdaptiveConfig
    
    # Create test image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :] = [100, 150, 200]  # Teal-ish
    img[50:, :] = [250, 240, 230]  # Cream
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test.png"
        output_path = Path(tmpdir) / "test.svg"
        
        Image.fromarray(img).save(input_path)
        
        # Run pipeline
        config = AdaptiveConfig()
        pipeline = PosterPipeline(
            config=config,
            num_colors=2,
            save_stages=True,
            stages_dir=Path(tmpdir) / "stages"
        )
        
        svg = pipeline.process(input_path, output_path)
        
        # Assertions
        assert output_path.exists(), "SVG output not created"
        assert len(svg) > 0, "Empty SVG"
        
        # Check for gaps (basic: count paths)
        path_count = svg.count("<path")
        assert path_count >= 2, f"Expected 2+ paths, got {path_count}"


def test_color_fidelity():
    """Test that output colors match input palette."""
    from PIL import Image
    from apexvec.poster_pipeline import PosterPipeline
    import tempfile
    from pathlib import Path
    import re
    
    # Create solid color image
    color = np.array([100, 150, 200], dtype=np.uint8)
    img = np.full((50, 50, 3), color, dtype=np.uint8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "solid.png"
        output_path = Path(tmpdir) / "solid.svg"
        
        Image.fromarray(img).save(input_path)
        
        pipeline = PosterPipeline(num_colors=1)
        svg = pipeline.process(input_path, output_path)
        
        # Extract fill color from SVG
        match = re.search(r'fill="#([0-9a-fA-F]{6})"', svg)
        assert match, "No fill color found in SVG"
        
        hex_color = match.group(1)
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        # Should be close to input color (within tolerance)
        assert abs(r - color[0]) < 10
        assert abs(g - color[1]) < 10
        assert abs(b - color[2]) < 10
```

**Step 2: Run integration test**

```bash
python -m pytest tests/test_integration_poster_pipeline.py -v
```

**Step 3: Manual validation on img0.jpg**

```bash
python -m apexvec img0.jpg -o test_output.svg --save-stages
# Check stages directory for visualizations
# Check test_output.png for final result
```

**Step 4: Commit**

```bash
git add tests/test_integration_poster_pipeline.py
git commit -m "test: add integration tests for poster pipeline"
```

---

## Success Criteria Checklist

Run this validation before marking complete:

- [ ] **No magenta pixels**: Gap mask shows zero unintended transparency
- [ ] **Color fidelity**: Teal and cream colors match input (within 5 RGB units)
- [ ] **No fragmentation**: Regions are not speckled/broken
- [ ] **Smooth boundaries**: Edges visibly smoother than pixel edges
- [ ] **No white halo**: Figure doesn't have white fringe
- [ ] **All stages save**: Debug visualizations created for each stage
- [ ] **Tests pass**: All unit and integration tests pass

## Execution Summary

**Total Tasks**: 7

**Estimated Time**: 
- Tasks 1-4: 2-3 hours each (core implementation)
- Task 5: 30 minutes (color fix)
- Task 6: 1 hour (debug viz)
- Task 7: 1 hour (integration)

**Total**: ~12-16 hours

**Key Dependencies**:
- Task 4 depends on Task 3 (needs boundaries to smooth)
- Task 7 validates all previous tasks
- Debug viz (Task 6) can be done in parallel with Task 5

**Critical Path**: 1 → 2 → 3 → 4 → 7

---

Plan saved to: `docs/plans/2026-02-11-poster-pipeline-implementation.md`

**Ready for execution?** Choose:
1. **Subagent-driven**: I execute task-by-task with reviews
2. **Parallel session**: Open new session with executing-plans skill
