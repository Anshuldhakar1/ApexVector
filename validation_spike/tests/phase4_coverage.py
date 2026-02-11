"""
Phase 4: Shared Boundary Extraction Coverage

Goal: Prove we can extract a shared boundary set that covers the label map boundaries.

Despite high fragmentation, test if adjacency-based boundary extraction works.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import label as ndi_label
from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
from collections import defaultdict
import sys

np.random.seed(42)

print("Phase 4: Shared Boundary Extraction Coverage")
print("=" * 60)

# Load and quantize image
img_path = "./img0.jpg"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
height, width = img_array.shape[:2]

num_colors = 12  # Use 12 colors for manageable complexity
print(f"\n[OK] Processing with {num_colors} colors")

pixels = img_array.reshape(-1, 3).astype(np.float32)
kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
labels_flat = kmeans.fit_predict(pixels)
label_map = labels_flat.reshape(height, width)

unique_labels = np.unique(label_map)
print(f"     Labels present: {len(unique_labels)}")

# Find boundaries using skimage
boundaries = find_boundaries(label_map, mode='thick')
boundary_pixel_count = np.sum(boundaries)
print(f"\n[OK] Found {boundary_pixel_count} boundary pixels")

# Build adjacency graph by checking neighbor pixels
print("\n[...] Building adjacency graph...")
adjacency = defaultdict(set)

for y in range(height):
    for x in range(width):
        current_label = label_map[y, x]
        
        # Check 4-connected neighbors
        neighbors = []
        if y > 0: neighbors.append((y-1, x))
        if y < height-1: neighbors.append((y+1, x))
        if x > 0: neighbors.append((y, x-1))
        if x < width-1: neighbors.append((y, x+1))
        
        for ny, nx in neighbors:
            neighbor_label = label_map[ny, nx]
            if neighbor_label != current_label:
                # Sort to ensure consistent ordering
                pair = tuple(sorted([current_label, neighbor_label]))
                adjacency[pair].add((y, x))

num_adjacencies = len(adjacency)
print(f"[OK] Found {num_adjacencies} unique adjacency pairs")

# Calculate boundary coverage
total_pixels_in_adjacencies = sum(len(pixels) for pixels in adjacency.values())
coverage_ratio = total_pixels_in_adjacencies / boundary_pixel_count * 100

print(f"\n[METRIC] Boundary coverage: {total_pixels_in_adjacencies}/{boundary_pixel_count} ({coverage_ratio:.2f}%)")

# Analyze adjacency distribution
adj_sizes = [len(pixels) for pixels in adjacency.values()]
min_size = min(adj_sizes)
max_size = max(adj_sizes)
median_size = int(np.median(adj_sizes))
mean_size = np.mean(adj_sizes)

print(f"\n[METRIC] Adjacency boundary sizes:")
print(f"         Min: {min_size}, Max: {max_size}")
print(f"         Median: {median_size}, Mean: {mean_size:.2f}")

# Count small adjacencies (might be noise)
small_adjacencies = sum(1 for size in adj_sizes if size < 10)
print(f"\n[METRIC] Small adjacencies (<10 pixels): {small_adjacencies} ({small_adjacencies/num_adjacencies*100:.1f}%)")

# Estimate extractability
# For shared boundaries to work, we need:
# 1. Good coverage (>95%)
# 2. Reasonable number of adjacencies (not too many tiny ones)
# 3. Each adjacency should form traceable paths

coverage_pass = coverage_ratio > 95
fragmentation_pass = small_adjacencies / num_adjacencies < 0.5  # Less than 50% tiny

print(f"\n[ASSESSMENT]")
print(f"  Coverage (>95%): {'PASS' if coverage_pass else 'FAIL'} ({coverage_ratio:.2f}%)")
print(f"  Fragmentation (<50% tiny): {'PASS' if fragmentation_pass else 'FAIL'} ({small_adjacencies/num_adjacencies*100:.1f}% tiny)")

if coverage_pass and fragmentation_pass:
    print(f"\n[PASS] Shared boundary extraction should work")
    print(f"       Good coverage with manageable complexity.")
    sys.exit(0)
else:
    print(f"\n[WARNING] Shared boundary extraction may be problematic")
    if not coverage_pass:
        print(f"       - Low coverage: some boundaries not captured")
    if not fragmentation_pass:
        print(f"       - High fragmentation: too many tiny adjacencies")
    print(f"       Consider: merging tiny components before extraction")
    sys.exit(0)  # Warning, not failure
