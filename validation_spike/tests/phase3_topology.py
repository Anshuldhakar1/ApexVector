"""
Phase 3: Real Image Topology Check

Goal: Ensure the quantized label map is a clean planar partition suitable 
for shared-boundary extraction.

Measures:
- Orphan pixels / speckle count (tiny islands)
- Connected-component fragmentation per label
- Boundary pixel sanity
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import label as ndi_label
from skimage.segmentation import find_boundaries
import sys

np.random.seed(42)

print("Phase 3: Real Image Topology Check")
print("=" * 60)

# Load image
img_path = "./img0.jpg"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
height, width = img_array.shape[:2]
print(f"\n[OK] Loaded image: {img_path}")
print(f"     Dimensions: {width}x{height}")

# Test with 12 and 24 colors
color_counts = [12, 24]
topology_results = {}

for num_colors in color_counts:
    print(f"\n{'='*60}")
    print(f"Testing with {num_colors} colors")
    print(f"{'='*60}")
    
    # Flatten image for K-means
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    
    # K-means quantization
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    labels_flat = kmeans.fit_predict(pixels)
    label_map = labels_flat.reshape(height, width)
    
    print(f"\n[OK] Quantized to {num_colors} colors")
    
    # Count unique labels actually present
    unique_labels = np.unique(label_map)
    actual_colors = len(unique_labels)
    print(f"     Actual labels present: {actual_colors}")
    
    # Measure 1: Orphan pixels / speckle count
    # A speckle is a connected component smaller than some threshold
    speckle_threshold = 10  # pixels
    total_speckles = 0
    speckle_pixels = 0
    
    for label_idx in unique_labels:
        mask = (label_map == label_idx)
        labeled_array, num_features = ndi_label(mask)
        
        for region_id in range(1, num_features + 1):
            region_size = np.sum(labeled_array == region_id)
            if region_size < speckle_threshold:
                total_speckles += 1
                speckle_pixels += region_size
    
    total_pixels = height * width
    speckle_pct = (speckle_pixels / total_pixels) * 100
    
    print(f"\n[METRIC] Speckle count (<{speckle_threshold} pixels): {total_speckles}")
    print(f"         Speckle pixels: {speckle_pixels} ({speckle_pct:.4f}%)")
    
    # Measure 2: Connected component fragmentation per label
    components_per_label = []
    for label_idx in unique_labels:
        mask = (label_map == label_idx)
        _, num_components = ndi_label(mask)
        components_per_label.append(num_components)
    
    min_components = min(components_per_label)
    max_components = max(components_per_label)
    median_components = int(np.median(components_per_label))
    mean_components = np.mean(components_per_label)
    total_components = sum(components_per_label)
    
    print(f"\n[METRIC] Connected components per label:")
    print(f"         Min: {min_components}, Max: {max_components}")
    print(f"         Median: {median_components}, Mean: {mean_components:.2f}")
    print(f"         Total components: {total_components}")
    
    # Measure 3: Boundary pixel sanity
    # Find boundaries using skimage
    boundaries = find_boundaries(label_map, mode='thick')
    boundary_pixels = np.sum(boundaries)
    
    # Check boundary pixel adjacency
    # Each boundary pixel should separate 2 labels (except at borders)
    valid_boundary_count = 0
    suspicious_boundary_count = 0
    
    for y in range(1, height-1):
        for x in range(1, width-1):
            if boundaries[y, x]:
                # Get 3x3 neighborhood
                neighborhood = label_map[y-1:y+2, x-1:x+2]
                unique_in_neighborhood = np.unique(neighborhood)
                num_labels_adjacent = len(unique_in_neighborhood)
                
                # Should be 2 (one on each side) or more at junctions
                if num_labels_adjacent >= 2:
                    valid_boundary_count += 1
                else:
                    suspicious_boundary_count += 1
    
    print(f"\n[METRIC] Boundary pixels: {boundary_pixels}")
    print(f"         Valid (separates 2+ labels): {valid_boundary_count}")
    print(f"         Suspicious (separates <2 labels): {suspicious_boundary_count}")
    
    # Calculate fragmentation score
    # Lower is better - ideal is 1 component per label
    fragmentation_score = total_components / actual_colors
    
    # Store results
    topology_results[num_colors] = {
        'actual_colors': actual_colors,
        'speckles': total_speckles,
        'speckle_pixels': speckle_pixels,
        'speckle_pct': speckle_pct,
        'min_components': min_components,
        'max_components': max_components,
        'median_components': median_components,
        'mean_components': mean_components,
        'total_components': total_components,
        'fragmentation_score': fragmentation_score,
        'boundary_pixels': boundary_pixels,
        'valid_boundaries': valid_boundary_count,
        'suspicious_boundaries': suspicious_boundary_count
    }
    
    # Pass/fail assessment
    speckle_pass = speckle_pct < 0.01  # < 0.01% orphan pixels
    fragmentation_pass = fragmentation_score < 10  # Not too fragmented
    
    print(f"\n[ASSESSMENT]")
    print(f"  Speckle check (<0.01%): {'PASS' if speckle_pass else 'FAIL'} ({speckle_pct:.4f}%)")
    print(f"  Fragmentation (<10x): {'PASS' if fragmentation_pass else 'FAIL'} ({fragmentation_score:.2f}x)")
    print(f"  Overall: {'PASS' if (speckle_pass and fragmentation_pass) else 'WARNING'}")

# Summary
print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)

print(f"\n{'Colors':<10} {'Speckle %':<12} {'Frag Score':<12} {'Total Comp':<12} {'Status'}")
print("-" * 60)

for num_colors in color_counts:
    r = topology_results[num_colors]
    speckle_pass = r['speckle_pct'] < 0.01
    fragmentation_pass = r['fragmentation_score'] < 10
    status = "PASS" if (speckle_pass and fragmentation_pass) else "WARN"
    print(f"{num_colors:<10} {r['speckle_pct']:<12.4f} {r['fragmentation_score']:<12.2f} {r['total_components']:<12} {status}")

print("-" * 60)

# Determine if topology is suitable
all_pass = all(
    topology_results[n]['speckle_pct'] < 0.01 and 
    topology_results[n]['fragmentation_score'] < 10 
    for n in color_counts
)

if all_pass:
    print("\n[PASS] Topology is suitable for shared-boundary extraction")
    print("       Low speckle count and reasonable fragmentation.")
    sys.exit(0)
else:
    print("\n[WARNING] Topology may cause issues")
    print("          High fragmentation could make shared-edge tracing complex.")
    print("          Consider reducing colors or adding spatial regularization.")
    sys.exit(0)  # Still exit 0 - this is a warning, not a failure
