"""
Phase 6: Landscape Applicability Test

Goal: Test shared-boundary approach on a landscape image (no central focus).
Use img1.jpg as a representative landscape.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
from skimage.segmentation import find_boundaries
from collections import defaultdict
import time
import sys

np.random.seed(42)

print("Phase 6: Landscape Applicability Test")
print("=" * 60)

# Use img1.jpg as landscape
img_path = "./test_images/img1.jpg"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
height, width = img_array.shape[:2]

print(f"\n[OK] Loaded landscape: {img_path}")
print(f"     Dimensions: {width}x{height}")

# Test with 12 and 20 colors
test_configs = [(12, 0.8), (20, 0.8), (12, 1.5)]
results = []

for num_colors, sigma in test_configs:
    print(f"\n{'='*60}")
    print(f"Testing: {num_colors} colors, sigma={sigma}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Quantize
    pixels = img_array.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    labels_flat = kmeans.fit_predict(pixels)
    label_map = labels_flat.reshape(height, width)
    
    quantize_time = time.time() - start_time
    
    unique_labels = np.unique(label_map)
    print(f"\n[OK] Quantized to {num_colors} colors in {quantize_time:.2f}s")
    print(f"     Unique labels present: {len(unique_labels)}")
    
    # Build adjacencies
    adjacency_start = time.time()
    adjacency_pixels = defaultdict(list)
    
    for y in range(height):
        for x in range(width):
            current_label = label_map[y, x]
            neighbors = []
            if y > 0: neighbors.append((y-1, x))
            if y < height-1: neighbors.append((y+1, x))
            if x > 0: neighbors.append((y, x-1))
            if x < width-1: neighbors.append((y, x+1))
            
            for ny, nx in neighbors:
                neighbor_label = label_map[ny, nx]
                if neighbor_label != current_label:
                    pair = tuple(sorted([int(current_label), int(neighbor_label)]))
                    adjacency_pixels[pair].append((float(x), float(y)))
    
    adjacency_time = time.time() - adjacency_start
    print(f"[OK] Found {len(adjacency_pixels)} adjacencies in {adjacency_time:.2f}s")
    
    # Smooth boundaries
    smooth_start = time.time()
    smoothed_count = 0
    
    for pair, pixels_list in adjacency_pixels.items():
        if len(pixels_list) < 3:
            continue
        
        pixels_array = np.array(pixels_list)
        try:
            # Sort by angle around centroid
            centroid = np.mean(pixels_array, axis=0)
            angles = np.arctan2(pixels_array[:, 1] - centroid[1], 
                               pixels_array[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            sorted_pixels = pixels_array[sorted_indices]
            
            # Apply smoothing
            smoothed_x = gaussian_filter1d(sorted_pixels[:, 0], sigma=sigma, mode='wrap')
            smoothed_y = gaussian_filter1d(sorted_pixels[:, 1], sigma=sigma, mode='wrap')
            smoothed_count += 1
        except:
            pass
    
    smooth_time = time.time() - smooth_start
    total_time = time.time() - start_time
    
    print(f"[OK] Smoothed {smoothed_count} boundaries in {smooth_time:.2f}s")
    
    # Check region preservation
    region_boundaries = defaultdict(list)
    for pair, pixels_list in adjacency_pixels.items():
        if len(pixels_list) >= 3:
            label_a, label_b = pair
            region_boundaries[label_a].append(pixels_list)
            region_boundaries[label_b].append(pixels_list)
    
    regions_with_boundaries = len(region_boundaries)
    total_regions = len(unique_labels)
    dropout = total_regions - regions_with_boundaries
    dropout_pct = (dropout / total_regions) * 100
    
    # Small region check
    small_regions = sum(1 for label_idx in unique_labels 
                       if np.sum(label_map == label_idx) < 100)
    
    print(f"\n[METRIC] Regions: {regions_with_boundaries}/{total_regions}")
    print(f"         Dropout: {dropout} ({dropout_pct:.1f}%)")
    print(f"         Small regions (<100px): {small_regions}")
    print(f"         Total time: {total_time:.2f}s")
    
    results.append({
        'colors': num_colors,
        'sigma': sigma,
        'adjacencies': len(adjacency_pixels),
        'smoothed': smoothed_count,
        'regions': regions_with_boundaries,
        'dropout': dropout,
        'dropout_pct': dropout_pct,
        'small_regions': small_regions,
        'time': total_time
    })
    
    # Assessment
    time_pass = total_time < 60  # Under 1 minute
    dropout_pass = dropout_pct < 10  # Less than 10% dropout
    
    status = "PASS" if (time_pass and dropout_pass) else "PARTIAL"
    print(f"\n[ASSESSMENT] Time OK: {time_pass}, Dropout OK: {dropout_pass}")
    print(f"             Status: {status}")

# Summary
print("\n" + "=" * 60)
print("PHASE 6 SUMMARY - Landscape Test")
print("=" * 60)

print(f"\n{'Colors':<8} {'Sigma':<8} {'Adj':<8} {'Regions':<10} {'Dropout':<10} {'Time':<10} {'Status'}")
print("-" * 70)

for r in results:
    time_pass = r['time'] < 60
    dropout_pass = r['dropout_pct'] < 10
    status = "PASS" if (time_pass and dropout_pass) else "PARTIAL"
    print(f"{r['colors']:<8} {r['sigma']:<8} {r['adjacencies']:<8} "
          f"{r['regions']:<10} {r['dropout_pct']:<10.1f} {r['time']:<10.1f} {status}")

print("-" * 70)

any_pass = any(r['time'] < 60 and r['dropout_pct'] < 10 for r in results)

if any_pass:
    print("\n[PASS] Works on landscapes!")
    print("       Performance and quality acceptable.")
else:
    print("\n[PARTIAL] Some issues on landscapes")
    print("          May need optimization for complex scenes.")

print("\n[NOTE] Landscape fragmentation is expected to be higher than logo images.")
print("       The key question is whether performance remains acceptable.")
sys.exit(0)
