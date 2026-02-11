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
