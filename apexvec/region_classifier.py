"""Region classification for routing to appropriate vectorization strategy."""
from typing import List, Tuple
import numpy as np

from apexvec.types import Region, RegionKind, AdaptiveConfig
from apexvec.compute_backend import (
    compute_edge_density,
    rgb_to_lab,
    delta_e_2000
)


def classify(regions: List[Region], image: np.ndarray, config: AdaptiveConfig) -> List[Region]:
    """
    Classify regions by type for strategy routing.

    Uses gradient model fitting to directly model what each region type looks like:
    - GRADIENT: Strong linear gradient fit (R² > 0.98)
    - FLAT: Uniform color (R²_flat > 0.95)
    - EDGE: Dominant edge with consistent direction
    - DETAIL: Complex patterns (default)

    Args:
        regions: List of regions to classify
        image: Original image (linear RGB)
        config: Adaptive configuration

    Returns:
        List of regions with kind attribute set
    """
    classified_regions = []

    for region in regions:
        # Fit gradient models
        gradient_result = _fit_gradient_model(region, image)
        r_squared = gradient_result['r_squared']
        r_squared_flat = gradient_result['r_squared_flat']
        gradient_coeffs = gradient_result['coeffs']

        # Compute edge features
        edge_density = compute_edge_density(region.mask, image)
        gradient_direction_variance = _compute_gradient_direction_variance(region, image)

        # Classification based on model fits
        if r_squared > 0.98 and (r_squared - r_squared_flat) > 0.15:
            # Strong linear gradient fit that's significantly better than flat
            region.kind = RegionKind.GRADIENT
            # Store gradient parameters for later use
            region.gradient_coeffs = gradient_coeffs

        elif r_squared_flat > 0.95:
            # Uniform color - no significant gradient
            region.kind = RegionKind.FLAT

        elif edge_density > 0.3 and gradient_direction_variance < 0.2:
            # Dominant edge with consistent direction
            region.kind = RegionKind.EDGE

        else:
            # Complex pattern - default to detail
            region.kind = RegionKind.DETAIL

        classified_regions.append(region)

    # Post-classification: merge adjacent regions of same kind with compatible parameters
    classified_regions = _merge_similar_regions(classified_regions, image)

    return classified_regions


def _fit_gradient_model(region: Region, image: np.ndarray) -> dict:
    """
    Fit linear gradient model: color = a·x + b·y + c

    Returns dict with:
        - r_squared: R² of linear gradient fit
        - r_squared_flat: R² of flat (mean only) fit
        - coeffs: Gradient coefficients [a, b, c] for each channel
        - improvement: r_squared - r_squared_flat
    """
    pixels = image[region.mask]  # shape (N, 3)
    positions = np.argwhere(region.mask)  # shape (N, 2)

    if len(pixels) < 3:
        # Not enough pixels for meaningful fit
        return {
            'r_squared': 0.0,
            'r_squared_flat': 0.0,
            'coeffs': np.zeros((3, 3)),
            'improvement': 0.0
        }

    N = len(pixels)

    # Build design matrix: [x, y, 1]
    X = np.column_stack([positions[:, 1], positions[:, 0], np.ones(N)])

    # Fit linear gradient model for each channel
    coeffs_list = []
    total_variance = 0.0
    residual_variance = 0.0

    for channel in range(3):
        y = pixels[:, channel]

        # Solve least squares: X @ coeffs = y
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        coeffs_list.append(coeffs)

        # Predict and compute R²
        predicted = X @ coeffs
        channel_residual_var = np.var(y - predicted)
        channel_total_var = np.var(y)

        total_variance += channel_total_var
        residual_variance += channel_residual_var

    coeffs = np.array(coeffs_list)  # shape (3, 3) - coeffs for each RGB channel

    # Overall R² for gradient model
    if total_variance > 1e-10:
        r_squared = 1 - residual_variance / total_variance
    else:
        r_squared = 1.0  # Perfect fit (no variance)

    # Fit flat model (mean only)
    mean_color = pixels.mean(axis=0)
    flat_residuals = pixels - mean_color
    flat_residual_var = np.var(flat_residuals)

    if total_variance > 1e-10:
        r_squared_flat = 1 - flat_residual_var / total_variance
    else:
        r_squared_flat = 1.0

    return {
        'r_squared': float(r_squared),
        'r_squared_flat': float(r_squared_flat),
        'coeffs': coeffs,
        'improvement': float(r_squared - r_squared_flat)
    }


def _compute_gradient_direction_variance(region: Region, image: np.ndarray) -> float:
    """
    Compute variance of gradient directions within a region.

    Low variance means consistent edge direction.
    """
    # Get coordinates of region pixels
    coords = np.where(region.mask)
    if len(coords[0]) == 0:
        return 1.0  # High variance (no data)

    # Sample pixels to compute gradients (don't need all of them)
    step = max(1, len(coords[0]) // 100)
    sample_indices = range(0, len(coords[0]), step)

    directions = []

    for i in sample_indices:
        y, x = coords[0][i], coords[1][i]

        # Skip border pixels
        if y <= 0 or y >= image.shape[0] - 1 or x <= 0 or x >= image.shape[1] - 1:
            continue

        # Compute local gradient using Sobel-like approximation
        dx = image[y, x + 1] - image[y, x - 1]
        dy = image[y + 1, x] - image[y - 1, x]

        # Average across channels for direction
        dx_mean = np.mean(dx)
        dy_mean = np.mean(dy)

        # Compute direction angle
        magnitude = np.sqrt(dx_mean**2 + dy_mean**2)
        if magnitude > 1e-6:
            angle = np.arctan2(dy_mean, dx_mean)
            directions.append(angle)

    if len(directions) < 2:
        return 1.0  # High variance (not enough samples)

    # Compute circular variance
    directions = np.array(directions)
    sin_mean = np.mean(np.sin(directions))
    cos_mean = np.mean(np.cos(directions))

    # Circular variance: 1 - R, where R is the mean resultant length
    # Closer to 0 means more consistent direction
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    circular_variance = 1 - r

    return float(circular_variance)


def _merge_similar_regions(regions: List[Region], image: np.ndarray) -> List[Region]:
    """
    Merge adjacent regions of same kind with compatible parameters.

    - Two GRADIENT regions with similar coefficients
    - Two FLAT regions with Delta E < 3.0
    """
    if len(regions) <= 1:
        return regions

    # Build adjacency graph
    adjacency = _build_adjacency_graph(regions)

    merged = True
    while merged:
        merged = False
        to_remove = set()

        for i, region in enumerate(regions):
            if i in to_remove:
                continue

            for j in adjacency.get(i, []):
                if j in to_remove or j >= len(regions):
                    continue

                neighbor = regions[j]

                # Check if regions can be merged
                if _can_merge_regions(region, neighbor, image):
                    # Merge neighbor into region
                    _merge_two_regions(region, neighbor, image)
                    to_remove.add(j)
                    merged = True
                    break

        # Remove merged regions
        regions = [r for i, r in enumerate(regions) if i not in to_remove]

        # Rebuild adjacency if we merged
        if merged:
            adjacency = _build_adjacency_graph(regions)

    return regions


def _build_adjacency_graph(regions: List[Region]) -> dict:
    """Build adjacency graph from regions."""
    from scipy.ndimage import binary_dilation

    adjacency = {i: set() for i in range(len(regions))}

    # Create a label map
    if len(regions) == 0:
        return adjacency

    # Get image dimensions from first region
    first_mask = regions[0].mask
    height, width = first_mask.shape

    label_map = np.zeros((height, width), dtype=int)
    for region in regions:
        label_map[region.mask] = region.label

    # Find neighbors for each region
    for i, region in enumerate(regions):
        dilated = binary_dilation(region.mask)
        boundary = dilated & ~region.mask

        neighbor_labels = set(label_map[boundary])
        neighbor_labels.discard(0)
        neighbor_labels.discard(region.label)

        # Map labels to indices
        for j, other in enumerate(regions):
            if other.label in neighbor_labels:
                adjacency[i].add(j)
                adjacency[j].add(i)

    return adjacency


def _can_merge_regions(region1: Region, region2: Region, image: np.ndarray) -> bool:
    """Check if two regions can be merged based on their kind and parameters."""
    if region1.kind != region2.kind:
        return False

    if region1.kind == RegionKind.FLAT:
        # Check color similarity
        if region1.mean_color is None or region2.mean_color is None:
            return False

        lab1 = rgb_to_lab(region1.mean_color.reshape(1, 1, 3)).flatten()
        lab2 = rgb_to_lab(region2.mean_color.reshape(1, 1, 3)).flatten()
        delta_e = delta_e_2000(lab1, lab2)

        return delta_e < 3.0

    elif region1.kind == RegionKind.GRADIENT:
        # Check gradient coefficient similarity
        coeffs1 = getattr(region1, 'gradient_coeffs', None)
        coeffs2 = getattr(region2, 'gradient_coeffs', None)

        if coeffs1 is None or coeffs2 is None:
            return False

        # Extract direction vectors (a, b coefficients) for each channel
        # coeffs shape: (3, 3) = [channel, [a, b, c]]
        dir1 = coeffs1[:, :2]  # shape (3, 2)
        dir2 = coeffs2[:, :2]

        # Compute direction for each channel
        for ch in range(3):
            v1 = dir1[ch]
            v2 = dir2[ch]

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 1e-6 and norm2 > 1e-6:
                # Normalize and compute angle
                v1_norm = v1 / norm1
                v2_norm = v2 / norm2

                # Compute angle between directions
                dot_product = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
                angle = np.arccos(abs(dot_product))  # Use abs to handle opposite directions
                angle_deg = np.degrees(angle)

                # Check angle threshold (15 degrees)
                if angle_deg > 15:
                    return False

                # Check magnitude ratio (within 20%)
                ratio = min(norm1, norm2) / max(norm1, norm2) if max(norm1, norm2) > 0 else 1.0
                if ratio < 0.8:
                    return False

        return True

    return False


def _merge_two_regions(region1: Region, region2: Region, image: np.ndarray):
    """Merge region2 into region1."""
    # Combine masks
    region1.mask = region1.mask | region2.mask

    # Recalculate centroid
    coords = np.where(region1.mask)
    if len(coords[0]) > 0:
        region1.centroid = (float(np.mean(coords[1])), float(np.mean(coords[0])))

    # Recalculate bbox
    if len(coords[0]) > 0:
        region1.bbox = (
            int(np.min(coords[1])),
            int(np.min(coords[0])),
            int(np.max(coords[1]) - np.min(coords[1]) + 1),
            int(np.max(coords[0]) - np.min(coords[0]) + 1)
        )

    # Recalculate mean color
    region1.mean_color = np.mean(image[region1.mask], axis=0)

    # Update label (use lower label)
    region1.label = min(region1.label, region2.label)

    # Combine neighbors
    region1.neighbors = list(set(region1.neighbors + region2.neighbors))

    # If gradient regions, recompute gradient coefficients
    if region1.kind == RegionKind.GRADIENT:
        gradient_result = _fit_gradient_model(region1, image)
        region1.gradient_coeffs = gradient_result['coeffs']


def classify_gradient_type(region: Region, image: np.ndarray) -> str:
    """
    Classify gradient type for gradient regions.

    Returns 'linear', 'radial', or 'mesh'.
    """
    gradient_coeffs = getattr(region, 'gradient_coeffs', None)

    if gradient_coeffs is None:
        # Recompute if not stored
        gradient_result = _fit_gradient_model(region, image)
        gradient_coeffs = gradient_result['coeffs']

    if gradient_coeffs is None:
        return 'linear'  # Default fallback

    # Analyze gradient consistency across channels
    directions = []
    magnitudes = []

    for ch in range(3):
        a, b = gradient_coeffs[ch, 0], gradient_coeffs[ch, 1]
        direction = np.arctan2(b, a)
        magnitude = np.sqrt(a**2 + b**2)
        directions.append(direction)
        magnitudes.append(magnitude)

    # Check if all channels have similar direction (linear)
    direction_variance = np.var(directions)
    mean_magnitude = np.mean(magnitudes)

    if direction_variance < 0.1 and mean_magnitude > 0.01:
        # Consistent direction across all channels
        return 'linear'

    # Check for radial pattern
    if _is_radial_gradient(region, image, gradient_coeffs):
        return 'radial'

    # Default to mesh for complex patterns
    return 'mesh'


def _is_radial_gradient(region: Region, image: np.ndarray, gradient_coeffs: np.ndarray) -> bool:
    """Check if gradient appears to be radial (centered)."""
    center_x, center_y = region.centroid

    coords = np.where(region.mask)
    if len(coords[0]) < 10:
        return False

    step = max(1, len(coords[0]) // 50)
    sample_indices = range(0, len(coords[0]), step)

    consistent_count = 0
    total_count = 0

    for i in sample_indices:
        y, x = coords[0][i], coords[1][i]

        # Vector from center to point
        to_center = np.array([center_x - x, center_y - y])
        to_center_norm = np.linalg.norm(to_center)

        if to_center_norm > 0:
            to_center = to_center / to_center_norm

            # Get gradient at this point from coefficients
            # Gradient direction is perpendicular to level sets
            # For linear gradient color = a*x + b*y + c, gradient is (a, b)
            avg_a = np.mean(gradient_coeffs[:, 0])
            avg_b = np.mean(gradient_coeffs[:, 1])
            local_grad = np.array([avg_a, avg_b])

            grad_norm = np.linalg.norm(local_grad)
            if grad_norm > 0:
                local_grad = local_grad / grad_norm

                # Check alignment with radial direction
                dot_product = np.dot(local_grad, to_center)
                if abs(dot_product) > 0.5:
                    consistent_count += 1
                total_count += 1

    if total_count == 0:
        return False

    return (consistent_count / total_count) > 0.6
