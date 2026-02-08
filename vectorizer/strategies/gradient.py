"""Gradient region vectorization strategy."""
import numpy as np
from typing import List, Tuple

from vectorizer.types import (
    Region, VectorRegion, RegionKind, 
    GradientType, ColorStop, Point, BezierCurve
)
from vectorizer.compute_backend import fit_bezier
from vectorizer.region_decomposer import extract_region_boundary
from vectorizer.region_classifier import classify_gradient_type


def vectorize_gradient(region: Region, image: np.ndarray, max_error: float = 2.0) -> VectorRegion:
    """
    Vectorize a gradient region.
    
    Detects and fits linear, radial, or mesh gradients.
    
    Args:
        region: Region to vectorize
        image: Original image
        max_error: Maximum bezier fitting error
        
    Returns:
        VectorRegion with gradient fill
    """
    # Classify gradient type
    gradient_type_str = classify_gradient_type(region, image)
    
    if gradient_type_str == 'radial':
        gradient_type = GradientType.RADIAL
    elif gradient_type_str == 'mesh':
        gradient_type = GradientType.MESH
    else:
        gradient_type = GradientType.LINEAR
    
    # Extract boundary
    boundary = extract_region_boundary(region, image.shape[:2])
    if len(boundary) < 2:
        boundary = _create_bbox_boundary(region)
    
    # Fit bezier curves to boundary
    bezier_curves = fit_bezier(boundary, max_error=max_error)
    if bezier_curves:
        bezier_curves = _close_path(bezier_curves)
    
    # Compute gradient parameters based on type
    if gradient_type == GradientType.LINEAR:
        start, end, stops = _fit_linear_gradient(region, image)
        return VectorRegion(
            kind=RegionKind.GRADIENT,
            path=bezier_curves,
            gradient_type=gradient_type,
            gradient_stops=stops,
            gradient_start=start,
            gradient_end=end
        )
    
    elif gradient_type == GradientType.RADIAL:
        center, radius, stops = _fit_radial_gradient(region, image)
        return VectorRegion(
            kind=RegionKind.GRADIENT,
            path=bezier_curves,
            gradient_type=gradient_type,
            gradient_stops=stops,
            gradient_center=center,
            gradient_radius=radius
        )
    
    else:  # MESH
        triangles, colors = _fit_mesh_gradient(region, image)
        return VectorRegion(
            kind=RegionKind.GRADIENT,
            path=bezier_curves,
            gradient_type=gradient_type,
            mesh_triangles=triangles,
            mesh_colors=colors
        )


def _create_bbox_boundary(region: Region) -> np.ndarray:
    """Create boundary from bounding box as fallback."""
    if region.bbox is None:
        return np.array([])
    
    x, y, w, h = region.bbox
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
        [x, y]
    ])


def _close_path(curves: List[BezierCurve]) -> List[BezierCurve]:
    """Ensure bezier path is closed."""
    if not curves:
        return curves
    
    first_curve = curves[0]
    last_curve = curves[-1]
    
    if (abs(last_curve.p3.x - first_curve.p0.x) < 0.001 and
        abs(last_curve.p3.y - first_curve.p0.y) < 0.001):
        return curves
    
    p0 = last_curve.p3
    p3 = first_curve.p0
    p1 = Point((p0.x + p3.x) / 2, (p0.y + p3.y) / 2)
    p2 = p1
    
    curves.append(BezierCurve(p0, p1, p2, p3))
    return curves


def _fit_linear_gradient(region: Region, image: np.ndarray) -> Tuple[Point, Point, List[ColorStop]]:
    """Fit a linear gradient to the region."""
    # Sample pixels along gradient direction
    coords = np.where(region.mask)
    
    if len(coords[0]) == 0:
        # Fallback to simple 2-color gradient
        center_x, center_y = region.centroid
        return (
            Point(center_x - 50, center_y),
            Point(center_x + 50, center_y),
            [
                ColorStop(0.0, np.array([0.0, 0.0, 0.0])),
                ColorStop(1.0, np.array([1.0, 1.0, 1.0]))
            ]
        )
    
    # Project pixels onto gradient direction
    # For simplicity, use left-to-right gradient based on region bbox
    if region.bbox:
        x, y, w, h = region.bbox
        start = Point(x, y + h/2)
        end = Point(x + w, y + h/2)
    else:
        cx, cy = region.centroid
        start = Point(cx - 50, cy)
        end = Point(cx + 50, cy)
    
    # Compute color stops
    stops = _compute_color_stops(region, image, start, end)
    
    return start, end, stops


def _fit_radial_gradient(region: Region, image: np.ndarray) -> Tuple[Point, float, List[ColorStop]]:
    """Fit a radial gradient to the region."""
    center_x, center_y = region.centroid
    center = Point(center_x, center_y)
    
    # Compute radius (half of max dimension)
    if region.bbox:
        x, y, w, h = region.bbox
        radius = max(w, h) / 2.0
    else:
        coords = np.where(region.mask)
        dx = np.max(coords[1]) - np.min(coords[1])
        dy = np.max(coords[0]) - np.min(coords[0])
        radius = max(dx, dy) / 2.0
    
    # Create radial color stops
    stops = _compute_radial_color_stops(region, image, center, radius)
    
    return center, radius, stops


def _fit_mesh_gradient(region: Region, image: np.ndarray, max_triangles: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a mesh gradient using Delaunay triangulation."""
    from vectorizer.compute_backend import delaunay_triangulation
    
    # Sample points from region
    coords = np.where(region.mask)
    if len(coords[0]) == 0:
        return np.array([]), np.array([])
    
    # Subsample points for mesh
    num_points = min(len(coords[0]), max_triangles)
    indices = np.linspace(0, len(coords[0]) - 1, num_points, dtype=int)
    
    points = np.column_stack([coords[1][indices], coords[0][indices]])
    
    # Add boundary points
    boundary = extract_region_boundary(region, image.shape[:2])
    if len(boundary) > 0:
        # Sample boundary points
        boundary_indices = np.linspace(0, len(boundary) - 1, min(20, len(boundary)), dtype=int)
        boundary_points = boundary[boundary_indices]
        points = np.vstack([points, boundary_points])
    
    # Compute Delaunay triangulation
    triangles = delaunay_triangulation(points)
    
    # Sample colors at triangle vertices
    colors = []
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            colors.append(image[y, x])
        else:
            colors.append(np.array([0.5, 0.5, 0.5]))
    
    return triangles, np.array(colors)


def _compute_color_stops(region: Region, image: np.ndarray, start: Point, end: Point, num_stops: int = 3) -> List[ColorStop]:
    """Compute color stops for a linear gradient."""
    stops = []
    
    dx = end.x - start.x
    dy = end.y - start.y
    length = np.sqrt(dx**2 + dy**2)
    
    if length == 0:
        # Single color
        mean_color = np.mean(image[region.mask], axis=0)
        return [ColorStop(0.0, mean_color), ColorStop(1.0, mean_color)]
    
    # Sample colors along gradient line
    for i in range(num_stops):
        t = i / (num_stops - 1)
        x = start.x + t * dx
        y = start.y + t * dy
        
        # Find nearest pixel in region
        coords = np.where(region.mask)
        if len(coords[0]) > 0:
            distances = np.sqrt((coords[1] - x)**2 + (coords[0] - y)**2)
            nearest_idx = np.argmin(distances)
            color = image[coords[0][nearest_idx], coords[1][nearest_idx]]
        else:
            color = np.array([0.5, 0.5, 0.5])
        
        stops.append(ColorStop(t, color))
    
    # Sort stops by offset
    stops.sort(key=lambda s: s.offset)
    
    return stops


def _compute_radial_color_stops(region: Region, image: np.ndarray, center: Point, radius: float, num_stops: int = 3) -> List[ColorStop]:
    """Compute color stops for a radial gradient."""
    stops = []
    
    for i in range(num_stops):
        t = i / (num_stops - 1)
        r = t * radius
        
        # Sample color at this radius
        if r == 0:
            x, y = int(center.x), int(center.y)
        else:
            x = int(center.x + r)
            y = int(center.y)
        
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0] and region.mask[y, x]:
            color = image[y, x]
        else:
            # Find nearest in-region pixel
            coords = np.where(region.mask)
            if len(coords[0]) > 0:
                distances = np.sqrt((coords[1] - x)**2 + (coords[0] - y)**2)
                nearest_idx = np.argmin(distances)
                color = image[coords[0][nearest_idx], coords[1][nearest_idx]]
            else:
                color = np.array([0.5, 0.5, 0.5])
        
        stops.append(ColorStop(t, color))
    
    stops.sort(key=lambda s: s.offset)
    return stops
