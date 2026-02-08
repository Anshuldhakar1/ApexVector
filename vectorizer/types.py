"""Core types for vectorization pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto
import numpy as np


class RegionKind(Enum):
    """Classification of region types for routing to appropriate strategy."""
    FLAT = auto()
    GRADIENT = auto()
    EDGE = auto()
    DETAIL = auto()


class GradientType(Enum):
    """Types of gradients supported."""
    LINEAR = auto()
    RADIAL = auto()
    MESH = auto()


@dataclass
class Point:
    """2D point with float coordinates."""
    x: float
    y: float


@dataclass
class BezierCurve:
    """Cubic bezier curve segment."""
    p0: Point
    p1: Point  # Control point
    p2: Point  # Control point
    p3: Point


@dataclass
class ColorStop:
    """Color stop for gradients."""
    offset: float  # 0.0 to 1.0
    color: np.ndarray  # RGB array


@dataclass
class Region:
    """Input region from segmentation."""
    mask: np.ndarray
    label: int
    neighbors: List[int] = field(default_factory=list)
    centroid: Optional[Tuple[float, float]] = None
    mean_color: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    
    def __post_init__(self):
        if self.centroid is None:
            coords = np.where(self.mask)
            if len(coords[0]) > 0:
                self.centroid = (float(np.mean(coords[1])), float(np.mean(coords[0])))
        
        if self.bbox is None:
            coords = np.where(self.mask)
            if len(coords[0]) > 0:
                self.bbox = (
                    int(np.min(coords[1])),
                    int(np.min(coords[0])),
                    int(np.max(coords[1]) - np.min(coords[1]) + 1),
                    int(np.max(coords[0]) - np.min(coords[0]) + 1)
                )


@dataclass
class VectorRegion:
    """Vectorized output region."""
    kind: RegionKind
    path: List[BezierCurve] = field(default_factory=list)
    fill_color: Optional[np.ndarray] = None
    gradient_type: Optional[GradientType] = None
    gradient_stops: List[ColorStop] = field(default_factory=list)
    gradient_start: Optional[Point] = None
    gradient_end: Optional[Point] = None
    gradient_center: Optional[Point] = None
    gradient_radius: Optional[float] = None
    mesh_triangles: Optional[np.ndarray] = None
    mesh_colors: Optional[np.ndarray] = None
    
    # Validation
    def validate(self) -> bool:
        """Validate region has required fields for its kind."""
        if self.kind == RegionKind.FLAT:
            return self.fill_color is not None and len(self.path) > 0
        elif self.kind == RegionKind.GRADIENT:
            return (self.gradient_type is not None and 
                    len(self.gradient_stops) > 0 and 
                    len(self.path) > 0)
        elif self.kind == RegionKind.EDGE:
            return len(self.path) > 0
        elif self.kind == RegionKind.DETAIL:
            return (self.mesh_triangles is not None and 
                    self.mesh_colors is not None and
                    len(self.path) > 0)
        return False


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive vectorization pipeline."""
    # Segmentation
    slic_segments: int = 400
    slic_compactness: float = 20.0
    slic_sigma: float = 1.0
    
    # Region merging
    merge_threshold_delta_e: float = 5.0
    min_region_size: int = 100
    
    # Classification thresholds
    gradient_threshold: float = 0.3
    edge_density_threshold: float = 0.1
    detail_complexity_threshold: float = 0.5
    
    # Strategy parameters
    max_bezier_error: float = 2.0
    max_mesh_triangles: int = 500
    
    # Performance
    parallel_workers: int = -1  # -1 = auto
    use_gpu: bool = False
    
    # Output
    precision: int = 2  # Decimal places for SVG coordinates
    simplify_tolerance: float = 0.8


@dataclass
class IngestResult:
    """Result from raster image ingestion."""
    image_linear: np.ndarray  # Linear RGB for processing
    image_srgb: np.ndarray    # sRGB for output comparison
    original_path: str
    width: int
    height: int
    has_alpha: bool


class VectorizationError(Exception):
    """Base exception for vectorization errors."""
    pass
