"""Core types for vectorization pipeline."""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto
import numpy as np


class RegionKind(Enum):
    """Classification of region types."""
    FLAT = auto()


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
    path: List[BezierCurve] = field(default_factory=list)  # Outer boundary
    hole_paths: List[List[BezierCurve]] = field(default_factory=list)  # Holes
    fill_color: Optional[np.ndarray] = None
    
    def validate(self) -> bool:
        """Validate region has required fields."""
        return self.fill_color is not None and len(self.path) > 0


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive vectorization pipeline."""
    # Region merging
    merge_threshold_delta_e: float = 5.0
    min_region_size: int = 100
    
    # Strategy parameters
    max_bezier_error: float = 2.0
    
    # Performance
    parallel_workers: int = -1  # -1 = auto
    use_gpu: bool = False
    
    # Output
    precision: int = 2  # Decimal places for SVG coordinates
    simplify_tolerance: float = 0.8
    
    # Display options
    transparent_background: bool = True
    boundary_smoothing_passes: int = 3
    boundary_smoothing_strength: float = 0.6


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
