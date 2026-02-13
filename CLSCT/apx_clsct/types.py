"""Common types and exceptions for apx-clsct."""

from dataclasses import dataclass, field
from typing import List, Tuple, Union
import numpy as np

# Type aliases
ImageArray = np.ndarray
Contour = np.ndarray
Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
PathData = str


@dataclass
class PipelineConfig:
    """Configuration for the CLSCT vectorization pipeline (general-purpose)."""

    # Color quantization
    n_colors: int = 24

    # Layer extraction
    min_area: int = 50
    dilate_iterations: int = 1

    # Contour detection
    contour_method: str = "simple"

    # Contour filtering
    min_contour_area: float = 50.0

    # Simplification
    epsilon_factor: float = 0.005

    # Smoothing
    smooth_method: str = "none"
    smooth_sigma: float = 1.0
    smoothness: float = 3.0

    # SVG output
    use_bezier: bool = True


@dataclass
class PosterPipelineConfig:
    """Configuration for the POSTER vectorization pipeline (sharp, geometric aesthetic).

    This pipeline is optimized for poster-style vector art with:
    - High color count for gradient separation
    - NO smoothing (hard-coded to "none")
    - NO dilation (crisp boundaries)
    - Minimal simplification (preserve detail)
    """

    # Color quantization - higher default for gradient separation
    n_colors: int = 32

    # Layer extraction - keep smaller regions
    min_area: int = 30
    dilate_iterations: int = 0  # NO dilation for sharp boundaries

    # Contour detection
    contour_method: str = "simple"

    # Contour filtering - keep small features
    min_contour_area: float = 20.0

    # Simplification - minimal to preserve detail
    epsilon_factor: float = 0.002

    # Smoothing - FORCED to "none" for poster aesthetic
    smooth_method: str = field(default="none", init=False)
    smooth_sigma: float = field(default=1.0, init=False)
    smoothness: float = field(default=3.0, init=False)

    # SVG output
    use_bezier: bool = True

    def __post_init__(self):
        """Validate poster-specific constraints."""
        if self.n_colors < 24:
            import warnings

            warnings.warn(
                f"Poster mode works best with n_colors >= 24, got {self.n_colors}. "
                "Consider increasing for better gradient separation."
            )
        if self.epsilon_factor > 0.003:
            import warnings

            warnings.warn(
                f"Poster mode recommends epsilon_factor <= 0.003, got {self.epsilon_factor}. "
                "Higher values may lose detail."
            )


class VectorizationError(Exception):
    """Base exception for vectorization errors."""

    pass


class QuantizationError(VectorizationError):
    """Exception raised during color quantization."""

    pass


class ContourError(VectorizationError):
    """Exception raised during contour detection."""

    pass


class SVGError(VectorizationError):
    """Exception raised during SVG generation."""

    pass
