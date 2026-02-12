"""Common types and exceptions for apx-clsct."""

from typing import List, Tuple, Union
import numpy as np

# Type aliases
ImageArray = np.ndarray
Contour = np.ndarray
Color = Union[Tuple[int, int, int], Tuple[int, int, int, int]]
PathData = str


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
