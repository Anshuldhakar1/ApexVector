# ApexVector - Agent Guidelines

## Project Overview
ApexVector is a Python-based image vectorization system that converts raster images to optimized SVG graphics using adaptive region classification and specialized vectorization strategies.

## Build/Test Commands

### Environment Setup
```bash
# Install dependencies (create requirements.txt first if needed)
pip install numpy opencv-python scikit-image scipy Pillow shapely
pip install pytest  # For testing
```

### Testing
```bash
# Run all tests
pytest tests/ -x

# Run single test file
pytest tests/test_module.py -x

# Run specific test function
pytest tests/test_module.py::test_function -x

# Run with verbose output
pytest tests/ -v -x
```

### Code Quality
```bash
# Lint with flake8 (add to requirements)
flake8 vectorizer/ tests/

# Type checking with mypy (add to requirements)
mypy vectorizer/

# Format with black (add to requirements)
black vectorizer/ tests/
```

### Running the Application
```bash
# Basic vectorization
python -m vectorizer input.png -o output.svg

# Speed mode
python -m vectorizer input.png -o output.svg --speed

# Quality mode
python -m vectorizer input.png -o output.svg --quality
```

## Code Style Guidelines

### Imports
- Group imports: standard library, third-party, local
- Use absolute imports for local modules
- Avoid wildcard imports (`from module import *`)

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Third-party
import numpy as np
import cv2
from skimage.segmentation import slic

# Local
from vectorizer.types import Region, VectorRegion
from vectorizer.compute_backend import ComputeBackend
```

### Type Hints
- Use type hints for all function signatures
- Prefer `typing` module over built-in generic types
- Use dataclasses for simple data containers

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

@dataclass
class Region:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    color: Optional[np.ndarray] = None

def process_regions(regions: List[Region]) -> List[VectorRegion]:
    """Process a list of regions into vector regions."""
    pass
```

### Naming Conventions
- **Variables**: `snake_case` - descriptive but concise
- **Functions**: `snake_case` - verb phrases for actions
- **Classes**: `PascalCase` - nouns for data structures
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore` for internal use

```python
MAX_REGIONS = 1000
DEFAULT_THRESHOLD = 0.05

class VectorRegion:
    def __init__(self, path: List[Tuple[float, float]], fill: str):
        self._path = path
        self.fill = fill
    
    def compute_area(self) -> float:
        """Calculate the area of the vector region."""
        pass
```

### Error Handling
- Use specific exceptions, avoid bare `except:`
- Create custom exception classes for domain errors
- Use context managers for resource management

```python
class VectorizationError(Exception):
    """Base exception for vectorization errors."""
    pass

class InsufficientDataError(VectorizationError):
    """Raised when input data is insufficient for processing."""
    pass

def process_image(image_path: Path) -> str:
    """Process an image and return SVG string."""
    try:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with Image.open(image_path) as img:
            return vectorize_image(img)
            
    except (IOError, OSError) as e:
        raise VectorizationError(f"Failed to process image: {e}")
```

### Documentation
- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring style
- Include type information in docstrings

```python
def segment_image(image: np.ndarray, n_segments: int = 100) -> np.ndarray:
    """Segment an image using SLIC superpixels.
    
    Args:
        image: Input image as RGB array (H, W, 3)
        n_segments: Approximate number of superpixels to generate
        
    Returns:
        Segmentation mask with integer labels
        
    Raises:
        ValueError: If image is not 3-channel RGB
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB with shape (H, W, 3)")
    
    return slic(image, n_segments=n_segments, compactness=10)
```

### Performance Guidelines
- Use NumPy vectorization over Python loops
- Pre-allocate arrays when size is known
- Use `@njit` or `@guvectorize` from Numba for hot paths
- Profile before optimizing

```python
# Good: Vectorized operations
def compute_color_distances(colors: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances from target color."""
    return np.linalg.norm(colors - target, axis=1)

# Avoid: Python loops for large arrays
def slow_distance(colors: List, target: List) -> List[float]:
    distances = []
    for color in colors:
        dist = sum((c - t) ** 2 for c, t in zip(color, target)) ** 0.5
        distances.append(dist)
    return distances
```

### Testing Guidelines
- Write tests for all public functions
- Use parametrized tests for multiple inputs
- Test edge cases and error conditions
- Use fixtures for common test data

```python
import pytest
import numpy as np
from vectorizer.region_decomposer import decompose

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

@pytest.mark.parametrize("n_regions", [10, 50, 100])
def test_decompose_regions(sample_image, n_regions):
    """Test region decomposition with different numbers of regions."""
    regions = decompose(sample_image, n_regions)
    assert len(regions) <= n_regions
    assert all(isinstance(r, Region) for r in regions)
```

## Architecture Notes

### Module Dependencies
Build modules in the order specified in `tasks.md`. Each module should be independently testable.

### Color Space Handling
- Work in linear RGB internally
- Convert to sRGB only for final output
- Use proper gamma correction for display

### Parallel Processing
- Use `ProcessPoolExecutor` for CPU-bound tasks
- Create worker-local `ComputeBackend` instances
- Avoid shared mutable state between workers

### Memory Management
- Release large arrays when no longer needed
- Use memory views for array slicing
- Monitor memory usage for large images

## Validation Requirements

After implementing each module:
1. Run `pytest tests/ -x` to ensure all tests pass
2. Verify no NaN values in outputs
3. Check memory usage stays reasonable
4. Validate SVG output is well-formed

Final validation should process all test images and verify:
- SSIM > 0.75
- Mean ΔE < 15
- SVG size < original image
- Processing time < 10s for 512×512 images