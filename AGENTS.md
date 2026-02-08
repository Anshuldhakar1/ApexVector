# ApexVector - Agent Guidelines

## Project Overview
Python image vectorization: raster → optimized SVG via adaptive region classification.

## Commands

```bash
# Setup
pip install -r requirements.txt

# Test
pytest -x                    # All tests
pytest -v -x path/to/test.py # Verbose, single file

# Run
python -m vectorizer input.png -o output.svg
python -m vectorizer input.png -o output.svg --speed    # Fast mode
python -m vectorizer input.png -o output.svg --quality  # Quality mode
```

## Code Style

| Element | Convention | Example |
|---------|-----------|---------|
| Variables | `snake_case` | `region_count` |
| Functions | `snake_case` | `process_image()` |
| Classes | `PascalCase` | `VectorRegion` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_REGIONS` |
| Private | `_leading_underscore` | `_internal_method` |

### Imports (order)
```python
import os                    # stdlib
import numpy as np           # third-party
from vectorizer.types import Region  # local
```

### Type Hints
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Region:
    mask: np.ndarray
    color: Optional[np.ndarray] = None

def process(path: Path) -> str: ...
```

### Error Handling
```python
class VectorizationError(Exception): pass

def process(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    try:
        ...
    except (IOError, OSError) as e:
        raise VectorizationError(f"Failed: {e}")
```

### Performance
- NumPy vectorization > Python loops
- Pre-allocate known-size arrays
- Profile before optimizing

## Architecture

- **Color space**: Linear RGB internally, sRGB for output
- **Parallel**: `ProcessPoolExecutor` with worker-local backends
- **Memory**: Release large arrays, use memory views for slices

## Validation

Per module:
```bash
pytest -x && echo "✓ Tests pass"
```

Final:
| Metric | Threshold |
|--------|-----------|
| SSIM | > 0.75 |
| Mean ΔE | < 15 |
| SVG size | < input |
| Speed (512×512) | < 10s |
