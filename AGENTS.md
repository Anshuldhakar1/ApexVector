## agents.md

## Project Overview

ApexVector is a Python image vectorization pipeline that converts raster images (JPG/PNG) to optimized SVG using adaptive region classification and shared boundary extraction.

### Key Components

- **Main Package**: `apexvec/` - Core vectorization pipeline
- **Poster Pipeline**: `corrected.py`, `optimized_v2.py` - Standalone poster-style vectorization
- **CLI**: `python -m apexvec` - Command-line interface

## Build / Lint / Test Commands

### Environment Setup

```bash
# Create virtual environment
uv venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests (when tests exist)
pytest

# Run a single test file
pytest tests/test_pipeline.py

# Run a specific test function
pytest tests/test_pipeline.py::test_quantization

# Run with coverage
pytest --cov=apexvec --cov-report=html

# Run in verbose mode
pytest -v
```

### Linting and Formatting

```bash
# Format code with ruff (if configured)
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Type checking with mypy (optional)
mypy apexvec/
```

### Running the Pipeline

```bash
# Basic vectorization
python -m apexvec test_images/img0.jpg -o output.svg

# Poster-style with debug stages
python corrected.py test_images/img0.jpg -o output.png --colors 20
python optimized_v2.py test_images/img0.jpg -o output.png --colors 20

# With stage visualization
python -m apexvec input.png -o output.svg --poster --save-stages debug/
```

## Code Style Guidelines

### Python Style

- **Formatter**: Use ruff or black (line length: 88-100)
- **Docstrings**: Google-style docstrings with Args/Returns/Raises
- **Type Hints**: Use `typing` module for all function signatures
- **Imports**: Group as: stdlib → third-party → local (alphabetical within groups)

### Import Order Example

```python
"""Module docstring."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.measure import find_contours

from apexvec.types import Region, VectorizationError
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `UnifiedPipeline`, `RegionClassifier`)
- **Functions/Variables**: `snake_case` (e.g., `extract_regions`, `label_map`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_COLORS`, `DEFAULT_SIGMA`)
- **Private**: Prefix with `_` (e.g., `_smooth_boundary`, `_temp_buffer`)
- **Type Variables**: Use descriptive names (e.g., `config` not `c`, `image` not `img`)

### Documentation Standards

```python
def process_image(
    image_path: Union[str, Path],
    config: Optional[AdaptiveConfig] = None
) -> str:
    """Process an image through the vectorization pipeline.
    
    Converts raster image to optimized SVG by extracting regions,
    smoothing boundaries, and generating gap-free vector paths.
    
    Args:
        image_path: Path to input image (JPG/PNG)
        config: Pipeline configuration. Uses defaults if None.
        
    Returns:
        SVG string containing vectorized image
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        VectorizationError: If processing fails
        
    Example:
        >>> svg = process_image("input.jpg", AdaptiveConfig(n_colors=12))
        >>> Path("output.svg").write_text(svg)
    """
    pass
```

### Error Handling

- Use custom exceptions from `apexvec.types.VectorizationError`
- Fail fast with clear error messages
- Avoid catching generic `Exception`; catch specific exceptions
- Log errors before raising when appropriate

```python
from apexvec.types import VectorizationError

def quantize_colors(image: np.ndarray, n_colors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize image colors."""
    if n_colors < 2:
        raise ValueError(f"n_colors must be >= 2, got {n_colors}")
    
    if image.size == 0:
        raise VectorizationError("Cannot quantize empty image")
    
    try:
        # Processing logic
        pass
    except Exception as e:
        raise VectorizationError(f"Color quantization failed: {e}") from e
```

### Numerical Operations

- Use `numpy` for array operations (vectorized preferred over loops)
- Use `scipy.ndimage` for image processing
- Prefer `skimage` for higher-level image operations
- Set `random_state` for reproducibility in stochastic algorithms

## Git Commit Guidelines

### When to Commit

- **ALWAYS**: After each meaningful change or test
- **FREQUENTLY**: Small, focused commits preferred over large batches
- **NEVER**: Commit broken code or failing tests to main branch

### Commit Message Format

Use conventional commits with format:
```
<type>: <description>

[optional body]

[optional footer]
```

### Commit Types

| Type | Use When | Example |
|------|----------|---------|
| `feat` | New feature | `feat: add shared boundary extraction` |
| `fix` | Bug fix | `fix: correct gap in region boundaries` |
| `refactor` | Code restructuring | `refactor: extract contour generation to module` |
| `test` | Adding/updating tests | `test: add quantization unit tests` |
| `docs` | Documentation only | `docs: update README with examples` |
| `style` | Formatting (no code change) | `style: format with ruff` |
| `chore` | Maintenance tasks | `chore: update dependencies` |
| `spike` | Experimental/prototype work | `spike: try watershed gap filling` |

### Commit Templates

**Feature Implementation:**
```
feat: implement shared boundary extraction

- Extract boundaries at region interfaces
- Apply Gaussian smoothing per shared edge
- Generate SVG with overlapping paths

Relates to #42
```

**Bug Fix:**
```
fix: resolve gap artifacts at region boundaries

Gap was caused by insufficient mask dilation.
Increased dilation iterations from 2 to 3.

Fixes #15
```

**Experimental:**
```
spike: test watershed segmentation for gap filling

- Replace Voronoi with watershed on gradient
- 89% gap reduction achieved
- Visual quality preserved
```

### Branch Naming

- Features: `feat/shared-boundary-extraction`
- Bug fixes: `fix/gap-artifacts`
- Spikes/Experiments: `spike/watershed-gapfill`
- Hotfixes: `hotfix/critical-svg-export`

## Important Constraints

### NEVER Touch

- **Default pipeline behavior** in `apexvec/pipeline.py` unless explicitly asked
- **Core quantization logic** without thorough testing
- **SVG export format** (maintains compatibility)

### Testing Requirements

- Test with `test_images/img0.jpg` after any change
- Verify both visual quality AND gap percentage
- Ensure Panel 5 matches saved SVG exactly
- Check color fidelity matches quantized image

## File Organization

```
apexvector/
├── apexvec/              # Main package (DO NOT MODIFY without explicit ask)
├── test_images/          # Test images (read-only)
├── corrected.py          # Standalone poster pipeline
├── optimized_v2.py       # Improved standalone pipeline
├── requirements.txt      # Dependencies
├── trash/               # Obsolete files (don't modify)
└── AGENTS.md            # This file
```

## Dependencies

Core: numpy, opencv-python, scikit-image, scipy, Pillow, shapely, scikit-learn
Testing: pytest, pytest-cov
Optional: cupy-cuda12x, cairosvg, triangle
