"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path

# Get the project root (CLSCT directory)
PROJECT_ROOT = Path(__file__).parent.parent
# Get test images directory (relative to project root)
TEST_IMAGES_DIR = PROJECT_ROOT.parent / "test_images"


def get_test_image(name: str) -> Path:
    """Get path to a test image."""
    return TEST_IMAGES_DIR / name


@pytest.fixture
def test_img0():
    """Path to img0.jpg test image."""
    return get_test_image("img0.jpg")


@pytest.fixture
def test_img1():
    """Path to img1.jpg test image."""
    return get_test_image("img1.jpg")


@pytest.fixture
def output_dir():
    """Create and return output directory for test results."""
    out_dir = PROJECT_ROOT / "tests" / "output"
    out_dir.mkdir(exist_ok=True)
    return out_dir
