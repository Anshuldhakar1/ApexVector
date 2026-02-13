"""Main pipeline orchestrator for apx-clsct."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

from .types import ImageArray, Contour, Color, VectorizationError
from .quantize import quantize_colors
from .extract import extract_color_layers, clean_mask
from .contour import find_contours, dilate_mask
from .simplify import simplify_contours
from .smooth import smooth_contour_bspline, gaussian_smooth_contour
from .svg import contours_to_svg, save_svg


@dataclass
class PipelineConfig:
    """Configuration for the vectorization pipeline."""

    # Color quantization
    n_colors: int = 24

    # Layer extraction
    min_area: int = 50  # Increased from 10 to filter more noise
    dilate_iterations: int = 0  # Reduced from 1 to prevent fragmentation

    # Contour detection
    contour_method: str = "simple"

    # Contour filtering
    min_contour_area: float = 50.0  # Minimum contour area in pixels

    # Simplification
    epsilon_factor: float = 0.005  # Reduced for better shape preservation

    # Smoothing
    smooth_method: str = "gaussian"  # "gaussian", "bspline", "none"
    smooth_sigma: float = 1.0
    smoothness: float = 3.0

    # SVG output
    use_bezier: bool = True


class Pipeline:
    """Main vectorization pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        self.debug_stages: List[Tuple[str, ImageArray]] = []

    def process(
        self, image_path: str, output_path: Optional[str] = None, debug: bool = False
    ) -> str:
        """Process an image through the vectorization pipeline.

        Args:
            image_path: Path to input image
            output_path: Path to output SVG (optional)
            debug: If True, collect intermediate stage images

        Returns:
            SVG string containing vectorized image

        Raises:
            FileNotFoundError: If input file doesn't exist
            VectorizationError: If processing fails
        """
        # Load image
        image = self._load_image(image_path)

        if debug:
            self.debug_stages = []
            self.debug_stages.append(("1_original", image.copy()))

        # Step 1: Color Quantization
        quantized, palette = self._quantize(image)

        # Calculate dominant color (most frequent) for background
        dominant_color = self._get_dominant_color(quantized)

        if debug:
            self.debug_stages.append(("2_quantized", quantized.copy()))

        # Step 2: Layer Extraction
        layers = self._extract_layers(quantized, palette)

        # Step 3-5: Contour Detection, Simplification, Smoothing
        processed_layers = self._process_contours(layers, debug)

        # Step 6: SVG Generation
        svg = self._generate_svg(
            processed_layers, image.shape[1], image.shape[0], dominant_color
        )

        # Save if output path provided
        if output_path:
            save_svg(svg, output_path)

        return svg

    def _load_image(self, path: str) -> ImageArray:
        """Load image from path."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path)

        # Convert to RGB if needed
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        return np.array(img)

    def _quantize(self, image: ImageArray) -> Tuple[ImageArray, np.ndarray]:
        """Quantize image colors."""
        return quantize_colors(image, self.config.n_colors)

    def _get_dominant_color(self, quantized: ImageArray) -> Color:
        """Get the background color from the quantized image.

        Samples from corner regions where background is most likely
        to be visible, even when the subject fills most of the frame.

        Args:
            quantized: Quantized image

        Returns:
            Background color as RGB tuple
        """
        h, w = quantized.shape[:2]

        # Sample corner regions (where background is most likely visible)
        corner_size = max(5, min(h, w) // 20)  # 5% of image size or at least 5 pixels
        corner_pixels = []

        # Top-left corner
        corner_pixels.extend(quantized[:corner_size, :corner_size].reshape(-1, 3))
        # Top-right corner
        corner_pixels.extend(quantized[:corner_size, -corner_size:].reshape(-1, 3))
        # Bottom-left corner
        corner_pixels.extend(quantized[-corner_size:, :corner_size].reshape(-1, 3))
        # Bottom-right corner
        corner_pixels.extend(quantized[-corner_size:, -corner_size:].reshape(-1, 3))

        corner_pixels = np.array(corner_pixels)

        # Count unique colors in corners
        unique, counts = np.unique(corner_pixels, axis=0, return_counts=True)

        # Get most frequent corner color (background)
        dominant = unique[np.argmax(counts)]

        # Ensure we return a proper RGB tuple
        if len(dominant) >= 3:
            return (int(dominant[0]), int(dominant[1]), int(dominant[2]))
        return (255, 255, 255)  # Default white

    def _extract_layers(
        self, quantized: ImageArray, palette: np.ndarray
    ) -> List[Tuple[Color, np.ndarray]]:
        """Extract binary masks for each color."""
        layers = extract_color_layers(quantized, palette)

        # Clean masks
        cleaned_layers = []
        for color, mask in layers:
            mask = clean_mask(mask, self.config.min_area)
            if self.config.dilate_iterations > 0:
                mask = dilate_mask(mask, self.config.dilate_iterations)
            if np.any(mask):
                cleaned_layers.append((color, mask))

        return cleaned_layers

    def _process_contours(
        self, layers: List[Tuple[Color, np.ndarray]], debug: bool
    ) -> List[Tuple[Color, List[Contour]]]:
        """Process contours for all layers."""
        import cv2

        processed = []
        all_contours_img = None

        if debug:
            # Create blank image for contour visualization
            h, w = layers[0][1].shape[:2] if layers else (100, 100)
            all_contours_img = np.zeros((h, w, 3), dtype=np.uint8)

        for color, mask in layers:
            # Find contours with hierarchy to handle holes
            contours, hierarchy = self._find_contours_hierarchical(mask)

            if len(contours) == 0:
                continue

            # Filter contours by area
            min_area = self.config.min_contour_area
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour.astype(np.float32))
                if area >= min_area:
                    filtered_contours.append(contour)

            if len(filtered_contours) == 0:
                continue

            # Simplify
            filtered_contours = simplify_contours(
                filtered_contours, self.config.epsilon_factor
            )

            # Smooth
            if self.config.smooth_method == "gaussian":
                filtered_contours = [
                    gaussian_smooth_contour(c, self.config.smooth_sigma)
                    for c in filtered_contours
                ]
            elif self.config.smooth_method == "bspline":
                filtered_contours = [
                    smooth_contour_bspline(c, self.config.smoothness)
                    for c in filtered_contours
                ]

            # Filter out empty contours
            filtered_contours = [c for c in filtered_contours if len(c) >= 3]

            if filtered_contours:
                processed.append((color, filtered_contours))

                if debug and all_contours_img is not None:
                    # Draw contours on debug image
                    for contour in filtered_contours:
                        pts = contour.reshape(-1, 1, 2).astype(np.int32)
                        cv2.polylines(
                            all_contours_img,
                            [pts],
                            True,
                            tuple(int(c) for c in color[:3]),
                            2,
                        )

        if debug and all_contours_img is not None:
            self.debug_stages.append(("3_contours", all_contours_img))

        return processed

    def _find_contours_hierarchical(self, mask: np.ndarray):
        """Find contours with hierarchy information.

        Args:
            mask: Binary mask

        Returns:
            Tuple of (contours, hierarchy)
        """
        import cv2

        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8)) * 255

        contours, hierarchy = cv2.findContours(
            mask,
            cv2.RETR_TREE,  # Retrieve all contours with hierarchy for holes
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Reshape contours
        result = []
        for contour in contours:
            if len(contour) >= 3:
                result.append(contour.reshape(-1, 2))

        return result, hierarchy

    def _generate_svg(
        self,
        layers: List[Tuple[Color, List[Contour]]],
        width: int,
        height: int,
        background_color: Color,
    ) -> str:
        """Generate SVG from processed layers."""
        return contours_to_svg(
            layers, width, height, self.config.use_bezier, background_color
        )


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    n_colors: int = 8,
    debug: bool = False,
    **kwargs,
) -> str:
    """Convenience function to process an image.

    Args:
        image_path: Path to input image
        output_path: Path to output SVG (optional)
        n_colors: Number of colors for quantization
        debug: If True, collect debug information
        **kwargs: Additional configuration options

    Returns:
        SVG string
    """
    config = PipelineConfig(n_colors=n_colors, **kwargs)
    pipeline = Pipeline(config)
    return pipeline.process(image_path, output_path, debug)
