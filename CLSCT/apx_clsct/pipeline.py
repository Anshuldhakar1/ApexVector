"""Main pipeline orchestrator for apx-clsct."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

from .types import ImageArray, Contour, Color, VectorizationError, PosterPipelineConfig
from .quantize import quantize_colors
from .extract import extract_color_layers, clean_mask
from .contour import find_contours, dilate_mask
from .simplify import simplify_contours
from .smooth import (
    smooth_contour_bspline,
    gaussian_smooth_contour,
    smart_smooth_contour,
)
from .svg import contours_to_svg, save_svg


@dataclass
class PipelineConfig:
    """Configuration for the vectorization pipeline."""

    # Color quantization
    n_colors: int = 24

    # Layer extraction
    min_area: int = 50  # Increased from 10 to filter more noise
    dilate_iterations: int = 1  # Slight dilation to prevent gaps

    # Contour detection
    contour_method: str = "simple"

    # Contour filtering
    min_contour_area: float = 50.0  # Minimum contour area in pixels

    # Simplification
    epsilon_factor: float = 0.005  # Reduced for better shape preservation

    # Smoothing
    smooth_method: str = "none"  # "gaussian", "bspline", "none", "smart"
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
        self.debug_stages: List[Tuple[str, np.ndarray]] = []

    def process(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        debug: bool = False,
    ) -> str:
        """Process an image through the vectorization pipeline.

        Args:
            image_path: Path to input image
            output_path: Optional path to save SVG output
            debug: If True, save intermediate stage images

        Returns:
            SVG string

        Raises:
            FileNotFoundError: If input file doesn't exist
            VectorizationError: If processing fails
        """
        try:
            # Load image
            image = self._load_image(image_path)
            height, width = image.shape[:2]

            # Clear debug stages
            self.debug_stages = []

            if debug:
                self.debug_stages.append(("1_original", image))

            # Step 1: Color Quantization
            quantized, palette = self._quantize_colors(image)

            if debug:
                self.debug_stages.append(("2_quantized", quantized))

            # Step 2: Extract Color Layers
            layers = self._extract_layers(quantized, palette)

            # Step 3: Process Contours with hole handling
            contour_layers = self._process_contours_with_holes(layers, debug)

            # Step 4: Detect and remove background color
            background_color = self._detect_background_color(quantized)
            foreground_layers = self._filter_background(
                contour_layers, background_color
            )

            # Step 5: Generate SVG
            svg = contours_to_svg(
                foreground_layers,
                width,
                height,
                smooth=self.config.use_bezier,
                background_color=background_color,
            )

            # Save if output path provided
            if output_path:
                save_svg(svg, output_path)

            return svg

        except FileNotFoundError:
            raise
        except Exception as e:
            raise VectorizationError(f"Pipeline processing failed: {e}") from e

    def _load_image(self, image_path: str) -> ImageArray:
        """Load image from path."""
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    def _quantize_colors(self, image: ImageArray) -> Tuple[ImageArray, np.ndarray]:
        """Quantize image colors."""
        return quantize_colors(image, self.config.n_colors)

    def _detect_background_color(self, quantized: ImageArray) -> Color:
        """Detect background color from image corners."""
        h, w = quantized.shape[:2]
        corner_size = min(h, w) // 10

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

    def _filter_background(
        self, contour_layers: List[Tuple[Color, List[Contour]]], background_color: Color
    ) -> List[Tuple[Color, List[Contour]]]:
        """Filter out the background color from contour layers.

        Args:
            contour_layers: List of (color, contours) tuples
            background_color: RGB color to filter out

        Returns:
            Filtered list without background color
        """
        filtered = []
        for color, contours in contour_layers:
            # Check if this color matches background (with tolerance)
            if len(color) >= 3 and len(background_color) >= 3:
                color_match = (
                    abs(int(color[0]) - int(background_color[0])) < 5
                    and abs(int(color[1]) - int(background_color[1])) < 5
                    and abs(int(color[2]) - int(background_color[2])) < 5
                )
                if color_match:
                    continue  # Skip background color
            filtered.append((color, contours))
        return filtered

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

    def _process_contours_with_holes(
        self, layers: List[Tuple[Color, np.ndarray]], debug: bool
    ) -> List[Tuple[Color, List[Contour]]]:
        """Process contours with proper hole handling using compound paths."""
        import cv2

        processed = []
        all_contours_img = None

        if debug:
            h, w = layers[0][1].shape[:2] if layers else (100, 100)
            all_contours_img = np.zeros((h, w, 3), dtype=np.uint8)

        for color, mask in layers:
            # Find contours with hierarchy
            contours, hierarchy = self._find_contours_with_hierarchy(mask)

            if len(contours) == 0:
                continue

            # Build parent-child relationships
            contour_groups = self._group_contours_by_hierarchy(contours, hierarchy)

            # Process each group (outer contour + its holes)
            final_contours = []
            for outer_idx, hole_indices in contour_groups.items():
                outer_contour = contours[outer_idx]
                hole_contours = [contours[i] for i in hole_indices if i < len(contours)]

                # Filter outer contour by area
                outer_area = cv2.contourArea(outer_contour.astype(np.float32))
                if outer_area < self.config.min_contour_area:
                    continue

                # Simplify outer contour
                outer_contour = self._simplify_and_smooth(outer_contour)

                if len(outer_contour) < 3:
                    continue

                # Process holes
                processed_holes = []
                for hole in hole_contours:
                    hole_area = cv2.contourArea(hole.astype(np.float32))
                    if (
                        hole_area >= self.config.min_contour_area * 0.5
                    ):  # Smaller threshold for holes
                        processed_hole = self._simplify_and_smooth(hole)
                        if len(processed_hole) >= 3:
                            processed_holes.append(processed_hole)

                # Store as tuple: (outer, [holes])
                final_contours.append((outer_contour, processed_holes))

            if final_contours:
                processed.append((color, final_contours))

                if debug and all_contours_img is not None:
                    for outer, holes in final_contours:
                        pts = outer.reshape(-1, 1, 2).astype(np.int32)
                        cv2.polylines(
                            all_contours_img,
                            [pts],
                            True,
                            tuple(int(c) for c in color[:3]),
                            2,
                        )
                        for hole in holes:
                            pts = hole.reshape(-1, 1, 2).astype(np.int32)
                            cv2.polylines(
                                all_contours_img, [pts], True, (255, 255, 255), 1
                            )

        if debug and all_contours_img is not None:
            self.debug_stages.append(("3_contours", all_contours_img))

        return processed

    def _find_contours_with_hierarchy(self, mask: np.ndarray):
        """Find contours with hierarchy information.

        Returns:
            Tuple of (contours, hierarchy)
        """
        import cv2

        if mask.dtype != np.uint8:
            mask = (mask.astype(np.uint8)) * 255

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Reshape contours
        result = []
        for contour in contours:
            if len(contour) >= 3:
                result.append(contour.reshape(-1, 2))

        return result, hierarchy

    def _group_contours_by_hierarchy(self, contours, hierarchy):
        """Group contours by parent-child relationships.

        Returns:
            Dict mapping outer contour index to list of hole indices
        """
        if hierarchy is None or len(contours) == 0:
            return {i: [] for i in range(len(contours))}

        hierarchy = hierarchy[0]  # OpenCV returns list of arrays
        groups = {}

        for i, h in enumerate(hierarchy):
            parent = h[3]  # Parent index

            if parent == -1:
                # This is an outer contour
                if i not in groups:
                    groups[i] = []
            else:
                # This is a hole - find its outermost parent
                outer = parent
                while hierarchy[outer][3] != -1:
                    outer = hierarchy[outer][3]

                if outer not in groups:
                    groups[outer] = []
                groups[outer].append(i)

        return groups

    def _simplify_and_smooth(self, contour: Contour) -> Contour:
        """Apply simplification and optional smart smoothing."""
        # Simplify
        simplified = simplify_contours([contour], self.config.epsilon_factor)[0]

        # Apply smoothing if configured
        if self.config.smooth_method == "gaussian":
            simplified = gaussian_smooth_contour(simplified, self.config.smooth_sigma)
        elif self.config.smooth_method == "bspline":
            simplified = smooth_contour_bspline(simplified, self.config.smoothness)
        elif self.config.smooth_method == "smart":
            simplified = smart_smooth_contour(simplified)

        return simplified


class PosterPipeline(Pipeline):
    """Poster-style vectorization pipeline.

    Optimized for sharp, geometric vector art with crisp edges and
    high color separation. No smoothing is applied to maintain
    the poster aesthetic.
    """

    def __init__(self, config: Optional[PosterPipelineConfig] = None):
        """Initialize poster pipeline with configuration.

        Args:
            config: Poster pipeline configuration. Uses defaults if None.
        """
        self.config = config or PosterPipelineConfig()
        self.debug_stages: List[Tuple[str, np.ndarray]] = []


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Process an image through the vectorization pipeline.

    Convenience function for one-off processing.

    Args:
        image_path: Path to input image (JPG/PNG)
        output_path: Optional path to save SVG output
        config: Optional configuration object

    Returns:
        SVG string containing vectorized image

    Example:
        >>> svg = process_image("input.jpg", "output.svg")
        >>> svg = process_image("input.jpg", config=PipelineConfig(n_colors=12))
    """
    pipeline = Pipeline(config)
    return pipeline.process(image_path, output_path)


def process_image_poster(
    image_path: str,
    output_path: Optional[str] = None,
    config: Optional[PosterPipelineConfig] = None,
) -> str:
    """Process an image through the poster vectorization pipeline.

    Convenience function for one-off poster-style processing.

    Args:
        image_path: Path to input image (JPG/PNG)
        output_path: Optional path to save SVG output
        config: Optional poster configuration object

    Returns:
        SVG string containing vectorized image with poster aesthetic

    Example:
        >>> svg = process_image_poster("input.jpg", "output.svg")
        >>> svg = process_image_poster("input.jpg", config=PosterPipelineConfig(n_colors=48))
    """
    pipeline = PosterPipeline(config)
    return pipeline.process(image_path, output_path)
