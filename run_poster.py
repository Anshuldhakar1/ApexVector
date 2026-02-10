#!/usr/bin/env python
"""Run poster pipeline on test image."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vectorizer.poster_pipeline import PosterPipeline
from vectorizer.types import AdaptiveConfig

if __name__ == "__main__":
    input_path = "test_images/img0.jpg"
    output_path = "test_images/out/img0_poster_smooth.svg"
    
    print(f"Running poster pipeline on {input_path}")
    print(f"Output: {output_path}")
    print("-" * 50)
    
    config = AdaptiveConfig()
    config.transparent_background = True
    config.boundary_smoothing_passes = 3
    config.boundary_smoothing_strength = 0.6
    config.debug_regions = False
    
    pipeline = PosterPipeline(config=config, num_colors=16)
    svg = pipeline.process(input_path, output_path)
    
    print("-" * 50)
    print(f"Done! Output saved to: {output_path}")
