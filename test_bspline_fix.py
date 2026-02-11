#!/usr/bin/env python3
"""Test B-spline fix on all images in test_images."""
import os
from pathlib import Path
from apexvec.poster_pipeline import PosterPipeline

TEST_DIR = Path("test_images")
OUTPUT_DIR = TEST_DIR / "output_bspline_fix"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    images = [f for f in TEST_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    images.sort()
    
    print(f"Found {len(images)} images to test\n")
    
    for i, img_path in enumerate(images, 1):
        output_svg = OUTPUT_DIR / f"{img_path.stem}.svg"
        output_png = OUTPUT_DIR / f"{img_path.stem}.png"
        
        print(f"[{i}/{len(images)}] Processing: {img_path.name}")
        
        try:
            pipeline = PosterPipeline(num_colors=12, save_stages=False)
            svg_result = pipeline.process(str(img_path), output_path=str(output_svg))
            
            svg_size = len(svg_result)
            print(f"  [OK] Success: SVG={svg_size:,} chars")
            
            if output_svg.exists():
                print(f"  [OK] Saved: {output_svg.name}")
                
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
        
        print()
    
    print("=" * 50)
    print(f"All images processed. Output in: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()
