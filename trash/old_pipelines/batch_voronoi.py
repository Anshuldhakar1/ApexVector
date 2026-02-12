#!/usr/bin/env python3
"""
Batch processing script for Voronoi pipeline
Processes all images in test_images/ and saves to voronoi_output/
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import voronoi_pipeline
sys.path.insert(0, str(Path(__file__).parent))

from voronoi_pipeline import process_image_voronoi, VoronoiConfig


def main():
    # Configuration
    input_dir = Path("test_images")
    output_dir = Path("voronoi_output")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = [
        f
        for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)

    # Configuration for processing
    config = VoronoiConfig(
        n_colors=24,
        text_sigma=0.5,
        general_sigma=1.5,
        min_region_area=50,
        min_text_area=10,
    )

    # Process each image
    results = []
    for i, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 60)

        # Output path preserves the original filename
        output_path = output_dir / f"voronoi_{image_path.stem}.png"

        try:
            metrics = process_image_voronoi(str(image_path), str(output_path), config)
            results.append(
                {"name": image_path.name, "success": True, "metrics": metrics}
            )
            print(f"[OK] Saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Error processing {image_path.name}: {e}")
            results.append({"name": image_path.name, "success": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)

    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\nTotal: {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed images:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

    print(f"\nAll outputs saved to: {output_dir.absolute()}")

    # Print summary table of metrics
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"{'Image':<25} {'Regions':<10} {'Boundaries':<12} {'SVG Size':<12}")
    print("-" * 60)

    for r in results:
        if r["success"]:
            m = r["metrics"]
            print(
                f"{r['name']:<25} {m['n_regions']:<10} {m['n_boundaries']:<12} {m['svg_bytes']:<12,}"
            )


if __name__ == "__main__":
    main()
