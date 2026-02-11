import subprocess
import time
from pathlib import Path
import sys

print("Phase 1: Baseline pipeline execution")
print("=" * 50)

# Verify image exists
img_path = Path("./img0.jpg")
if not img_path.exists():
    print(f"ERROR: {img_path} not found")
    sys.exit(1)

print(f"[OK] Test image found: {img_path} ({img_path.stat().st_size} bytes)")

# Run baseline pipeline
output_svg = Path("validation_spike/artifacts/baseline/img0_baseline.svg")
output_png = output_svg.with_suffix(".png")

cmd = [
    "python", "-m", "apexvec",
    "./img0.jpg",
    "-o", str(output_svg),
    "--colors", "24"
]

print(f"\nRunning: {' '.join(cmd)}")
start_time = time.time()

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120
    )
    runtime = time.time() - start_time
    
    print(f"Runtime: {runtime:.2f}s")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print(f"\nSTDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"\nSTDERR:\n{result.stderr}")
    
    # Check outputs
    if output_svg.exists():
        svg_size = output_svg.stat().st_size
        print(f"\n[OK] SVG output: {output_svg} ({svg_size} bytes)")
    else:
        print(f"\n[FAIL] SVG output not found at {output_svg}")
    
    if output_png.exists():
        png_size = output_png.stat().st_size
        print(f"[OK] PNG output: {output_png} ({png_size} bytes)")
    else:
        print(f"[FAIL] PNG output not found at {output_png}")
    
    print(f"\n{'='*50}")
    print("Phase 1 complete - Baseline recorded")
    
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out after 120s")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
