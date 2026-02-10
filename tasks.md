## Self-Correcting Implementation Plan

### Core Principle

Every change is followed by a test run. The test output is compared against explicit pass/fail criteria before moving to the next step. If a step fails, there's a defined rollback and fix path.

---

### Setup: Baseline Snapshot

```bash
# 1. Save current (broken) output as baseline
python -m apexvec test_images/img0.jpg --poster --colors 24 --save-stages test_images/debug/baseline -o test_images/out/baseline.svg

# 2. Record baseline stats
#    - Number of regions
#    - File size
#    - Which colors appear in SVG (grep hex codes)
grep -oP 'fill="#[0-9a-fA-F]{6}"' test_images/out/baseline.svg | sort | uniq -c | sort -rn > test_images/debug/baseline_colors.txt

# 3. Save a rasterized PNG of the SVG for visual comparison
# (use any tool: inkscape, librsvg, browser screenshot)
```

**Known bugs to fix (in priority order):**
1. Dark colors become transparent (regions dropped)
2. White background appeared (was not there before)
3. Boundaries are jagged

---

### Iteration 1: Find and Fix the Region Dropout

**Goal:** Every palette color that covers >0.05% of pixels must appear in the final SVG.

**Step 1.1 — Add a diagnostic script**

Create `scripts/audit_regions.py`:

```python
"""
Run the pipeline up to region extraction and report what survives.
Usage: python scripts/audit_regions.py test_images/img0.jpg --colors 24
"""
import sys
import numpy as np
from pathlib import Path

from apexvec.raster_ingest import ingest_image
from apexvec.color_quantizer import quantize_colors
from apexvec.region_extractor import extract_regions
from apexvec.types import ApexConfig


def audit(image_path: str, n_colors: int = 24):
    config = ApexConfig()
    config.colors = n_colors

    print("=== INGEST ===")
    image_srgb, image_linear = ingest_image(image_path)
    h, w = image_srgb.shape[:2]
    total = h * w
    print(f"  Size: {w}x{h}, total pixels: {total}")

    print("\n=== QUANTIZE ===")
    label_map, palette = quantize_colors(image_srgb, n_colors)
    print(f"  Palette shape: {palette.shape}, dtype: {palette.dtype}")
    print(f"  Label map unique values: {np.unique(label_map).tolist()}")

    # Audit palette coverage
    print("\n=== PALETTE COVERAGE ===")
    dark_colors = []
    for i in range(len(palette)):
        count = np.sum(label_map == i)
        pct = count / total * 100
        rgb = palette[i]
        brightness = int(rgb.mean())
        marker = " ◄ DARK" if brightness < 60 else ""
        print(
            f"  [{i:2d}] RGB({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d})"
            f"  brightness={brightness:3d}"
            f"  pixels={count:6d} ({pct:5.2f}%){marker}"
        )
        if brightness < 60 and pct > 0.05:
            dark_colors.append(i)

    print(f"\n  Dark colors to track: {dark_colors}")

    print("\n=== EXTRACT REGIONS ===")
    regions = extract_regions(label_map, palette, image_linear, config)
    print(f"  Regions returned: {len(regions)}")

    # Check which palette colors have regions
    region_color_indices = set()
    for r in regions:
        cidx = getattr(r, "_color_idx", None)
        region_color_indices.add(cidx)

    print("\n=== DROPOUT CHECK ===")
    missing = []
    for i in range(len(palette)):
        count = np.sum(label_map == i)
        if count < 10:
            continue
        if i not in region_color_indices:
            print(f"  DROPPED: palette[{i}] RGB{tuple(palette[i])} "
                  f"({count} pixels, no region)")
            missing.append(i)

    # Check for transparent pixels
    all_masks = np.zeros((h, w), dtype=bool)
    for r in regions:
        all_masks |= r.mask
    uncovered = total - np.sum(all_masks)
    print(f"\n  Uncovered pixels: {uncovered} ({uncovered/total*100:.2f}%)")

    if uncovered > 0:
        print("  WARNING: These pixels will be TRANSPARENT in SVG")

    # Check mean_color matches palette
    print("\n=== COLOR INTEGRITY ===")
    mismatches = 0
    for r in regions:
        cidx = getattr(r, "_color_idx", None)
        if cidx is None:
            print(f"  Region {r.label}: missing _color_idx!")
            mismatches += 1
            continue
        expected = palette[cidx].astype(np.float64)
        actual = r.mean_color
        if not np.allclose(expected, actual, atol=1):
            print(
                f"  Region {r.label}: expected RGB{tuple(expected.astype(int))}"
                f" got RGB{tuple(actual.astype(int))}"
            )
            mismatches += 1

    if mismatches == 0:
        print("  All regions match palette colors ✓")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"  Palette colors:    {len(palette)}")
    print(f"  Active colors:     {len(region_color_indices)}")
    print(f"  Dropped colors:    {len(missing)}")
    print(f"  Regions:           {len(regions)}")
    print(f"  Uncovered pixels:  {uncovered}")
    print(f"  Color mismatches:  {mismatches}")

    ok = len(missing) == 0 and uncovered == 0 and mismatches == 0
    print(f"\n  STATUS: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/img0.jpg"
    colors = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    audit(path, colors)
```

**Run:**
```bash
python scripts/audit_regions.py test_images/img0.jpg 24
```

**Pass criteria:**
- 0 dropped colors
- 0 uncovered pixels
- 0 color mismatches

**If it fails → the output tells you exactly which colors are dropped and how many pixels are uncovered. Fix `extract_regions` accordingly and re-run until PASS.**

---

### Iteration 2: Find and Fix the White Background

**Goal:** SVG has no `<rect>` background and no region assigned white unless white is in the palette.

**Step 2.1 — Diagnostic**

```bash
# Check if there's a rect element
grep '<rect' test_images/out/baseline.svg

# Check if white appears as a fill
grep -i 'fill="#fff' test_images/out/baseline.svg
grep -i 'fill="white' test_images/out/baseline.svg
grep -i 'fill="#fefefe' test_images/out/baseline.svg

# Check SVG root for style/background
head -5 test_images/out/baseline.svg
```

**Step 2.2 — Fix locations to check (in order)**

1. `svg_export.py` — search for `<rect`, `background`, `white`, or any hardcoded fill on the root `<svg>` element.
2. `poster_pipeline.py` — search for any background insertion step.
3. Check if the `<svg>` tag has `style="background:white"` or similar.

**Step 2.3 — The fix**

The SVG root should look like this (transparent background):

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
```

NOT like this:

```xml
<svg ... style="background: white">
<!-- or -->
<rect width="100%" height="100%" fill="white"/>
```

Remove any background rect or style that wasn't there before.

**Run:**
```bash
python -m apexvec test_images/img0.jpg --poster --colors 24 -o test_images/out/iter2.svg
```

**Pass criteria:**
- No `<rect>` in SVG unless it corresponds to a real palette color
- No `background` in SVG `<style>`
- Visual comparison: open `iter2.svg` in browser, background should be transparent (checkerboard in image editors)

**If it fails → re-examine what was added between the version that had no background and the current version. `git diff` the svg_export file.**

---

### Iteration 3: Fix the Boundary Smoother

**Goal:** Smooth, non-jagged boundaries. No regions dropped by the smoother.

**Step 3.1 — Add a smoother audit**

Create `scripts/audit_smoother.py`:

```python
"""
Run pipeline through smoothing and count survivors.
Usage: python scripts/audit_smoother.py test_images/img0.jpg --colors 24
"""
import sys
import numpy as np
from pathlib import Path

from apexvec.raster_ingest import ingest_image
from apexvec.color_quantizer import quantize_colors
from apexvec.region_extractor import extract_regions
from apexvec.boundary_smoother import smooth_boundaries  # adjust import
from apexvec.types import ApexConfig


def audit_smoother(image_path: str, n_colors: int = 24):
    config = ApexConfig()
    config.colors = n_colors

    image_srgb, image_linear = ingest_image(image_path)
    label_map, palette = quantize_colors(image_srgb, n_colors)
    regions = extract_regions(label_map, palette, image_linear, config)

    print(f"Regions BEFORE smoothing: {len(regions)}")

    # Track which regions have dark colors
    dark_regions = [
        r.label for r in regions
        if hasattr(r, "mean_color") and np.mean(r.mean_color) < 60
    ]
    print(f"Dark regions: {dark_regions}")

    # Run smoother
    smoothed = smooth_boundaries(regions, config)

    # smoothed might be a list of (region, path) tuples or modified regions
    # Adapt this based on your actual return type:
    if isinstance(smoothed, list) and len(smoothed) > 0:
        if isinstance(smoothed[0], tuple):
            out_count = sum(1 for _, path in smoothed if path is not None)
            dropped = [
                r.label for r, path in smoothed if path is None
            ]
            dropped_dark = [
                r.label for r, path in smoothed
                if path is None and np.mean(r.mean_color) < 60
            ]
        else:
            out_count = len(smoothed)
            dropped = []
            dropped_dark = []
    else:
        out_count = len(smoothed) if smoothed else 0
        dropped = []
        dropped_dark = []

    print(f"Regions AFTER smoothing:  {out_count}")
    print(f"Dropped by smoother:      {len(regions) - out_count}")
    if dropped:
        print(f"Dropped region labels:    {dropped}")
    if dropped_dark:
        print(f"Dropped DARK regions:     {dropped_dark}")

    ok = out_count == len(regions)
    print(f"\nSTATUS: {'PASS ✓' if ok else 'FAIL ✗'}")

    if not ok:
        print("\nACTION: Make smooth_boundaries infallible.")
        print("Every region must produce a path. Add fallbacks.")

    return ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_images/img0.jpg"
    colors = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    audit_smoother(path, colors)
```

**Run:**
```bash
python scripts/audit_smoother.py test_images/img0.jpg 24
```

**Pass criteria:**
- Regions AFTER == Regions BEFORE (zero dropped)

**Step 3.2 — Rewrite `smooth_boundaries` with three-tier fallback**

In `apexvec/boundary_smoother.py`:

```python
import numpy as np
from skimage.measure import find_contours
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional


def chaikin_smooth(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Chaikin corner-cutting subdivision."""
    pts = points.copy()
    for _ in range(iterations):
        n = len(pts)
        if n < 3:
            return pts
        new = np.empty((2 * n, 2), dtype=pts.dtype)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new[2 * i] = 0.75 * p0 + 0.25 * p1
            new[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        pts = new
    return pts


def resample_contour(
    contour: np.ndarray, spacing: float = 3.0
) -> np.ndarray:
    """Resample contour to uniform arc-length spacing."""
    diffs = np.diff(contour, axis=0)
    segment_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cumulative = np.concatenate([[0], np.cumsum(segment_lengths)])
    total_length = cumulative[-1]

    if total_length < 1:
        return contour

    n_points = max(int(total_length / spacing), 10)
    uniform_distances = np.linspace(0, total_length, n_points, endpoint=False)

    resampled = np.zeros((n_points, 2))
    for i, d in enumerate(uniform_distances):
        idx = np.searchsorted(cumulative, d, side="right") - 1
        idx = min(idx, len(contour) - 2)
        seg_len = segment_lengths[idx]
        if seg_len < 1e-9:
            resampled[i] = contour[idx]
        else:
            t = (d - cumulative[idx]) / seg_len
            resampled[i] = (1 - t) * contour[idx] + t * contour[idx + 1]

    return resampled


def contour_to_bezier_path(contour: np.ndarray) -> str:
    """Convert a 2D point array to an SVG cubic Bézier path string."""
    if len(contour) < 3:
        return ""

    # Close the contour
    pts = np.vstack([contour, contour[0]])

    # Build path: M start C control1 control2 end ...
    # Use Catmull-Rom → cubic Bézier conversion
    n = len(contour)
    commands = [f"M{pts[0][1]:.1f},{pts[0][0]:.1f}"]

    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        p_prev = pts[(i - 1) % n]

        # Catmull-Rom to cubic Bézier control points
        cp1 = p0 + (p1 - p_prev) / 6.0
        cp2 = p1 - (p2 - p0) / 6.0

        commands.append(
            f"C{cp1[1]:.1f},{cp1[0]:.1f}"
            f" {cp2[1]:.1f},{cp2[0]:.1f}"
            f" {p1[1]:.1f},{p1[0]:.1f}"
        )

    commands.append("Z")
    return "".join(commands)


def smooth_single_region(
    mask: np.ndarray,
    smoothness: float = 0.5,
) -> Optional[str]:
    """
    Three-tier smoothing for a single region mask.

    Tier 1: sub-pixel contour → resample → Chaikin → spline → Bézier
    Tier 2: sub-pixel contour → resample → Chaikin → Catmull-Rom Bézier
    Tier 3: sub-pixel contour → raw polygon path
    """
    contours = find_contours(mask.astype(np.float64), level=0.5)

    if not contours:
        return None

    # Use the longest contour as the outer boundary
    contours.sort(key=len, reverse=True)
    paths = []

    for contour in contours:
        if len(contour) < 4:
            continue

        path = None

        # ── Tier 1: Full spline smoothing ──
        try:
            resampled = resample_contour(contour, spacing=3.0)
            smoothed = chaikin_smooth(resampled, iterations=2)

            x, y = smoothed[:, 0], smoothed[:, 1]
            tck, u = splprep(
                [x, y],
                s=smoothness * len(smoothed),
                per=True,
                k=3,
            )
            n_eval = max(60, len(smoothed) // 2)
            u_new = np.linspace(0, 1, n_eval, endpoint=False)
            x_new, y_new = splev(u_new, tck)
            spline_pts = np.column_stack([x_new, y_new])

            path = contour_to_bezier_path(spline_pts)
        except Exception:
            pass

        # ── Tier 2: Chaikin + Catmull-Rom (no spline) ──
        if not path:
            try:
                resampled = resample_contour(contour, spacing=4.0)
                smoothed = chaikin_smooth(resampled, iterations=3)
                path = contour_to_bezier_path(smoothed)
            except Exception:
                pass

        # ── Tier 3: Raw polygon ──
        if not path:
            coords = " ".join(
                f"{p[1]:.1f},{p[0]:.1f}" for p in contour
            )
            path = f"M{coords}Z"

        if path:
            paths.append(path)

    if not paths:
        return None

    # First path is outer, rest are holes (evenodd fill handles this)
    return " ".join(paths)
```

**Step 3.3 — Wire into pipeline and test**

```bash
python -m apexvec test_images/img0.jpg --poster --colors 24 --save-stages test_images/debug/iter3 -o test_images/out/iter3.svg
python scripts/audit_smoother.py test_images/img0.jpg 24
```

**Pass criteria:**
- Audit: 0 dropped regions
- Visual: boundaries are smooth curves, not staircases
- Visual: no white background
- Visual: dark colors present

**If boundaries are still jagged →** increase Chaikin iterations to 4, decrease resample spacing to 2.0, re-run.

**If regions are dropped →** check which tier is failing, add logging inside `smooth_single_region`, re-run.

---

### Iteration 4: End-to-End SVG Audit

**Step 4.1 — Create `scripts/audit_svg.py`**

```python
"""
Compare SVG output against palette to find missing colors.
Usage: python scripts/audit_svg.py test_images/out/iter3.svg test_images/img0.jpg --colors 24
"""
import sys
import re
import numpy as np

from apexvec.raster_ingest import ingest_image
from apexvec.color_quantizer import quantize_colors


def audit_svg(svg_path: str, image_path: str, n_colors: int = 24):
    image_srgb, _ = ingest_image(image_path)
    _, palette = quantize_colors(image_srgb, n_colors)

    with open(svg_path) as f:
        svg_text = f.read()

    # Extract all fill colors from SVG
    hex_colors = re.findall(
        r'fill=["\']#([0-9a-fA-F]{6})["\']', svg_text
    )
    svg_colors = set()
    for h in hex_colors:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        svg_colors.add((r, g, b))

    print(f"SVG contains {len(svg_colors)} unique colors")
    print(f"Palette has {len(palette)} colors")

    # Check for background rect
    has_bg_rect = "<rect" in svg_text
    print(f"Has <rect>: {has_bg_rect}")
    if has_bg_rect:
        print("  WARNING: Unexpected background rect!")

    # Check for background style
    has_bg_style = "background" in svg_text.lower()
    print(f"Has background style: {has_bg_style}")
    if has_bg_style:
        print("  WARNING: Unexpected background style!")

    # Count paths
    path_count = svg_text.count("<path")
    print(f"Path count: {path_count}")

    # Check each palette color
    print("\n=== PALETTE → SVG MAPPING ===")
    missing = []
    for i, rgb in enumerate(palette):
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        # Allow ±2 tolerance for rounding
        found = False
        for sr, sg, sb in svg_colors:
            if abs(sr - r) <= 2 and abs(sg - g) <= 2 and abs(sb - b) <= 2:
                found = True
                break

        brightness = (r + g + b) // 3
        status = "✓" if found else "MISSING ✗"
        dark = " (DARK)" if brightness < 60 else ""
        print(
            f"  [{i:2d}] RGB({r:3d},{g:3d},{b:3d})"
            f"  brightness={brightness:3d}{dark}  {status}"
        )
        if not found:
            missing.append(i)

    print(f"\n=== SUMMARY ===")
    print(f"  Missing colors: {len(missing)}")
    print(f"  Background rect: {'YES ✗' if has_bg_rect else 'NO ✓'}")
    print(f"  Background style: {'YES ✗' if has_bg_style else 'NO ✓'}")

    ok = len(missing) == 0 and not has_bg_rect and not has_bg_style
    print(f"\n  STATUS: {'PASS ✓' if ok else 'FAIL ✗'}")

    if missing:
        print(f"\n  ACTION: These palette indices have no SVG path: {missing}")
        print("  Check boundary_smoother and svg_export for silent drops.")

    return ok


if __name__ == "__main__":
    svg_path = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else "test_images/img0.jpg"
    colors = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    audit_svg(svg_path, img_path, colors)
```

**Run:**
```bash
python scripts/audit_svg.py test_images/out/iter3.svg test_images/img0.jpg 24
```

**Pass criteria:**
- 0 missing colors
- No background rect
- No background style
- Path count ≈ region count

---

### Iteration 5: Visual Regression Test

**Step 5.1 — Create `scripts/visual_diff.py`**

```python
"""
Render SVG to PNG and compare pixel coverage against original.
Usage: python scripts/visual_diff.py test_images/img0.jpg test_images/out/iter3.svg
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False


def visual_diff(image_path: str, svg_path: str):
    original = np.array(Image.open(image_path).convert("RGBA"))
    h, w = original.shape[:2]

    if not HAS_CAIRO:
        print("Install cairosvg for pixel-level comparison:")
        print("  pip install cairosvg")
        print("Falling back to file-size check only.")

        svg_size = Path(svg_path).stat().st_size
        img_size = Path(image_path).stat().st_size
        ratio = svg_size / img_size * 100
        print(f"SVG size: {svg_size} bytes ({ratio:.1f}% of original)")
        return

    # Render SVG to PNG
    png_bytes = cairosvg.svg2png(
        url=svg_path, output_width=w, output_height=h
    )

    from io import BytesIO
    rendered = np.array(Image.open(BytesIO(png_bytes)).convert("RGBA"))

    # Check transparent pixels in rendered output
    alpha = rendered[:, :, 3]
    transparent_count = np.sum(alpha < 128)
    transparent_pct = transparent_count / (h * w) * 100

    print(f"Image size: {w}x{h}")
    print(f"Transparent pixels: {transparent_count} ({transparent_pct:.2f}%)")

    if transparent_pct > 1.0:
        print(f"  WARNING: >{transparent_pct:.1f}% transparent — regions are being dropped!")

        # Save a diagnostic image showing transparent areas
        diag = original.copy()
        diag[alpha < 128] = [255, 0, 255, 255]  # magenta for missing
        Image.fromarray(diag).save("test_images/debug/missing_pixels.png")
        print("  Saved: test_images/debug/missing_pixels.png")
    else:
        print("  Coverage looks good ✓")

    # Simple color distance check
    mask = alpha >= 128
    orig_rgb = original[:, :, :3][mask].astype(float)
    rend_rgb = rendered[:, :, :3][mask].astype(float)
    mean_diff = np.mean(np.abs(orig_rgb - rend_rgb))
    print(f"Mean color difference (covered pixels): {mean_diff:.1f}/255")

    ok = transparent_pct < 1.0
    print(f"\nSTATUS: {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


if __name__ == "__main__":
    visual_diff(sys.argv[1], sys.argv[2])
```

---

### Master Test Script

Create `scripts/run_all_audits.sh`:

```bash
#!/bin/bash
set -e

IMG="test_images/img0.jpg"
COLORS=24
OUT="test_images/out/latest.svg"
DEBUG="test_images/debug/latest"

echo "============================================"
echo "  STEP 1: Run pipeline"
echo "============================================"
python -m apexvec "$IMG" --poster --colors "$COLORS" --save-stages "$DEBUG" -o "$OUT"

echo ""
echo "============================================"
echo "  STEP 2: Audit region extraction"
echo "============================================"
python scripts/audit_regions.py "$IMG" "$COLORS"

echo ""
echo "============================================"
echo "  STEP 3: Audit boundary smoother"
echo "============================================"
python scripts/audit_smoother.py "$IMG" "$COLORS"

echo ""
echo "============================================"
echo "  STEP 4: Audit SVG output"
echo "============================================"
python scripts/audit_svg.py "$OUT" "$IMG" "$COLORS"

echo ""
echo "============================================"
echo "  STEP 5: Visual diff"
echo "============================================"
python scripts/visual_diff.py "$IMG" "$OUT"

echo ""
echo "============================================"
echo "  ALL AUDITS COMPLETE"
echo "============================================"
```

---

### The Iteration Loop

```text
┌─────────────────────────────────┐
│  Run: bash scripts/run_all_audits.sh  │
└──────────────┬──────────────────┘
               │
               ▼
       ┌───────────────┐
       │  All 4 PASS?  │
       └───┬───────┬───┘
          YES      NO
           │       │
           ▼       ▼
        DONE    Read which audit failed
                       │
                       ▼
              ┌─────────────────────┐
              │ audit_regions FAIL? │──► Fix region_extractor.py
              │ audit_smoother FAIL?│──► Fix boundary_smoother.py
              │ audit_svg FAIL?     │──► Fix svg_export.py
              │ visual_diff FAIL?   │──► Check all three above
              └─────────┬───────────┘
                        │
                        ▼
                 Make ONE fix
                        │
                        ▼
              Re-run audit script for
              just that stage first
                        │
                        ▼
                 Stage passes?
                  YES      NO
                   │       │
                   ▼       └──► Try different fix, re-run
            Re-run full audit
                   │
                   ▼
              (back to top)
```

**Rules:**
1. Fix ONE thing at a time
2. Run the specific audit for that thing
3. Only run the full suite when the specific audit passes
4. Never move to boundaries until regions are correct
5. Never move to SVG until boundaries are correct

Start with `bash scripts/run_all_audits.sh` and let the output guide you. Want me to implement the boundary smoother fix from Step 3.2 into your actual codebase?