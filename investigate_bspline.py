"""
Diagnostic script for B-spline failure investigation.

Root Cause Found:
================
The Chaikin corner-cutting algorithm, when applied with iterations >= 2,
creates a discontinuity at the curve boundary:
1. Duplicate points at the wrap-around (last two points become identical)
2. Zero-length segments
3. 180-degree angle changes

This violates splprep's requirements and causes "Invalid inputs" error.

Solution Options:
================
1. Reduce Chaikin iterations to 1 (works but less smoothing)
2. Remove duplicate/zero-length points before spline fitting
3. Properly close the curve after Chaikin
4. Skip B-spline entirely and use Chaikin polygon directly
"""

import numpy as np
from scipy.interpolate import splprep, splev
from apexvec.boundary_smoother import resample_uniform_spacing, chaikin_smooth

def diagnose_bspline_failure():
    """Diagnose why B-spline fails."""
    print("=" * 60)
    print("B-SPLINE FAILURE DIAGNOSIS")
    print("=" * 60)

    # Test input
    small_points = np.array([[10, 10], [20, 20], [30, 10], [40, 20], [50, 10]])

    # Step by step processing
    uniform = resample_uniform_spacing(small_points, spacing=3.0)
    chaikin = chaikin_smooth(uniform, iterations=2)

    print(f"\n1. Input points: {len(small_points)}")
    print(f"2. After uniform resampling: {len(uniform)} points")
    print(f"3. After Chaikin (2 iterations): {len(chaikin)} points")

    # Check for problems
    print(f"\n4. Curve closure check:")
    print(f"   First point: {chaikin[0]}")
    print(f"   Last point: {chaikin[-1]}")
    print(f"   Closed: {np.allclose(chaikin[0], chaikin[-1])}")

    # Check for duplicates
    unique, counts = np.unique(np.round(chaikin, 6), axis=0, return_counts=True)
    duplicates = counts[counts > 1]
    print(f"\n5. Duplicate points: {len(duplicates)} occurrences")
    if len(duplicates) > 0:
        print(f"   Duplicate counts: {duplicates}")

    # Check for zero-length segments
    diffs = np.diff(chaikin, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    zero_segments = np.sum(dists < 0.001)
    print(f"\n6. Zero-length segments: {zero_segments}")

    # Check for sharp angle changes
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    angle_changes = np.diff(angles)
    angle_changes = np.mod(angle_changes + np.pi, 2*np.pi) - np.pi
    sharp_turns = np.sum(np.abs(angle_changes) > 3.0)
    print(f"\n7. Sharp 180-degree turns: {sharp_turns}")

    # Try splprep
    print(f"\n8. Testing splprep:")
    try:
        tck, u = splprep([chaikin[:, 0], chaikin[:, 1]], s=0, per=1, k=3)
        print("   SUCCESS - B-spline fitted")
    except Exception as e:
        print(f"   FAILED: {e}")

    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    print("""
The fix should be in _fit_bspline_with_chaikin() function:

Option A: Reduce Chaikin iterations to 1
    chaikin_points = chaikin_smooth(uniform_points, iterations=1)

Option B: Clean up points before spline fitting
    1. Remove duplicate points
    2. Ensure first == last for proper closure
    3. Remove zero-length segments

Option C: Skip B-spline entirely
    Return Chaikin-smoothed polygon directly as polylines
""")


def test_fixed_bspline():
    """Test B-spline with fix applied."""
    print("\n" + "=" * 60)
    print("TESTING FIXED B-SPLINE")
    print("=" * 60)

    small_points = np.array([[10, 10], [20, 20], [30, 10], [40, 20], [50, 10]])
    uniform = resample_uniform_spacing(small_points, spacing=3.0)

    # FIX: Clean up Chaikin output
    chaikin = chaikin_smooth(uniform, iterations=2)

    # Remove duplicates
    _, unique_idx = np.unique(np.round(chaikin, 6), axis=0, return_index=True)
    chaikin_clean = chaikin[np.sort(unique_idx)]

    # Ensure closed curve
    if not np.allclose(chaikin_clean[0], chaikin_clean[-1]):
        chaikin_clean = np.vstack([chaikin_clean, chaikin_clean[0]])

    print(f"\nOriginal Chaikin: {len(chaikin)} points")
    print(f"After cleanup: {len(chaikin_clean)} points")
    print(f"Closed: {np.allclose(chaikin_clean[0], chaikin_clean[-1])}")

    try:
        tck, u = splprep([chaikin_clean[:, 0], chaikin_clean[:, 1]],
                        s=0, per=1, k=3)
        print("B-spline fitting: SUCCESS")
        return True
    except Exception as e:
        print(f"B-spline fitting: FAILED - {e}")
        return False


if __name__ == "__main__":
    diagnose_bspline_failure()
    test_fixed_bspline()
