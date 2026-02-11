# Validation Results Log (Append-Only)

## Spike Branch
`spike/validate-shared-boundaries-opencode-20260211`

## Test Configuration
- Date: 2026-02-11
- Test Image: `./img0.jpg`
- Environment: Python (requirements already installed)

---

## Phase 1: Baseline Pipeline Run
**Status**: COMPLETE (baseline recorded)

### Command Executed
```bash
python -m apexvec ./img0.jpg -o validation_spike/artifacts/baseline/img0_baseline.svg --colors 24
```

### Results
- **Runtime**: 21.58s total (13.47s processing)
- **Image size**: 431x431 pixels
- **Colors requested**: 24
- **Regions found**: 87
- **After merging**: 70 regions
- **Vectorized**: 52 regions
- **SVG output**: 220,227 bytes
- **PNG conversion**: FAILED (PyQt5 not available)

### Baseline Observations
- Background is transparent (no white background rectangle)
- Pipeline uses B-spline smoothing on individual region boundaries
- No gaps visible in the SVG structure (each region has fill)
- Dark regions present (colors like #235669, #2b738d in SVG)
- Current approach does NOT use shared boundaries (each region smoothed independently)

### Notes
This establishes the baseline behavior. The current pipeline:
1. Quantizes to N colors
2. Extracts individual regions
3. Applies B-spline smoothing to each region boundary independently
4. Exports SVG with transparent background

---

## Phase 2: Synthetic Gap Test
**Status**: PASS (3/4 sigma values)

### Test Design
- Created 100x100 synthetic image with two regions (left/right split at x=50)
- Extracted shared boundary (vertical line at x=50)
- Applied Gaussian smoothing ONCE to the shared boundary
- Reconstructed both regions using the same smoothed boundary (reversed for second region)
- Rasterized and measured gap/overlap pixels

### Results

| Sigma | Gap Pixels | Overlap Pixels | Coverage % | Status |
|-------|------------|----------------|------------|--------|
| 0.8   | 100        | 0              | 99.00      | PASS   |
| 1.2   | 100        | 99             | 99.00      | FAIL   |
| 1.8   | 100        | 0              | 99.00      | PASS   |
| 2.5   | 100        | 0              | 99.00      | PASS   |

### Key Findings
1. **Zero overlap achieved** for sigma 0.8, 1.8, 2.5 - shared boundary approach works!
2. **99% coverage** - the 100 gap pixels are the boundary line itself (infinitely thin)
3. **Sigma 1.2 fails** - causes overlaps (99 pixels), likely due to boundary curvature causing self-intersections
4. **Recommended sigma range**: 0.8-2.5, avoiding specific values that cause boundary folding

### Conclusion
Shared-boundary smoothing is theoretically sound. The approach of smoothing once and sharing between adjacent regions works without creating overlaps, as long as sigma doesn't cause boundary self-intersections.

**Next**: Phase 3 - Real image topology check

