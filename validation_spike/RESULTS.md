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

**Next**: Phase 2 - Test shared-boundary approach on synthetic data

