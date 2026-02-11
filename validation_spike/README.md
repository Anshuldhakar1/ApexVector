# Validation Spike: Shared Boundaries + Gaussian Smoothing

**Purpose**: Validate whether shared-boundary extraction with Gaussian smoothing can produce gap-free, flat-color SVGs before committing to a full pipeline rewrite.

**Test Image**: `./img0.jpg` (copied from `test_images/img0.jpg`)

**Invariants to Validate**:
1. No gaps: ~0% unintended transparent pixels
2. No region dropout: dark regions present
3. Flat colors only: solid fills, no gradients
4. Shared edge consistency: boundaries shared identically between adjacent regions
