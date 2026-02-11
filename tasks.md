## Executable Prototype Plan (repo: `apexvector`, Python env already set)

Purpose: validate (or falsify) the **shared-boundary + Gaussian smoothing** idea *before* committing to a full pipeline rewrite. This plan is designed to be handed to an agentic coding IDE and is intentionally test-driven and self-correcting.

You will place the test image at repo root as: `./img0.jpg`.

---

# 0) Operating Rules (How we’ll “know” it works)

The prototype is considered successful only if it satisfies these measurable invariants on `img0.jpg`:

1. **No gaps invariant**: rasterized output has ~zero unintended transparent pixels inside the image bounds (allow tiny tolerance like <0.1%).
2. **No region dropout**: dark/near-black regions are still present in the final SVG fills (i.e., not silently omitted).
3. **Flat colors invariant**: every pixel belongs to exactly one palette label; SVG uses solid fills only (no gradients, no alpha-driven disappearance).
4. **Shared edge consistency**: for any adjacency \(A,B\), the boundary geometry used by region A is identical to region B (reversed), not independently smoothed.

If any invariant fails, the plan forces an explicit “why” and a next corrective experiment.

---

# 1) Git Workflow (repo already has many branches)

## 1.1 Create a unique spike branch (no assumptions about existing branch names)
- Use a prefix that won’t collide:
  - `spike/validate-shared-boundaries-<yourname>-<yyyymmdd>`
- Do **not** reuse old spike branches; create a fresh one from your current integration base (usually `main` or your team’s default branch).

## 1.2 “Git a lot” policy
- Every prototype phase = at least one commit.
- Every failed test run = commit containing:
  - what you ran
  - what failed (numbers)
  - what you changed next
- No large “mega commit”.

## 1.3 Output artifacts tracked
- Put generated artifacts under `validation_spike/artifacts/`
- Commit only small artifacts (plots, small PNGs). If your repo discourages binary commits, log paths and keep artifacts untracked but referenced in results (follow repo norms).

---

# 2) Add a Spike Workspace (isolated, minimal impact)

Create a new directory at repo root:

- `validation_spike/`
  - `README.md` (what this spike is)
  - `RESULTS.md` (append-only test log)
  - `STATUS.txt` (current phase + PASS/FAIL)
  - `artifacts/` (renders, plots)
  - `tests/` (executable prototype scripts)

Rule: The spike should **not** require editing the main pipeline code. If small hooks are needed, they must be clearly marked and reversible.

---

# 3) Prototype Phases (self-correcting loop)

## Phase 1 — Sanity: environment + image + baseline pipeline behavior
**Goal:** Ensure the agent can run the existing CLI and capture baseline outputs for later comparison.

Actions:
1. Verify `./img0.jpg` exists and loads.
2. Run the existing CLI poster mode on `img0.jpg` with your “stress” color count (e.g., 24).
3. Save baseline SVG + any stage dumps into:
   - `validation_spike/artifacts/baseline/`

Record in `RESULTS.md`:
- command used
- total runtime (rough)
- whether background is transparent or white
- whether dark regions disappear
- region count if available in logs

**Pass/Fail meaning here:** This phase doesn’t “pass” the concept; it establishes baseline symptoms and gives a reference.

Commit: `spike: baseline run recorded`

Self-correct trigger:
- If CLI doesn’t run, stop and fix environment invocation only (not pipeline logic).

---

## Phase 2 — Minimal “No Gaps” Proof on Synthetic Data
**Goal:** Prove Gaussian smoothing on a shared edge can be used by both sides with **zero gap and zero overlap** *in a controlled setting*.

Actions:
1. Create a synthetic label map with two regions sharing a long boundary (vertical split).
2. Extract that boundary once.
3. Apply Gaussian smoothing **once** to that boundary geometry.
4. Reconstruct two region fills from the same boundary (reversed direction for the other region).
5. Rasterize both and measure:
   - gap pixels (neither region)
   - overlap pixels (both regions)

Record in `RESULTS.md`:
- sigma tested (at least 2 values)
- gap count and overlap count

**Pass Criteria:**
- gap = 0 and overlap = 0 for at least one sigma in your intended range.

Self-correct:
- If gap/overlap occurs, the smoothing method or reconstruction is flawed:
  - adjust endpoint handling, closed/open assumptions
  - re-run
- If you can’t achieve zero gap/overlap, **abort shared boundary approach** (for Gaussian smoothing) and document.

Commit: `spike: phase2 synthetic gap invariant PASS/FAIL`

---

## Phase 3 — Real Image Topology Check (label map must be a valid partition)
**Goal:** Ensure the quantized label map is a clean planar partition suitable for shared-boundary extraction.

Actions:
1. Quantize `img0.jpg` to 12 colors (and optionally 24 to stress).
2. Compute:
   - orphan pixels / speckle count (tiny islands)
   - connected-component fragmentation per label
   - boundary pixel sanity: boundary pixels should separate 2 labels (except borders)

Record:
- number of connected components per label (min/median/max)
- orphan pixel count
- whether topology is “stable enough” for edge extraction

**Pass Criteria (pragmatic):**
- orphan pixels negligible (<0.01% of pixels)
- fragmentation not extreme (if extreme, shared-edge tracing becomes messy and slow)

Self-correct:
- If fragmentation too high:
  - reduce colors or add spatial regularization to quantization (prototype-level adjustment only)
  - or decide landscapes/complex images need a different segmentation method than K-means labels

Commit: `spike: phase3 topology metrics recorded`

---

## Phase 4 — Shared Boundary Extraction Coverage (real image)
**Goal:** Prove you can extract a shared boundary set that covers the label map boundaries.

Actions:
1. From label map, build adjacency information (pairs of labels that touch).
2. Extract boundary geometry per adjacency pair (or globally then attribute segments).
3. Measure **coverage**: compare “extracted shared edges rasterized” vs `find_boundaries()` raster.
4. Record:
   - % boundary coverage
   - adjacency count (# unique label pairs touching)
   - number of boundary segments
   - failure cases: edges that cannot be attributed cleanly

**Pass Criteria:**
- boundary coverage > 95% (tunable) on `img0.jpg`.

Self-correct:
- If coverage low:
  - change extraction strategy (e.g., marching squares vs contour tracing)
  - or refine attribution method (segment-level rather than contour-level attribution)
- If still low after a couple of attempts, shared boundaries may be too complex with current segmentation.

Commit: `spike: phase4 shared boundary coverage PASS/FAIL`

---

## Phase 5 — Apply Gaussian smoothing to shared edges and reconstruct regions
**Goal:** Demonstrate end-to-end: shared-edge smoothing + region reconstruction yields an SVG with **no gaps** and **no dropped dark regions**.

Actions:
1. Smooth each shared edge geometry with Gaussian (test sigma set: e.g., 0.8, 1.2, 1.8).
2. Reconstruct each region’s closed path(s) from the smoothed shared edges.
3. Export SVG with **flat fills only** (no gradients).
4. Rasterize the SVG to PNG (use any available tool in your environment; record which).
5. Compute:
   - transparent pixels inside bounds (gap metric)
   - count of unique colors used in SVG vs palette (dropout metric)
   - “dark color presence”: at least one fill in low brightness range, and those regions appear (by pixel coverage in raster)

Record:
- metrics per sigma
- representative renders saved under `validation_spike/artifacts/phase5/`

**Pass Criteria:**
- gaps < 0.1% (ideally ~0)
- palette colors present in SVG consistent with label coverage (no systematic loss of dark colors)
- visual inspection: boundaries smoother than baseline and not “melted”

Self-correct:
- If gaps exist:
  - it means region reconstruction from edges is incomplete, cycle finding failed, or edge attribution inconsistency.
  - debug by visualizing missing pixels overlay.
- If dark colors drop:
  - locate where drop occurs: extraction, smoothing, cycle reconstruction, or SVG export skipping empty/invalid paths.
  - enforce “never skip region, use fallback polygon” at prototype level and re-test.

Commit: `spike: phase5 end-to-end PASS/FAIL with metrics`

---

## Phase 6 — Landscape question (no central focus) as an explicit experiment
**Goal:** Answer your earlier concern: shared-boundary smoothing should work regardless of “central object”, but segmentation may be the limiting factor.

Actions:
1. Add a second test image (a landscape) only if available. If not, simulate by using any non-logo photo already in repo test set.
2. Repeat Phase 3–5 quickly at lower colors (e.g., 12) and moderate (e.g., 20).
3. Record:
   - fragmentation metrics explode?
   - run time increases sharply?
   - gaps appear?

Decision:
- If it works but is slow: you’ll need performance strategies (edge caching, simpler segmentation).
- If it fails due to fragmentation: landscapes need a different segmentation method (SLIC/superpixels + merging) before shared boundaries.

Commit: `spike: phase6 landscape applicability notes`

---

# 4) Deliverable: Decision Document (what to do next)

At the end, update `validation_spike/RESULTS.md` with:

- A table of phases with PASS/FAIL
- The best sigma value(s) found
- A short “Root causes if failed”
- A recommendation:
  1) proceed to full implementation with shared edges, or
  2) fallback to overlap/dilation method, or
  3) change segmentation method for landscapes

