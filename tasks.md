# ApexVector — Curvature-Preserving Plan

## Core Innovation

Replace standard Bézier fitting with **clothoid-based G² curve synthesis** that explicitly preserves curvature magnitude and variation.

## Key Research Insights

| Problem | Research Solution |
|---------|-----------------|
| Bézier curves destroy curvature | Use **clothoid segments** (Euler spirals) with linear curvature variation |
| Jagged polygonal output | **κ-curves** or **εκ-curves** — curvature extrema at control points |
| Geometric-only optimization | **Fairing** with curvature energy minimization |

## Revised Pipeline

```
Input → EXIF fix → Edge detection (Canny) → 
Tangent/curvature estimation → Clothoid fitting → 
G² Bézier conversion → Differentiable refinement → SVG
```

## Critical Implementation

**Tangent/Curvature Estimation** (`curvature_estimator.py`):
- Compute image gradient ∇I
- Tangent direction: perpendicular to gradient
- Curvature κ: divergence of normalized gradient field

**Clothoid Fitting** (`clothoid_fitter.py`):
- Fit piecewise clothoids to edge points using **Fresnel integrals**
- Match position, tangent, and curvature at joints
- Clothoid: curve where curvature varies linearly with arc length

**G² Bézier Conversion** (`g2_converter.py`):
- Convert clothoids to cubic Bézier with **G² continuity constraints**
- Control points derived from osculating circles at endpoints
- Formula: `Q₁ = P₀ + (L/3)T₀`, `Q₂ = P₁ - (L/3)T₁` with `L` from curvature match

**Curvature-Aware Refinement** (`curvature_optimizer.py`):
- Differentiable rasterization (DiffVG)
- Loss = MSE + λ₁·curvature_error + λ₂·G²_violation
- Curvature error: `∫(κ_rendered - κ_target)²ds`

## Build Order (10 modules)

| # | Module | Key Output |
|---|--------|-----------|
| 1 | `types.py` | `ClothoidSegment`, `G2BezierPath`, `CurvatureField` |
| 2 | `raster_ingest.py` | EXIF-corrected, linear RGB |
| 3 | `curvature_estimator.py` | Tangent field T(x,y), curvature field κ(x,y) |
| 4 | `edge_tracer.py` | Edge chains with (position, tangent, curvature) samples |
| 5 | `clothoid_fitter.py` | `list[ClothoidSegment]` matching edge chains |
| 6 | `g2_converter.py` | `list[G2BezierPath]` with C² continuity |
| 7 | `curvature_optimizer.py` | DiffVG-refined paths minimizing curvature error |
| 8 | `topology_merger.py` | Shared boundary deduplication |
| 9 | `svg_export.py` | Valid SVG with curve commands only |
| 10 | `pipeline.py` | End-to-end orchestration |

## Key Formulas

**Clothoid parametric form:**
```
x(s) = ∫₀ˢ cos(½κ't² + κ₀t + θ₀)dt
y(s) = ∫₀ˢ sin(½κ't² + κ₀t + θ₀)dt
```
where κ' = curvature derivative, κ₀ = start curvature, θ₀ = start angle.

**G² Bézier from clothoid:**
- Sample clothoid at endpoints P₀, P₁
- Compute tangents T₀, T₁ and curvatures κ₀, κ₁
- Bézier control points:
  ```
  Q₀ = P₀
  Q₁ = P₀ + αT₀
  Q₂ = P₁ - βT₁  
  Q₃ = P₁
  ```
  where α, β solve G² continuity: `|Q₁-Q₀|³/|Q₂-Q₁|³ = κ₁/κ₀` (with sign)

## Validation

Run on `./testing/` images. Verify:
- **Visual**: No jagged edges, smooth curves preserved
- **Numerical**: Curvature field SSIM > 0.90 vs target
- **Geometric**: No G² discontinuities (curvature jumps < 5%)

## Run test

run the pipeline on every image in "test_images" folder, the output shpuld be in "test_images/out" folder with the same name. replace any previous output present in it.