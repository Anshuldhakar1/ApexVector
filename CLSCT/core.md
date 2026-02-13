Pipeline Fixes:

Increase Color Quantization

24 colors might not be enough for images with subtle gradients
Try 32-48 colors to capture more tonal variation
Better shading layers = better depth


Reduce/Remove Smoothing

The "smart" and "gaussian" smoothing are destroying the poster aesthetic
Set default to smooth_method="none"
Vector art should have crisp, defined edges


Adjust Epsilon Factor (Simplification)

Current epsilon_factor=0.005 is too aggressive
Reduce to 0.002 or 0.003 to preserve more detail
Keep more points on the contours


Contour Hierarchy Issues

The holes (eyes, paw pads) seem correctly handled
But make sure small features aren't being filtered out by min_contour_area=50
Try reducing to min_contour_area=20 or 25


Remove Dilation (or reduce it)

dilate_iterations=1 is causing colors to bleed into each other
Try dilate_iterations=0 for sharper boundaries
Or only dilate specific problematic layers, not all



What the Output Should Look Like:

Sharp, angular edges like the Brawl Stars logo you showed
Clear color separation with no blending
Defined geometric shapes, not organic blobs
Crisp corners and straight lines where appropriate

Action Plan
Tell the agent to:

Create a new test with these settings:

python   PipelineConfig(
       n_colors=32,              # More colors for gradients
       min_area=30,              # Keep smaller regions
       dilate_iterations=0,      # No dilation
       min_contour_area=20.0,    # Keep small features
       epsilon_factor=0.002,     # Less simplification
       smooth_method="none",     # NO SMOOTHING
       use_bezier=True
   )

Test on img0 and compare side-by-side with the Brawl Stars reference
If edges are still too jagged, try epsilon_factor=0.003 instead of 0.002
Analyze where colors are being merged incorrectly and adjust n_colors up/down

The goal: Crisp, poster-like vector art with solid color fills and sharp geometric shapes - not smooth, organic, rounded blobs.


## Target Aesthetic: Poster-Style Vector Art

The output should resemble:
- Bold, flat color fills (like screen printing/pop art)
- Sharp, geometric edges (not organic/smooth)
- Clear color boundaries with no blending
- Defined shapes with angular features preserved
- Think: Brawl Stars logo, concert posters, vintage travel posters

NOT like:
- Smooth, rounded organic blobs
- Gradient-like color transitions
- Over-simplified shapes that lose character
- Blurred or anti-aliased edges

## Critical Parameters for Poster Look

1. **Color Count**: Start at 32+ for images with shading/gradients
   - Too few colors = lost detail and merged regions
   - More colors = better tonal separation = better depth

2. **No Smoothing by Default**: Sharp edges are the goal
   - Smoothing destroys the vector poster aesthetic
   - Only apply if user explicitly requests it

3. **Minimal Simplification**: epsilon_factor = 0.002-0.003
   - Too high = loss of character and detail
   - Keep more contour points to preserve features

4. **No Dilation**: Sharp color boundaries
   - Dilation causes color bleeding
   - Creates mushy borders between regions
   - Only use if absolutely necessary for gap prevention

5. **Keep Small Features**: min_contour_area = 20-25
   - Eyes, highlights, small details matter
   - Don't over-filter based on size


   ## Pipeline Variants

### Pipeline 1: CLSCT (Original - Color Layer Separation + Contour Tracing)
**Status**: Complete and working
**Purpose**: General-purpose vectorization with smoothing options
**Characteristics**:
- Flexible smoothing (none/smart/gaussian/bspline)
- Conservative defaults (24 colors, some dilation)
- Balances quality and file size
**Config**:
```python
PipelineConfig(
    n_colors=24,
    min_area=50,
    dilate_iterations=1,
    min_contour_area=50.0,
    epsilon_factor=0.005,
    smooth_method="none",  # user configurable
    use_bezier=True
)
```

### Pipeline 2: POSTER (New - Poster-Style Vector Art)
**Status**: To be implemented
**Purpose**: Sharp, geometric poster aesthetic like Brawl Stars logo
**Characteristics**:
- High color count for gradient separation
- NO smoothing (hard-coded)
- NO dilation (crisp boundaries)
- Minimal simplification (preserve detail)
- Sharp, angular edges
**Config**:
```python
PosterPipelineConfig(
    n_colors=32,              # Higher for gradients
    min_area=30,              # Keep smaller regions
    dilate_iterations=0,      # NO dilation - sharp boundaries
    min_contour_area=20.0,    # Keep small features
    epsilon_factor=0.002,     # Minimal simplification
    smooth_method="none",     # FORCED - no user override
    use_bezier=True
)
```

## Pipeline Architecture Pattern

Both pipelines follow the same stages but with different parameters:
1. Load → Quantize → Extract → Clean → Find Contours → Group → Simplify → Smooth → Generate SVG

The difference is in:
- **Parameter defaults** (optimized for different aesthetics)
- **Parameter restrictions** (POSTER locks smoothing to "none")
- **Validation rules** (POSTER warns if colors < 24)

## Implementation Strategy

1. Keep existing `Pipeline` class unchanged (CLSCT mode)
2. Create new `PosterPipeline` class that:
   - Inherits from `Pipeline` OR
   - Uses composition (wraps Pipeline with fixed config) OR
   - Shares modules but uses separate config dataclass
3. CLI `--mode` flag switches between them:
   - `--mode clsct` (default, existing behavior)
   - `--mode poster` (new, locked poster aesthetic)