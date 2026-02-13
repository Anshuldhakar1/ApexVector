This will be a python module "apx-clsct".

## GIT commit

 - Always commit after meaningful changes.
 - Use conventional commits with format ```<type>: <description>```

## Readme 

 - Update with each Git commit. 
 - Should contain the new commands for each mode (pipeline).

## Testing

### Test Configuration
- **Default colors**: Start with 32 (not 24) for gradient-rich images
- Test on multiple image types: flat colors, gradients, complex shading
- No linter or typechecker present

### Test Structure
- Create tests/ folder
- Use actual images from "../test_images/"
- Create separate output folders per image: `output/img0/`, `output/img1/`, etc.
- Generate both SVG and PNG for visual comparison

### Visual Analysis Workflow
1. Convert SVG to PNG using svg_to_png converter
2. Compare output against:
   - Original image (quality check)
   - Reference vector art (aesthetic check - like Brawl Stars logo)
3. Identify specific issues:
   - Over-smoothing (blobs instead of shapes)
   - Lost details (features filtered out)
   - Color merging (not enough quantization)
   - Wrong aesthetic (smooth vs sharp)
4. Propose specific parameter changes
5. Re-test and compare

### Success Criteria
- Sharp, poster-like edges (not smooth)
- Distinct color layers with no blending
- Small features preserved (eyes, highlights)
- Geometric shapes maintain character
- Looks like it could be screen-printed

## Building

 - Use the plan present in core.md to implement new code.
 - For this package "apex-clsct" treat "CLSCT" as root directory of the project.
 - Architecture.md is how the user is expected to use this

 ## Pipeline Development Strategy

### Existing Pipeline (CLSCT)
- **DO NOT MODIFY** the existing working pipeline
- Keep all current tests passing
- Maintain backward compatibility
- Current config and behavior are frozen

### New Pipeline (POSTER)
- Create as separate mode: `--mode poster`
- Can share same modules (quantize, extract, contour, etc.)
- But uses different config with locked parameters
- Has its own test suite in `tests/test_poster_pipeline.py`

### Implementation Approach
1. Create `PosterPipelineConfig` dataclass in `types.py`
2. Create `PosterPipeline` class in `pipeline.py` (or separate `poster_pipeline.py`)
3. Update CLI to support `--mode` parameter with validation
4. Add poster-specific tests comparing output to reference aesthetic
5. Both pipelines co-exist - user chooses via `--mode` flag

### Testing Both Pipelines
- `tests/test_pipeline.py` - Tests CLSCT mode (existing)
- `tests/test_poster_pipeline.py` - Tests POSTER mode (new)
- `tests/test_real_images.py` - Runs BOTH modes on same images for comparison
- Output folders: `output/clsct/` and `output/poster/` for side-by-side analysis

## Iterative Building

  - Run the updated pipeline whenever to create outputs that the user can see as images and svgs both. 
  - Analyze your own output and create a plan to fix it. Ask the user if it is correct.