
Major Changes:
I want the module name to be ApexVector, so it changes like this,
old: "python -m vectorizer input.png -o output.svg --poster --colors 8"
new: "python -m apexvec input.png -o output.svg --poster --colors 8"

(optional attempt after all other tasks)i would like if somehow we could get the commandline to directly accept it like this, 
new: "apexvec input.png -o output.svg --poster --colors 8"

---

# ApexVector — Contrast-Aware Plan (Revised)

## Core Insight

Instead of creating multiple images and superimposing (complex blending), use **multi-scale quantization** — quantize at different color granularities and merge intelligently.

## Pipeline

```
Input → EXIF fix → Multi-Scale Quantization (3 levels) → 
Hierarchical Region Merge → Boundary Smoothing → SVG
```

## Multi-Scale Quantization

| Scale | Colors | Purpose |
|-------|--------|---------|
| Coarse | 4-6 colors | Large regions, sky, ground |
| Medium | 8-12 colors | Mid-size features, mountains |
| Fine | 16-24 colors | Small details, trees, textures |

## Merge Strategy

1. **Start with coarse regions** as base
2. **Add medium regions** only where they differ significantly (ΔE > 10)
3. **Add fine regions** only for small high-contrast areas
4. **Prefer larger regions** — reject fine segmentation if it doesn't improve perceptual loss

## Implementation

```python
def multi_scale_quantize(image, scales=[6, 10, 18]):
    """Generate hierarchical label maps."""
    label_maps = []
    for n_colors in scales:
        labels, palette = kmeans_quantize(image, n_colors)
        label_maps.append((labels, palette))
    return label_maps

def hierarchical_merge(label_maps, image):
    """Merge from coarse to fine, keeping large regions."""
    coarse_labels, coarse_palette = label_maps[0]
    regions = extract_regions(coarse_labels, coarse_palette)
    
    for fine_labels, fine_palette in label_maps[1:]:
        fine_regions = extract_regions(fine_labels, fine_palette)
        for fr in fine_regions:
            # Only add if small and high contrast with parent
            if fr.area < 0.02 and contrast_with_parent(fr, image) > threshold:
                regions.append(fr)
    
    return regions
```

## Modules (8 total)

| # | Module | Function |
|---|--------|----------|
| 1 | `types.py` | Dataclasses |
| 2 | `raster_ingest.py` | EXIF fix |
| 3 | `multi_scale_quantizer.py` | K-means at 3 scales |
| 4 | `hierarchical_merger.py` | Merge coarse→fine |
| 5 | `boundary_smoother.py` | Spline smoothing |
| 6 | `svg_export.py` | Solid fills, compact |
| 7 | `pipeline.py` | Orchestrate |
| 8 | `cli.py` | `apexvector input.png -o out.svg --scales 6,10,18` |


Create a new readme with better showcase, and list all the commands in cli available