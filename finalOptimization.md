# SVG Optimization Documentation

## Overview
This document describes all optimizations implemented in `svg_optimizer.py`. These optimizations may affect final output quality and should be reviewed if visual issues occur.

---

## 1. Coordinate Optimization

### 1.1 Coordinate Quantization
- **Function**: `quantize_coordinates(bezier_curves, grid_size=0.5)`
- **Description**: Rounds all coordinates to nearest grid point
- **Default**: 0.5 pixel grid
- **Impact**: Reduces file size by making coordinates more repetitive
- **Quality Impact**: Minor - usually not visible at grid sizes ≤ 1.0
- **Used In**: All optimization modes except basic

### 1.2 Relative Coordinates
- **Function**: `_bezier_to_svg_path()` with `use_relative=True`
- **Description**: Uses relative cubic bezier commands (`c`) instead of absolute (`C`)
- **Impact**: Smaller numbers, better compression
- **Quality Impact**: None
- **Default**: Enabled

### 1.3 Adaptive Precision
- **Function**: `optimize_precision_adaptive()`
- **Description**: Adjusts decimal places based on coordinate values
  - Precision 0: For values mostly integers
  - Precision 1: For values mostly half-integers
  - Precision 2+: For values with fractional parts
- **Impact**: Reduces file size significantly
- **Quality Impact**: Minimal when adaptive
- **Default**: 2 decimal places base

---

## 2. Path Simplification

### 2.1 Bezier Curve Simplification
- **Function**: `simplify_bezier_curves(tolerance=0.5)`
- **Description**: Removes redundant control points using RDP-like algorithm
- **Algorithm**: Checks if control points deviate from straight line
- **Parameters**:
  - `tolerance=0.5`: Normal mode
  - `tolerance=1.0`: Aggressive (ultra mode)
  - `tolerance=2.0`: Extreme mode
  - `tolerance=4.0`: Insane mode
  - `tolerance=8.0`: Monochrome mode
- **Quality Impact**: Higher tolerance = more quality loss, blockier curves
- **Merge Behavior**: Merges consecutive curves into single curve

---

## 3. Color Optimization

### 3.1 Color Quantization
- **Function**: `_quantize_color(color, levels=16)`
- **Description**: Reduces color palette to specified levels
- **Levels**:
  - `levels=16`: Normal (256 colors)
  - `levels=8`: Aggressive (64 colors)
  - `levels=4`: Insane (16 colors)
- **Impact**: Reduces file size, creates color banding
- **Quality Impact**: Visible posterization at lower levels

### 3.2 Color Merging
- **Function**: `_merge_similar_colors(tolerance=0.02)`
- **Description**: Merges regions with similar colors (2% tolerance)
- **Impact**: Groups paths by color for `<g>` elements
- **Quality Impact**: Minor color shifts

### 3.3 Hex Shorthand
- **Function**: `_color_to_hex_compact()`
- **Description**: Uses 3-digit hex (#RGB) when possible
- **Condition**: All channels divisible by 17
- **Impact**: 50% size reduction for compatible colors
- **Quality Impact**: None (same color)

---

## 4. Compression Modes

### 4.1 Standard Mode (`regions_to_svg`)
- **Options**:
  - `compact=True`: Uses `_regions_to_svg_compact()`
  - `compact=False`: Uses `_regions_to_svg_pretty()` (pretty-printed)
- **Optimizations**:
  - Color grouping by `<g>` elements
  - Short gradient IDs (a, b, c...)
  - Relative coordinates
  - Compact number formatting

### 4.2 Ultra Compressed Mode (`generate_ultra_compressed_svg`)
**WARNING: Quality loss expected**
- 1-pixel coordinate quantization
- 0 decimal precision (integers only)
- Aggressive color merging (5% tolerance)
- Path deduplication
- Ultra-compact number formatting

### 4.3 Merged Mode (`generate_merged_svg`)
**WARNING: May create artifacts**
- Merges ALL same-color paths globally (not just adjacent)
- 16 color levels quantization
- Integer-only coordinates
- Single quotes, no spaces
- Removes `xmlns` attribute

### 4.4 Extreme Mode (`generate_extreme_svg`)
**WARNING: Significant quality loss**
- 2-pixel coordinate quantization
- 2.0 tolerance curve simplification
- Only 8 color levels
- Maximum aggression

### 4.5 Insane Mode (`generate_insane_svg`)
**WARNING: Massive quality loss**
- 4-pixel coordinate quantization
- 4.0 tolerance curve simplification
- Only 4 color levels
- Very coarse detail

### 4.6 Monochrome Mode (`generate_monochrome_svg`)
**WARNING: Extreme quality loss - only black and white!**
- 8-pixel coordinate quantization
- 8.0 tolerance curve simplification
- Converts all colors to black or white based on luminance
- Background is white, dark areas become black paths

### 4.7 Symbol Optimized Mode (`generate_symbol_optimized_svg`)
- Uses SVG `<symbol>` and `<use>` elements
- Best for images with repeated patterns
- Detects duplicate paths and references them
- 16 color levels, 0.5 tolerance

---

## 5. Complete Optimization Mode (`generate_optimized_svg`)

This is the main optimization function used by default. It applies:

1. **Path Simplification**: `simplify_tolerance=0.5`
2. **Coordinate Quantization**: `quantization_grid=0.5`
3. **Adaptive Precision**: Based on coordinate analysis
4. **Relative Coordinates**: Enabled
5. **Compact Number Formatting**: Removes trailing zeros
6. **Color Grouping**: Groups by fill color
7. **Minified XML**: No whitespace

**Default Parameters**:
```python
generate_optimized_svg(
    regions, width, height,
    quantization_grid=0.5,    # 0.5 pixel grid
    simplify_tolerance=0.5,    # Moderate simplification
    base_precision=2          # 2 decimal places
)
```

---

## 6. Potential Issues & Debugging

### Issue: Blocky/Pixelated Output
**Cause**: Coordinate quantization too aggressive
**Solution**: Reduce `quantization_grid` or use `compact=False`

### Issue: Color Banding/Posterization
**Cause**: Color quantization levels too low
**Solution**: Avoid ultra/extreme/insane modes

### Issue: Lost Curves/Smoothness
**Cause**: Path simplification tolerance too high
**Solution**: Reduce `simplify_tolerance` in config

### Issue: Wrong Colors
**Cause**: Color merging combining different colors
**Solution**: Increase merge threshold or disable color grouping

### Issue: Missing Details
**Cause**: Multiple aggressive optimizations combined
**Solution**: Use standard mode or increase precision settings

---

## 7. Usage Examples

### High Quality (No Optimization)
```python
svg = regions_to_svg(regions, width, height, compact=False)
```

### Standard Optimization (Recommended)
```python
svg = generate_optimized_svg(
    regions, width, height,
    quantization_grid=0.5,
    simplify_tolerance=0.5,
    base_precision=2
)
```

### Maximum Compression (Quality Loss)
```python
svg = generate_ultra_compressed_svg(regions, width, height)
# or
svg = generate_extreme_svg(regions, width, height)
```

### Disable Specific Optimizations
```python
# No coordinate quantization
quantized = bezier_curves  # Skip quantize_coordinates()

# No path simplification
simplified = bezier_curves  # Skip simplify_bezier_curves()

# Absolute coordinates
path_data = _bezier_to_svg_path(curves, use_relative=False)
```

---

## 8. Testing Quality Impact

To verify optimization impact on your images:

1. **Generate with and without optimization**:
   ```bash
   # Standard (optimized)
   python -m vectorizer input.png -o output.svg
   
   # Manual high quality
   python -c "from vectorizer.svg_optimizer import regions_to_svg; ..."
   ```

2. **Compare visual output**:
   - Open both SVGs in browser
   - Zoom to 100% and 400%
   - Check for blockiness, color shifts, lost details

3. **Check file sizes**:
   ```bash
   ls -la output*.svg
   ```

4. **Validate with SSIM**:
   ```bash
   python -m vectorizer input.png -o output.svg --validate
   ```

---

## 9. Recommendations by Use Case

| Use Case | Recommended Mode | Settings |
|----------|-----------------|----------|
| Web display | Standard | Default settings |
| Print quality | Standard | `quantization_grid=0.25` |
| Maximum compression | Ultra | Use with caution |
| Archival | Standard | `compact=False` |
| Icons/Logos | Symbol | For repeated shapes |
| Preview/Thumbnail | Extreme | Acceptable quality loss |

---

## Summary

The optimizer applies multiple compression techniques that can compound to cause visible quality degradation. Key settings to adjust if quality issues occur:

1. **Increase coordinate precision**: Raise `base_precision` to 3 or 4
2. **Reduce quantization**: Set `quantization_grid` to 0.25 or 0.1
3. **Reduce simplification**: Lower `simplify_tolerance` to 0.1 or 0.2
4. **Avoid aggressive modes**: Don't use ultra/extreme/insane modes for quality work
