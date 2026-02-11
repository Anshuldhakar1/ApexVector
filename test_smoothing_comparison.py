"""Test harness for comparing boundary smoothing techniques."""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
from apexvec.smoothing_experiments import (
    smooth_gaussian,
    smooth_moving_average,
    smooth_savgol,
    smooth_cubic_spline,
    smooth_bspline_scipy,
    smooth_douglas_peucker,
    points_to_bezier_curves
)


def create_test_contours():
    """Create various test contours with different characteristics."""
    contours = {}
    
    # 1. Circle with noise
    t = np.linspace(0, 2*np.pi, 100)
    radius = 50
    noise = np.random.randn(100) * 2
    circle_noisy = np.column_stack([
        (radius + noise) * np.cos(t) + 100,
        (radius + noise) * np.sin(t) + 100
    ])
    contours['circle_noisy'] = circle_noisy
    
    # 2. Jagged rectangle
    x = np.concatenate([
        np.linspace(0, 100, 25),
        np.full(25, 100),
        np.linspace(100, 0, 25),
        np.full(25, 0)
    ])
    y = np.concatenate([
        np.full(25, 0),
        np.linspace(0, 100, 25),
        np.full(25, 100),
        np.linspace(100, 0, 25)
    ])
    rectangle = np.column_stack([x + np.random.randn(100)*2, y + np.random.randn(100)*2])
    contours['rectangle_jagged'] = rectangle
    
    # 3. Star shape (high frequency detail)
    t = np.linspace(0, 2*np.pi, 200)
    r = 50 + 20 * np.sin(5*t)
    star = np.column_stack([
        r * np.cos(t) + 150,
        r * np.sin(t) + 150
    ])
    contours['star'] = star
    
    # 4. Smooth ellipse (should preserve)
    t = np.linspace(0, 2*np.pi, 100)
    ellipse = np.column_stack([
        80 * np.cos(t) + 100,
        40 * np.sin(t) + 100
    ])
    contours['ellipse'] = ellipse
    
    # 5. Zigzag (high frequency noise)
    x = np.linspace(0, 200, 100)
    y = 100 + 10 * np.sin(x * 0.5) + np.random.randn(100) * 3
    zigzag = np.column_stack([x, y])
    contours['zigzag'] = zigzag
    
    return contours


def measure_smoothness(points):
    """Measure boundary smoothness using angle changes."""
    if len(points) < 3:
        return float('inf')
    
    # Calculate angles between consecutive segments
    angles = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        p2 = points[(i + 2) % len(points)]
        
        v1 = p1 - p0
        v2 = p2 - p1
        
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angles.append(abs(angle))
    
    # Return average angle change (lower is smoother)
    return np.mean(angles) if angles else float('inf')


def measure_shape_preservation(original, smoothed):
    """Measure how well the smoothed version preserves the original shape."""
    if len(original) != len(smoothed):
        # Interpolate to match lengths
        from scipy.interpolate import interp1d
        t_orig = np.linspace(0, 1, len(original))
        t_smooth = np.linspace(0, 1, len(smoothed))
        
        f_x = interp1d(t_orig, original[:, 0], kind='linear', fill_value='extrapolate')
        f_y = interp1d(t_orig, original[:, 1], kind='linear', fill_value='extrapolate')
        
        original_resampled = np.column_stack([f_x(t_smooth), f_y(t_smooth)])
    else:
        original_resampled = original
    
    # Calculate Hausdorff distance
    distances = np.sqrt(np.sum((original_resampled - smoothed)**2, axis=1))
    return np.max(distances)


def test_technique(technique_name, points, **kwargs):
    """Test a single smoothing technique."""
    try:
        if technique_name == 'gaussian':
            smoothed = smooth_gaussian(points, **kwargs)
        elif technique_name == 'moving_average':
            smoothed = smooth_moving_average(points, **kwargs)
        elif technique_name == 'savgol':
            smoothed = smooth_savgol(points, **kwargs)
        elif technique_name == 'cubic_spline':
            smoothed = smooth_cubic_spline(points, **kwargs)
        elif technique_name == 'bspline':
            smoothed = smooth_bspline_scipy(points, **kwargs)
        elif technique_name == 'douglas_peucker':
            smoothed = smooth_douglas_peucker(points, **kwargs)
        else:
            raise ValueError(f"Unknown technique: {technique_name}")
        
        # Calculate metrics
        smoothness = measure_smoothness(smoothed)
        preservation = measure_shape_preservation(points, smoothed)
        num_points = len(smoothed)
        
        return {
            'success': True,
            'points': smoothed,
            'smoothness': smoothness,
            'preservation': preservation,
            'num_points': num_points,
            'reduction_ratio': len(points) / num_points if num_points > 0 else 1.0
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def run_comparison_tests():
    """Run comprehensive comparison of all smoothing techniques."""
    print("=" * 80)
    print("Boundary Smoothing Technique Comparison")
    print("=" * 80)
    
    # Create test contours
    contours = create_test_contours()
    
    # Define techniques with parameters
    techniques = {
        'gaussian': {'sigma': 2.0},
        'moving_average': {'window_size': 5},
        'savgol': {'window_length': 7, 'polyorder': 3},
        'cubic_spline': {},
        'bspline': {'smoothness': 1.0},
        'douglas_peucker': {'epsilon': 2.0},
    }
    
    # Store results
    all_results = {}
    
    # Test each contour with each technique
    for contour_name, points in contours.items():
        print(f"\nTesting contour: {contour_name} ({len(points)} points)")
        print("-" * 60)
        
        contour_results = {}
        
        for tech_name, params in techniques.items():
            result = test_technique(tech_name, points.copy(), **params)
            contour_results[tech_name] = result
            
            if result['success']:
                print(f"  {tech_name:20s}: smoothness={result['smoothness']:.4f}, "
                      f"preservation={result['preservation']:.2f}, "
                      f"points={result['num_points']:3d}")
            else:
                print(f"  {tech_name:20s}: FAILED - {result.get('error', 'Unknown error')}")
        
        all_results[contour_name] = contour_results
    
    # Create visualization
    create_comparison_plots(contours, all_results, techniques)
    
    # Create summary report
    create_summary_report(all_results)
    
    return all_results


def create_comparison_plots(contours, results, techniques):
    """Create visualization comparing all techniques."""
    output_dir = Path('smoothing_comparison')
    output_dir.mkdir(exist_ok=True)
    
    for contour_name, points in contours.items():
        n_techniques = len(techniques)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Smoothing Comparison: {contour_name}', fontsize=16)
        
        axes = axes.flatten()
        
        for idx, (tech_name, params) in enumerate(techniques.items()):
            ax = axes[idx]
            
            # Plot original
            ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.3, linewidth=1, label='Original')
            
            # Plot smoothed
            result = results[contour_name][tech_name]
            if result['success']:
                smoothed = result['points']
                ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='Smoothed')
                
                # Add metrics as text
                metrics_text = (
                    f"Smoothness: {result['smoothness']:.4f}\n"
                    f"Preservation: {result['preservation']:.2f}\n"
                    f"Points: {result['num_points']}"
                )
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                ax.text(0.5, 0.5, 'FAILED', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, color='red')
            
            ax.set_title(tech_name.replace('_', ' ').title())
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{contour_name}_comparison.png', dpi=150)
        plt.close()
    
    print(f"\nPlots saved to: {output_dir}")


def create_summary_report(results):
    """Create JSON summary of results."""
    # Calculate average metrics for each technique
    summary = {}
    
    for contour_name, contour_results in results.items():
        for tech_name, result in contour_results.items():
            if tech_name not in summary:
                summary[tech_name] = {
                    'smoothness': [],
                    'preservation': [],
                    'reduction_ratio': [],
                    'success_count': 0,
                    'fail_count': 0
                }
            
            if result['success']:
                summary[tech_name]['smoothness'].append(result['smoothness'])
                summary[tech_name]['preservation'].append(result['preservation'])
                summary[tech_name]['reduction_ratio'].append(result['reduction_ratio'])
                summary[tech_name]['success_count'] += 1
            else:
                summary[tech_name]['fail_count'] += 1
    
    # Calculate averages
    for tech_name in summary:
        stats = summary[tech_name]
        if stats['success_count'] > 0:
            stats['avg_smoothness'] = np.mean(stats['smoothness'])
            stats['avg_preservation'] = np.mean(stats['preservation'])
            stats['avg_reduction'] = np.mean(stats['reduction_ratio'])
        
        # Clean up lists for JSON
        del stats['smoothness']
        del stats['preservation']
        del stats['reduction_ratio']
    
    # Save report
    output_path = Path('smoothing_comparison') / 'summary_report.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Summary: Average Performance Across All Contours")
    print("=" * 80)
    print(f"{'Technique':<20} {'Success':<10} {'Smoothness':<12} {'Preservation':<12} {'Reduction':<10}")
    print("-" * 80)
    
    for tech_name in sorted(summary.keys()):
        stats = summary[tech_name]
        if stats['success_count'] > 0:
            print(f"{tech_name:<20} {stats['success_count']:<10} "
                  f"{stats.get('avg_smoothness', 0):<12.4f} "
                  f"{stats.get('avg_preservation', 0):<12.2f} "
                  f"{stats.get('avg_reduction', 0):<10.2f}")
        else:
            print(f"{tech_name:<20} {stats['success_count']:<10} FAILED")


if __name__ == '__main__':
    # Run tests
    results = run_comparison_tests()
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("Check 'smoothing_comparison/' directory for visualizations")
    print("=" * 80)
