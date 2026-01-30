#!/usr/bin/env python3
"""
Visualization tools for camera motion estimation evaluation results.

Creates plots for:
- Error distributions (rotation/translation)
- Trajectory visualization
- Per-pair error over time
- Success rate by baseline distance

Usage:
    python visualize_results.py euroc_evaluation.json
    python visualize_results.py kitti_evaluation.json --output plots/
    python visualize_results.py evaluation.json --show
"""

import json
import argparse
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    exit(1)


def load_evaluation_results(json_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_error_distribution(results: dict, output_dir: Path, show: bool = False):
    """Plot histogram of rotation and translation errors."""
    pairs = results.get('per_pair_results', [])
    
    r_errors = [p['rotation_error_deg'] for p in pairs if 'rotation_error_deg' in p]
    t_errors = [p['translation_error_deg'] for p in pairs if 'translation_error_deg' in p]
    
    if not r_errors or not t_errors:
        print("No error data found in results")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rotation error histogram
    ax1 = axes[0]
    ax1.hist(r_errors, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(r_errors), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(r_errors):.2f}°')
    ax1.axvline(np.median(r_errors), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(r_errors):.2f}°')
    ax1.axvline(5.0, color='green', linestyle=':', linewidth=2, label='Threshold: 5°')
    ax1.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Rotation Error Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Translation error histogram
    ax2 = axes[1]
    ax2.hist(t_errors, bins=30, color='coral', edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(t_errors), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(t_errors):.2f}°')
    ax2.axvline(np.median(t_errors), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(t_errors):.2f}°')
    ax2.axvline(15.0, color='green', linestyle=':', linewidth=2, label='Threshold: 15°')
    ax2.set_xlabel('Translation Direction Error (degrees)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Translation Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'error_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_error_over_time(results: dict, output_dir: Path, show: bool = False):
    """Plot errors over frame pairs."""
    pairs = results.get('per_pair_results', [])
    
    indices = [p['pair'] for p in pairs if 'rotation_error_deg' in p]
    r_errors = [p['rotation_error_deg'] for p in pairs if 'rotation_error_deg' in p]
    t_errors = [p['translation_error_deg'] for p in pairs if 'translation_error_deg' in p]
    
    if not r_errors:
        print("No error data found")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Rotation error over time
    ax1 = axes[0]
    ax1.plot(indices, r_errors, 'o-', color='steelblue', markersize=4, alpha=0.7)
    ax1.axhline(5.0, color='green', linestyle='--', linewidth=2, label='Threshold: 5°')
    ax1.fill_between(indices, 0, r_errors, alpha=0.2, color='steelblue')
    ax1.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax1.set_title('Errors Over Frame Pairs', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Translation error over time
    ax2 = axes[1]
    ax2.plot(indices, t_errors, 'o-', color='coral', markersize=4, alpha=0.7)
    ax2.axhline(15.0, color='green', linestyle='--', linewidth=2, label='Threshold: 15°')
    ax2.fill_between(indices, 0, t_errors, alpha=0.2, color='coral')
    ax2.set_xlabel('Frame Pair Index', fontsize=12)
    ax2.set_ylabel('Translation Error (deg)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    output_path = output_dir / 'error_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_error_vs_baseline(results: dict, output_dir: Path, show: bool = False):
    """Plot errors vs baseline distance (if available)."""
    pairs = results.get('per_pair_results', [])
    
    # Check if baseline data exists
    if not any('baseline_m' in p for p in pairs):
        print("No baseline data available, skipping baseline plot")
        return
    
    baselines = [p['baseline_m'] * 100 for p in pairs if 'baseline_m' in p and 'rotation_error_deg' in p]  # Convert to cm
    r_errors = [p['rotation_error_deg'] for p in pairs if 'baseline_m' in p and 'rotation_error_deg' in p]
    t_errors = [p['translation_error_deg'] for p in pairs if 'baseline_m' in p and 'translation_error_deg' in p]
    
    if not baselines:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color by success
    colors = ['green' if r < 5 and t < 15 else 'red' for r, t in zip(r_errors, t_errors)]
    
    # Rotation vs baseline
    ax1 = axes[0]
    ax1.scatter(baselines, r_errors, c=colors, alpha=0.6, s=40)
    ax1.axhline(5.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Baseline Distance (cm)', fontsize=12)
    ax1.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax1.set_title('Rotation Error vs Baseline', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Translation vs baseline
    ax2 = axes[1]
    ax2.scatter(baselines, t_errors, c=colors, alpha=0.6, s=40)
    ax2.axhline(15.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Baseline Distance (cm)', fontsize=12)
    ax2.set_ylabel('Translation Error (deg)', fontsize=12)
    ax2.set_title('Translation Error vs Baseline', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Legend
    success_patch = mpatches.Patch(color='green', label='Success (R<5°, t<15°)')
    fail_patch = mpatches.Patch(color='red', label='Failure')
    ax2.legend(handles=[success_patch, fail_patch], loc='upper right')
    
    plt.tight_layout()
    
    output_path = output_dir / 'error_vs_baseline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_success_summary(results: dict, output_dir: Path, show: bool = False):
    """Plot success rate summary bar chart."""
    summary = results.get('summary', {}).get('accuracy_percentages', {})
    
    if not summary:
        print("No accuracy summary found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data
    metrics = [
        ('Rotation < 1°', summary.get('rotation_under_1deg', 0)),
        ('Rotation < 2°', summary.get('rotation_under_2deg', 0)),
        ('Rotation < 5°', summary.get('rotation_under_5deg', 0)),
        ('Translation < 5°', summary.get('translation_under_5deg', 0)),
        ('Translation < 10°', summary.get('translation_under_10deg', 0)),
        ('Translation < 15°', summary.get('translation_under_15deg', 0)),
        ('Combined Strict\n(R<2°, t<5°)', summary.get('combined_strict', 0)),
        ('Combined Normal\n(R<5°, t<15°)', summary.get('combined_normal', 0)),
        ('Combined Relaxed\n(R<10°, t<30°)', summary.get('combined_relaxed', 0)),
    ]
    
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    
    # Color scheme
    colors = ['#3498db'] * 3 + ['#e74c3c'] * 3 + ['#2ecc71'] * 3
    
    bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Accuracy Metrics Summary', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    rotation_patch = mpatches.Patch(color='#3498db', label='Rotation Metrics')
    translation_patch = mpatches.Patch(color='#e74c3c', label='Translation Metrics')
    combined_patch = mpatches.Patch(color='#2ecc71', label='Combined Metrics')
    ax.legend(handles=[rotation_patch, translation_patch, combined_patch], 
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'success_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def plot_quality_vs_error(results: dict, output_dir: Path, show: bool = False):
    """Plot quality score vs errors."""
    pairs = results.get('per_pair_results', [])
    
    quality = [p['quality_score'] for p in pairs if 'quality_score' in p and 'rotation_error_deg' in p]
    r_errors = [p['rotation_error_deg'] for p in pairs if 'quality_score' in p and 'rotation_error_deg' in p]
    inliers = [p['inliers'] for p in pairs if 'inliers' in p and 'rotation_error_deg' in p]
    
    if not quality:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Quality vs rotation error
    ax1 = axes[0]
    sc1 = ax1.scatter(quality, r_errors, c=inliers, cmap='viridis', alpha=0.7, s=50)
    ax1.axhline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 5°')
    ax1.set_xlabel('Quality Score', fontsize=12)
    ax1.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax1.set_title('Quality Score vs Rotation Error', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.colorbar(sc1, ax=ax1, label='Inliers')
    
    # Inliers vs rotation error
    ax2 = axes[1]
    sc2 = ax2.scatter(inliers, r_errors, c=quality, cmap='plasma', alpha=0.7, s=50)
    ax2.axhline(5.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 5°')
    ax2.set_xlabel('Number of Inliers', fontsize=12)
    ax2.set_ylabel('Rotation Error (deg)', fontsize=12)
    ax2.set_title('Inliers vs Rotation Error', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.colorbar(sc2, ax=ax2, label='Quality')
    
    plt.tight_layout()
    
    output_path = output_dir / 'quality_vs_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize camera motion estimation evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_results.py euroc_evaluation.json
    python visualize_results.py kitti_evaluation.json --output plots/
    python visualize_results.py evaluation.json --show
        """
    )
    parser.add_argument('results_json', type=str, help='Path to evaluation results JSON')
    parser.add_argument('--output', '-o', type=str, default='plots',
                        help='Output directory for plots (default: plots/)')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_json}")
    results = load_evaluation_results(args.results_json)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_error_distribution(results, output_dir, args.show)
    plot_error_over_time(results, output_dir, args.show)
    plot_error_vs_baseline(results, output_dir, args.show)
    plot_success_summary(results, output_dir, args.show)
    plot_quality_vs_error(results, output_dir, args.show)
    
    print("\n✓ All plots generated!")


if __name__ == '__main__':
    main()
