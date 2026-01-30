#!/usr/bin/env python3
"""
KITTI Visual Odometry Dataset Evaluation Script

Evaluates the camera motion estimator against the KITTI odometry benchmark.
Supports sequences 00-10 (with ground truth) and 11-21 (test sequences).

Dataset structure expected:
    kitti_odometry/
    ├── sequences/
    │   ├── 00/
    │   │   ├── image_0/        # Left grayscale camera
    │   │   ├── image_1/        # Right grayscale camera
    │   │   ├── image_2/        # Left color camera
    │   │   ├── image_3/        # Right color camera
    │   │   ├── calib.txt       # Calibration
    │   │   └── times.txt       # Timestamps
    │   ├── 01/
    │   │   └── ...
    │   └── ...
    └── poses/
        ├── 00.txt              # Ground truth poses (sequences 00-10)
        ├── 01.txt
        └── ...

Download from: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Download "odometry data set (grayscale, 22 GB)" or "odometry data set (color, 65 GB)"
- Download "odometry ground truth poses (4 MB)"

Usage:
    python evaluate_kitti.py <kitti_path> --sequence 00 --num_pairs 100
    python evaluate_kitti.py <kitti_path> --sequence 00 --start_idx 0 --num_pairs 50 --step 1
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2

from camera_motion_estimator import CameraMotionEstimator, MotionEstimationResult


def load_kitti_calibration(calib_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load KITTI calibration from calib.txt file.
    
    Returns:
        K: 3x3 camera intrinsic matrix for camera 0 (left grayscale) or camera 2 (left color)
        baseline: baseline between stereo cameras (optional)
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
    
    # P0 is the projection matrix for camera 0 (left grayscale)
    # P2 is the projection matrix for camera 2 (left color)
    # Format: P = [fx 0 cx tx; 0 fy cy ty; 0 0 1 tz] (3x4)
    
    # Try P2 (color) first, then P0 (grayscale)
    if 'P2' in calib:
        P = calib['P2'].reshape(3, 4)
    elif 'P0' in calib:
        P = calib['P0'].reshape(3, 4)
    else:
        raise ValueError(f"No projection matrix found in {calib_path}")
    
    # Extract intrinsic matrix K from projection matrix
    K = P[:3, :3]
    
    # Calculate baseline if stereo info available
    baseline = None
    if 'P1' in calib:
        P1 = calib['P1'].reshape(3, 4)
        # Baseline = -tx / fx (tx is P1[0,3], fx is P1[0,0])
        baseline = -P1[0, 3] / P1[0, 0]
    
    return K, baseline


def create_kitti_calibration_json(calib_path: Path, output_path: Path, 
                                   image_width: int = 1241, image_height: int = 376) -> Path:
    """
    Create a calibration JSON file from KITTI calib.txt for our estimator.
    """
    K, baseline = load_kitti_calibration(calib_path)
    
    calib_data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],  # KITTI images are already rectified
        "image_width": image_width,
        "image_height": image_height,
        "source": "KITTI Odometry Dataset",
        "calibration_file": str(calib_path)
    }
    
    if baseline is not None:
        calib_data["stereo_baseline"] = baseline
    
    with open(output_path, 'w') as f:
        json.dump(calib_data, f, indent=2)
    
    return output_path


def load_kitti_poses(poses_path: Path) -> np.ndarray:
    """
    Load KITTI ground truth poses.
    
    Each line contains a 3x4 transformation matrix [R|t] in row-major order.
    This transforms points from the camera coordinate system to the world coordinate system.
    
    Returns:
        poses: Nx4x4 array of transformation matrices
    """
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            if len(values) == 12:
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
                poses.append(T)
    return np.array(poses)


def compute_relative_pose(T1: np.ndarray, T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relative pose from T1 to T2.
    
    Args:
        T1: 4x4 world-to-camera transformation for frame 1
        T2: 4x4 world-to-camera transformation for frame 2
        
    Returns:
        R: 3x3 relative rotation matrix
        t: 3x1 relative translation vector (normalized)
    """
    # Relative transformation: T_rel = T2 @ inv(T1)
    T_rel = T2 @ np.linalg.inv(T1)
    
    R = T_rel[:3, :3]
    t = T_rel[:3, 3:4]
    
    # Normalize translation to unit vector (we only estimate direction)
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-10:
        t = t / t_norm
    
    return R, t


def rotation_error(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_err = R_est @ R_gt.T
    trace = np.clip(np.trace(R_err), -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle)


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Compute translation direction error in degrees."""
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    
    t_est_norm = np.linalg.norm(t_est)
    t_gt_norm = np.linalg.norm(t_gt)
    
    if t_est_norm < 1e-10 or t_gt_norm < 1e-10:
        return 180.0
    
    t_est = t_est / t_est_norm
    t_gt = t_gt / t_gt_norm
    
    # Handle sign ambiguity - take minimum of both orientations
    cos_angle = np.clip(np.dot(t_est, t_gt), -1.0, 1.0)
    angle1 = np.degrees(np.arccos(cos_angle))
    angle2 = 180.0 - angle1
    
    return min(angle1, angle2)


def get_sequence_info(kitti_path: Path, sequence: str) -> Dict:
    """Get information about a KITTI sequence."""
    seq_path = kitti_path / 'sequences' / sequence
    
    if not seq_path.exists():
        raise FileNotFoundError(f"Sequence {sequence} not found at {seq_path}")
    
    # Check for images
    image_dirs = ['image_2', 'image_0', 'image_3', 'image_1']  # Prefer color
    image_dir = None
    for d in image_dirs:
        if (seq_path / d).exists():
            image_dir = seq_path / d
            break
    
    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in {seq_path}")
    
    # Count images
    images = sorted(image_dir.glob('*.png'))
    if not images:
        images = sorted(image_dir.glob('*.jpg'))
    
    # Get image resolution
    sample_img = cv2.imread(str(images[0]))
    height, width = sample_img.shape[:2]
    
    # Check for calibration
    calib_path = seq_path / 'calib.txt'
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    
    # Check for ground truth
    poses_path = kitti_path / 'poses' / f'{sequence}.txt'
    has_gt = poses_path.exists()
    
    return {
        'sequence': sequence,
        'image_dir': image_dir,
        'images': images,
        'num_images': len(images),
        'width': width,
        'height': height,
        'calib_path': calib_path,
        'poses_path': poses_path if has_gt else None,
        'has_ground_truth': has_gt
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate camera motion estimator on KITTI Odometry dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_kitti.py ./kitti_odometry --sequence 00 --num_pairs 100
    python evaluate_kitti.py ./kitti_odometry --sequence 00 --start_idx 0 --end_idx 500 --step 1
    python evaluate_kitti.py ./kitti_odometry --sequence 00 --verbose
    
Download KITTI Odometry dataset from:
    http://www.cvlibs.net/datasets/kitti/eval_odometry.php
        """
    )
    parser.add_argument('kitti_path', type=str, help='Path to KITTI odometry dataset root')
    parser.add_argument('--sequence', type=str, default='00', 
                        help='Sequence number (00-21, default: 00)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting image index (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Ending image index (default: all)')
    parser.add_argument('--num_pairs', type=int, default=100,
                        help='Number of image pairs to evaluate (default: 100)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step between consecutive frames (default: 1)')
    parser.add_argument('--output', type=str, default='kitti_evaluation.json',
                        help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed output for each pair')
    
    args = parser.parse_args()
    
    kitti_path = Path(args.kitti_path)
    if not kitti_path.exists():
        print(f"Error: KITTI path not found: {kitti_path}")
        print("\nTo download KITTI Odometry dataset:")
        print("1. Go to: http://www.cvlibs.net/datasets/kitti/eval_odometry.php")
        print("2. Download 'odometry data set (grayscale)' or 'odometry data set (color)'")
        print("3. Download 'odometry ground truth poses'")
        print("4. Extract to a folder with structure:")
        print("   kitti_odometry/")
        print("   ├── sequences/")
        print("   │   ├── 00/, 01/, ... 21/")
        print("   └── poses/")
        print("       ├── 00.txt, 01.txt, ... 10.txt")
        sys.exit(1)
    
    print("=" * 70)
    print("KITTI Odometry Dataset Evaluation")
    print("=" * 70)
    
    # Get sequence info
    print(f"\n[1] Loading sequence {args.sequence}...")
    try:
        seq_info = get_sequence_info(kitti_path, args.sequence)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"    Image directory: {seq_info['image_dir']}")
    print(f"    Total images: {seq_info['num_images']}")
    print(f"    Resolution: {seq_info['width']}x{seq_info['height']}")
    print(f"    Ground truth: {'Available' if seq_info['has_ground_truth'] else 'Not available'}")
    
    # Load calibration
    print("\n[2] Loading calibration...")
    calib_json = kitti_path / 'sequences' / args.sequence / 'calibration.json'
    create_kitti_calibration_json(
        seq_info['calib_path'], 
        calib_json,
        seq_info['width'],
        seq_info['height']
    )
    print(f"    Created calibration: {calib_json}")
    
    K, baseline = load_kitti_calibration(seq_info['calib_path'])
    print(f"    Camera matrix K:\n{K}")
    if baseline:
        print(f"    Stereo baseline: {baseline:.4f} m")
    
    # Load ground truth if available
    gt_poses = None
    if seq_info['has_ground_truth']:
        print("\n[3] Loading ground truth poses...")
        gt_poses = load_kitti_poses(seq_info['poses_path'])
        print(f"    Loaded {len(gt_poses)} poses")
    else:
        print("\n[3] No ground truth available for this sequence")
        print("    Will run estimation without accuracy evaluation")
    
    # Initialize estimator (preserve_aspect_ratio=True for KITTI)
    print("\n[4] Initializing motion estimator...")
    estimator = CameraMotionEstimator(str(calib_json), preserve_aspect_ratio=True)
    
    # Setup evaluation range
    images = seq_info['images']
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx else len(images) - args.step
    end_idx = min(end_idx, len(images) - args.step)
    
    # Calculate number of pairs
    if args.num_pairs:
        end_idx = min(end_idx, start_idx + args.num_pairs * args.step)
    
    num_pairs = (end_idx - start_idx) // args.step
    
    print(f"\n[5] Evaluating {num_pairs} image pairs...")
    print(f"    Range: frame {start_idx} to {end_idx}, step {args.step}")
    print("-" * 70)
    
    # Evaluation results
    results = []
    rotation_errors = []
    translation_errors = []
    skipped = 0
    errors_count = 0
    
    for i in range(num_pairs):
        idx1 = start_idx + i * args.step
        idx2 = idx1 + args.step
        
        if idx2 >= len(images):
            break
        
        img1_path = str(images[idx1])
        img2_path = str(images[idx2])
        
        try:
            # Run estimation
            result = estimator.estimate(img1_path, img2_path)
            
            pair_result = {
                'pair': i,
                'idx1': idx1,
                'idx2': idx2,
                'quality_score': result.metrics.quality_score,
                'inliers': result.metrics.ransac_inliers,
                'reprojection_error': result.metrics.mean_reprojection_error
            }
            
            # Compare with ground truth if available
            if gt_poses is not None and idx1 < len(gt_poses) and idx2 < len(gt_poses):
                R_gt, t_gt = compute_relative_pose(gt_poses[idx1], gt_poses[idx2])
                
                r_err = rotation_error(result.R, R_gt)
                t_err = translation_error(result.t, t_gt)
                
                pair_result['rotation_error_deg'] = r_err
                pair_result['translation_error_deg'] = t_err
                
                rotation_errors.append(r_err)
                translation_errors.append(t_err)
                
                # Determine success
                success = r_err < 5.0 and t_err < 15.0
                
                if args.verbose or (i % 10 == 0):
                    status = "✓" if success else "✗"
                    print(f"    Pair {i:4d}/{num_pairs}: R_err={r_err:6.2f}° | "
                          f"t_err={t_err:6.2f}° | Q={result.metrics.quality_score:5.1f} | "
                          f"inliers={result.metrics.ransac_inliers:4d} {status}")
            else:
                if args.verbose or (i % 10 == 0):
                    print(f"    Pair {i:4d}/{num_pairs}: Q={result.metrics.quality_score:5.1f} | "
                          f"inliers={result.metrics.ransac_inliers:4d}")
            
            results.append(pair_result)
            
        except Exception as e:
            errors_count += 1
            if args.verbose:
                print(f"    Pair {i}: Error - {str(e)[:50]}")
    
    print(f"\n    Completed: {len(results)} pairs evaluated, {skipped} skipped, {errors_count} errors")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if rotation_errors and translation_errors:
        r_errors = np.array(rotation_errors)
        t_errors = np.array(translation_errors)
        
        print(f"\n[ROTATION ERROR (degrees)]:")
        print(f"    Mean:       {np.mean(r_errors):.3f} deg")
        print(f"    Median:     {np.median(r_errors):.3f} deg")
        print(f"    Std:        {np.std(r_errors):.3f} deg")
        print(f"    Min:        {np.min(r_errors):.3f} deg")
        print(f"    Max:        {np.max(r_errors):.3f} deg")
        
        print(f"\n[TRANSLATION DIRECTION ERROR (degrees)]:")
        print(f"    Mean:       {np.mean(t_errors):.3f} deg")
        print(f"    Median:     {np.median(t_errors):.3f} deg")
        print(f"    Std:        {np.std(t_errors):.3f} deg")
        print(f"    Min:        {np.min(t_errors):.3f} deg")
        print(f"    Max:        {np.max(t_errors):.3f} deg")
        
        # Accuracy thresholds
        n = len(r_errors)
        print(f"\n[ACCURACY THRESHOLDS]:")
        
        print(f"\n    Rotation accuracy:")
        for thresh in [1, 2, 5, 10]:
            pct = 100.0 * np.sum(r_errors < thresh) / n
            print(f"      < {thresh}°:    {pct:5.1f}%  ({np.sum(r_errors < thresh)}/{n})")
        
        print(f"\n    Translation direction accuracy:")
        for thresh in [5, 10, 15, 30]:
            pct = 100.0 * np.sum(t_errors < thresh) / n
            print(f"      < {thresh}°:    {pct:5.1f}%  ({np.sum(t_errors < thresh)}/{n})")
        
        # Combined metrics
        strict = np.sum((r_errors < 2) & (t_errors < 10)) / n * 100
        normal = np.sum((r_errors < 5) & (t_errors < 15)) / n * 100
        relaxed = np.sum((r_errors < 10) & (t_errors < 30)) / n * 100
        
        print(f"\n    Combined success rates (R and t):")
        print(f"      Strict  (R<2°, t<10°):     {strict:.1f}%")
        print(f"      Normal  (R<5°, t<15°):     {normal:.1f}%")
        print(f"      Relaxed (R<10°, t<30°):    {relaxed:.1f}%")
        
        print("\n" + "=" * 70)
        print(f">>> OVERALL ACCURACY: {normal:.1f}% (R<5deg, t<15deg threshold)")
        print("=" * 70)
        
        # Save results
        summary = {
            'kitti_sequence': args.sequence,
            'num_pairs_evaluated': len(results),
            'calibration_file': str(calib_json),
            'summary': {
                'rotation_error': {
                    'mean': float(np.mean(r_errors)),
                    'median': float(np.median(r_errors)),
                    'std': float(np.std(r_errors)),
                    'min': float(np.min(r_errors)),
                    'max': float(np.max(r_errors))
                },
                'translation_error': {
                    'mean': float(np.mean(t_errors)),
                    'median': float(np.median(t_errors)),
                    'std': float(np.std(t_errors)),
                    'min': float(np.min(t_errors)),
                    'max': float(np.max(t_errors))
                },
                'accuracy_percentages': {
                    'rotation_under_1deg': float(np.sum(r_errors < 1) / n * 100),
                    'rotation_under_2deg': float(np.sum(r_errors < 2) / n * 100),
                    'rotation_under_5deg': float(np.sum(r_errors < 5) / n * 100),
                    'translation_under_5deg': float(np.sum(t_errors < 5) / n * 100),
                    'translation_under_10deg': float(np.sum(t_errors < 10) / n * 100),
                    'translation_under_15deg': float(np.sum(t_errors < 15) / n * 100),
                    'combined_strict': float(strict),
                    'combined_normal': float(normal),
                    'combined_relaxed': float(relaxed)
                }
            },
            'per_pair_results': results
        }
    else:
        print("\nNo ground truth available - cannot compute accuracy metrics")
        summary = {
            'kitti_sequence': args.sequence,
            'num_pairs_evaluated': len(results),
            'calibration_file': str(calib_json),
            'per_pair_results': results
        }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
