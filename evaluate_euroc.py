#!/usr/bin/env python3
"""
EuRoC MAV Dataset Evaluation Script
====================================
Evaluates the camera motion estimator against EuRoC ground truth.

Usage:
    python evaluate_euroc.py <euroc_sequence_path> [--num_pairs N]

Example:
    python evaluate_euroc.py euroc_test/machine_hall/MH_01_easy/mav0 --num_pairs 10
"""

import argparse
import csv
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# Import our motion estimator
from camera_motion_estimator import CameraMotionEstimator, MotionEstimationResult


def load_camera_body_transform(sensor_yaml_path: str) -> np.ndarray:
    """
    Load camera-to-body transformation matrix T_BS from sensor.yaml.
    This transforms points from body frame to camera frame.
    """
    with open(sensor_yaml_path, 'r') as f:
        content = f.read()
    
    # Extract T_BS matrix data
    match = re.search(r'T_BS:.*?data:\s*\[([^\]]+)\]', content, re.DOTALL)
    if match:
        data = [float(x.strip()) for x in match.group(1).replace('\n', '').split(',')]
        T_BS = np.array(data).reshape(4, 4)
        return T_BS
    else:
        # Return identity if not found
        return np.eye(4)


def load_euroc_calibration(sensor_yaml_path: str) -> np.ndarray:
    """Load camera calibration from EuRoC sensor.yaml file."""
    # Parse YAML manually (avoid pyyaml dependency)
    with open(sensor_yaml_path, 'r') as f:
        content = f.read()
    
    # Extract intrinsics: [fu, fv, cu, cv]
    match = re.search(r'intrinsics:\s*\[([^\]]+)\]', content)
    if match:
        intrinsics = [float(x.strip()) for x in match.group(1).split(',')]
        fu, fv, cu, cv = intrinsics
        
        K = np.array([
            [fu, 0, cu],
            [0, fv, cv],
            [0, 0, 1]
        ], dtype=np.float64)
        return K
    else:
        raise ValueError("Could not parse intrinsics from sensor.yaml")


def load_euroc_ground_truth(gt_csv_path: str) -> Dict[int, Dict]:
    """
    Load ground truth poses from EuRoC data.csv file.
    Returns dict mapping timestamp to pose (position + quaternion).
    """
    ground_truth = {}
    
    with open(gt_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#'):
                continue  # Skip header
            
            timestamp = int(row[0])
            # Position: p_RS_R_x, p_RS_R_y, p_RS_R_z
            position = np.array([float(row[1]), float(row[2]), float(row[3])])
            # Quaternion: q_RS_w, q_RS_x, q_RS_y, q_RS_z (w, x, y, z format)
            quaternion = np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])])
            
            ground_truth[timestamp] = {
                'position': position,
                'quaternion': quaternion  # w, x, y, z
            }
    
    return ground_truth


def load_euroc_image_timestamps(data_csv_path: str) -> List[int]:
    """Load image timestamps from cam0/data.csv."""
    timestamps = []
    
    with open(data_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith('#'):
                continue
            timestamps.append(int(row[0]))
    
    return sorted(timestamps)


def find_closest_ground_truth(timestamp: int, ground_truth: Dict[int, Dict], max_diff_ns: int = 50000000) -> Optional[Dict]:
    """Find closest ground truth pose to given timestamp (within max_diff nanoseconds)."""
    gt_timestamps = list(ground_truth.keys())
    
    # Binary search for closest
    idx = np.searchsorted(gt_timestamps, timestamp)
    
    candidates = []
    if idx > 0:
        candidates.append(gt_timestamps[idx - 1])
    if idx < len(gt_timestamps):
        candidates.append(gt_timestamps[idx])
    
    if not candidates:
        return None
    
    closest = min(candidates, key=lambda t: abs(t - timestamp))
    
    if abs(closest - timestamp) > max_diff_ns:
        return None
    
    return ground_truth[closest]


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    # scipy uses (x, y, z, w) format
    q_scipy = np.array([q[1], q[2], q[3], q[0]])  # Convert from w,x,y,z to x,y,z,w
    r = Rotation.from_quat(q_scipy)
    return r.as_matrix()


def compute_relative_pose(pose1: Dict, pose2: Dict, T_BC: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute relative pose from pose1 to pose2 in camera frame.
    
    Args:
        pose1: First pose with 'quaternion' and 'position' in body frame
        pose2: Second pose with 'quaternion' and 'position' in body frame
        T_BC: 4x4 transformation matrix from body to camera frame (optional)
    
    Returns (R_rel, t_rel, baseline) where:
        - pose2 = pose1 * (R_rel, t_rel) in camera frame
        - baseline is the actual translation magnitude in meters
    """
    # Get rotation matrices in body frame
    R1_body = quaternion_to_rotation_matrix(pose1['quaternion'])
    R2_body = quaternion_to_rotation_matrix(pose2['quaternion'])
    
    # Get positions in world frame
    t1_world = pose1['position']
    t2_world = pose2['position']
    
    # Relative pose in body frame
    R_rel_body = R1_body.T @ R2_body
    t_rel_body = R1_body.T @ (t2_world - t1_world)
    
    # Transform to camera frame if T_BC is provided
    if T_BC is not None:
        R_BC = T_BC[:3, :3]  # Body to camera rotation
        
        # Transform rotation to camera frame: R_cam = R_BC @ R_body @ R_BC.T
        R_rel_cam = R_BC @ R_rel_body @ R_BC.T
        
        # Transform translation to camera frame: t_cam = R_BC @ t_body
        t_rel_cam = R_BC @ t_rel_body
        
        R_rel = R_rel_cam
        t_rel = t_rel_cam
    else:
        R_rel = R_rel_body
        t_rel = t_rel_body
    
    # Compute baseline (translation magnitude)
    baseline = np.linalg.norm(t_rel)
    
    # Normalize translation to unit vector (since monocular VO only gives direction)
    if baseline > 1e-6:
        t_rel_unit = t_rel / baseline
    else:
        t_rel_unit = t_rel
    
    return R_rel, t_rel_unit, float(baseline)


def compute_rotation_error(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R_est.T @ R_gt
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    return np.degrees(angle)


def compute_translation_error(t_est: np.ndarray, t_gt: np.ndarray, flip_z: bool = False) -> float:
    """
    Compute translation direction error in degrees (since scale is unknown).
    
    Args:
        t_est: Estimated translation vector
        t_gt: Ground truth translation vector
        flip_z: If True, flip the Z component of estimated translation
                (for coordinate system convention differences)
    """
    t_est_flat = t_est.flatten().copy()
    t_gt_flat = t_gt.flatten()
    
    # Apply Z flip if requested (handles OpenCV vs EuRoC convention difference)
    if flip_z:
        t_est_flat[2] = -t_est_flat[2]
    
    # Normalize
    t_est_norm = t_est_flat / (np.linalg.norm(t_est_flat) + 1e-10)
    t_gt_norm = t_gt_flat / (np.linalg.norm(t_gt_flat) + 1e-10)
    
    # Compute angle - translation can be in either direction (sign ambiguity)
    dot = np.clip(np.dot(t_est_norm, t_gt_norm), -1, 1)
    angle1 = np.arccos(dot)  # Direct comparison
    angle2 = np.arccos(-dot)  # Opposite direction
    
    # Take the smaller angle (accounts for sign ambiguity in monocular VO)
    angle = min(angle1, angle2)
    
    return np.degrees(angle)


def main():
    parser = argparse.ArgumentParser(description='Evaluate motion estimator on EuRoC dataset')
    parser.add_argument('euroc_path', type=str, help='Path to EuRoC mav0 folder')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of image pairs to test')
    parser.add_argument('--start_idx', type=int, default=100, help='Starting image index')
    parser.add_argument('--step', type=int, default=1, help='Step between consecutive frames')
    parser.add_argument('--output', type=str, default='euroc_evaluation.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Print details for each pair')
    parser.add_argument('--calibration', type=str, default=None, help='Path to calibration JSON file (optional)')
    parser.add_argument('--flip_z', action='store_true', help='Flip Z axis of estimated translation (for coordinate convention)')

    
    args = parser.parse_args()
    
    euroc_path = Path(args.euroc_path)
    
    # Check paths
    cam0_path = euroc_path / 'cam0'
    gt_path = euroc_path / 'state_groundtruth_estimate0'
    
    if not cam0_path.exists():
        print(f"Error: cam0 folder not found at {cam0_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("EuRoC MAV Dataset Evaluation")
    print("=" * 70)
    
    # Load calibration
    print("\n[1] Loading calibration...")
    
    # Check for existing calibration file or use provided one
    if args.calibration:
        calib_file = Path(args.calibration)
    else:
        # Look for calibration file in parent directory first
        calib_file = euroc_path.parent / 'calibration_cam0.json'
        if not calib_file.exists():
            calib_file = euroc_path / 'calibration.json'
    
    if calib_file.exists():
        print(f"    Using existing calibration: {calib_file}")
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
            K = np.array(calib_data['K'], dtype=np.float64)
    else:
        # Parse from sensor.yaml
        K = load_euroc_calibration(str(cam0_path / 'sensor.yaml'))
        # Save for future use
        calib_file = euroc_path / 'calibration.json'
        with open(calib_file, 'w') as f:
            json.dump({'K': K.tolist()}, f)
        print(f"    Created calibration file: {calib_file}")
    
    print(f"    Camera matrix K:\n{K}")
    
    # Load camera-body transformation (T_BS = body-to-sensor, we need sensor-to-body inverse for ground truth)
    print("\n[2] Loading camera-body transformation...")
    T_BS = load_camera_body_transform(str(cam0_path / 'sensor.yaml'))
    # T_BS transforms from body to camera, so T_BC (body-to-camera) = T_BS
    T_BC = T_BS
    R_BC = T_BC[:3, :3]
    print(f"    T_BS (body-to-camera) loaded")
    print(f"    Camera orientation relative to body (R_BC):\n{R_BC}")
    
    # Load ground truth
    print("\n[3] Loading ground truth...")
    ground_truth = load_euroc_ground_truth(str(gt_path / 'data.csv'))
    print(f"    Loaded {len(ground_truth)} ground truth poses")
    
    # Load image timestamps
    print("\n[4] Loading image timestamps...")
    timestamps = load_euroc_image_timestamps(str(cam0_path / 'data.csv'))
    print(f"    Found {len(timestamps)} images")
    
    # Get image list
    image_dir = cam0_path / 'data'
    images = sorted(image_dir.glob('*.png'))
    print(f"    Image files: {len(images)}")
    
    # Initialize estimator
    print("\n[4] Initializing motion estimator...")
    estimator = CameraMotionEstimator(str(calib_file))
    
    # Evaluation results
    results = []
    rotation_errors = []
    translation_errors = []
    
    print(f"\n[5] Evaluating {args.num_pairs} image pairs...")
    print("-" * 70)
    
    skipped = 0
    errors_count = 0
    
    for i in range(args.num_pairs):
        idx1 = args.start_idx + i * args.step
        idx2 = idx1 + args.step
        
        if idx2 >= len(images):
            print(f"    Reached end of sequence at pair {i}")
            break
        
        img1_path = str(images[idx1])
        img2_path = str(images[idx2])
        
        ts1 = timestamps[idx1]
        ts2 = timestamps[idx2]
        
        # Get ground truth poses
        gt1 = find_closest_ground_truth(ts1, ground_truth)
        gt2 = find_closest_ground_truth(ts2, ground_truth)
        
        if gt1 is None or gt2 is None:
            skipped += 1
            if args.verbose:
                print(f"    Pair {i}: No ground truth available, skipping")
            continue
        
        # Compute ground truth relative pose (transformed to camera frame)
        R_gt, t_gt, baseline = compute_relative_pose(gt1, gt2, T_BC)
        
        # Skip pairs with insufficient baseline (translation direction undefined)
        MIN_BASELINE = 0.01  # 1cm minimum translation
        if baseline < MIN_BASELINE:
            skipped += 1
            if args.verbose:
                print(f"    Pair {i}: Baseline too small ({baseline*1000:.1f}mm < {MIN_BASELINE*1000:.0f}mm), skipping")
            continue
        
        # Run our estimator (suppress output for cleaner logs)
        try:
            # Temporarily redirect stdout to suppress estimator output
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                result = estimator.estimate(img1_path, img2_path)
            
            # Compute errors (flip_z for coordinate convention difference)
            rot_err = compute_rotation_error(result.R, R_gt)
            trans_err = compute_translation_error(result.t, t_gt, flip_z=args.flip_z)
            
            rotation_errors.append(rot_err)
            translation_errors.append(trans_err)
            
            results.append({
                'pair': i,
                'idx1': idx1,
                'idx2': idx2,
                'baseline_m': baseline,
                'rotation_error_deg': rot_err,
                'translation_error_deg': trans_err,
                'quality_score': result.metrics.quality_score,
                'inliers': result.metrics.ransac_inliers,
                'reprojection_error': result.metrics.mean_reprojection_error
            })
            
            status = "✓" if rot_err < 5 and trans_err < 15 else "✗"
            
            if args.verbose or (i % 50 == 0):
                print(f"    Pair {i:4d}/{args.num_pairs}: R_err={rot_err:6.2f}° | t_err={trans_err:6.2f}° | "
                      f"B={baseline*100:5.1f}cm | Q={result.metrics.quality_score:5.1f} | inliers={result.metrics.ransac_inliers:4d} {status}")
            
            # Debug: show severe failures
            if trans_err > 60 and args.verbose:
                t_est_flat = result.t.flatten().copy()
                if args.flip_z:
                    t_est_flat[2] = -t_est_flat[2]
                t_est_norm = t_est_flat / (np.linalg.norm(t_est_flat) + 1e-10)
                t_gt_norm = t_gt / (np.linalg.norm(t_gt) + 1e-10)
                print(f"         t_est={t_est_norm}, t_gt={t_gt_norm}")
            
        except Exception as e:
            errors_count += 1
            if args.verbose:
                print(f"    Pair {i}: Error - {str(e)}")
            continue
    
    print(f"\n    Completed: {len(results)} pairs evaluated, {skipped} skipped, {errors_count} errors")
    
    # Initialize accuracy variables with defaults
    rot_1deg = rot_2deg = rot_5deg = rot_10deg = 0.0
    trans_5deg = trans_10deg = trans_15deg = trans_30deg = 0.0
    success_strict = success_normal = success_relaxed = 0.0
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if rotation_errors:
        rot_arr = np.array(rotation_errors)
        trans_arr = np.array(translation_errors)
        
        print(f"\n[ROTATION ERROR (degrees)]:")
        print(f"    Mean:     {np.mean(rot_arr):7.3f} deg")
        print(f"    Median:   {np.median(rot_arr):7.3f} deg")
        print(f"    Std:      {np.std(rot_arr):7.3f} deg")
        print(f"    Min:      {np.min(rot_arr):7.3f} deg")
        print(f"    Max:      {np.max(rot_arr):7.3f} deg")
        
        print(f"\n[TRANSLATION DIRECTION ERROR (degrees)]:")
        print(f"    Mean:     {np.mean(trans_arr):7.3f} deg")
        print(f"    Median:   {np.median(trans_arr):7.3f} deg")
        print(f"    Std:      {np.std(trans_arr):7.3f} deg")
        print(f"    Min:      {np.min(trans_arr):7.3f} deg")
        print(f"    Max:      {np.max(trans_arr):7.3f} deg")
        
        print(f"\n[ACCURACY THRESHOLDS]:")
        
        # Rotation accuracy at different thresholds
        rot_1deg = np.sum(rot_arr < 1) / len(rot_arr) * 100
        rot_2deg = np.sum(rot_arr < 2) / len(rot_arr) * 100
        rot_5deg = np.sum(rot_arr < 5) / len(rot_arr) * 100
        rot_10deg = np.sum(rot_arr < 10) / len(rot_arr) * 100
        
        print(f"\n    Rotation accuracy:")
        print(f"      < 1°:   {rot_1deg:6.1f}%  ({np.sum(rot_arr < 1)}/{len(rot_arr)})")
        print(f"      < 2°:   {rot_2deg:6.1f}%  ({np.sum(rot_arr < 2)}/{len(rot_arr)})")
        print(f"      < 5°:   {rot_5deg:6.1f}%  ({np.sum(rot_arr < 5)}/{len(rot_arr)})")
        print(f"      < 10°:  {rot_10deg:6.1f}%  ({np.sum(rot_arr < 10)}/{len(rot_arr)})")
        
        # Translation accuracy at different thresholds
        trans_5deg = np.sum(trans_arr < 5) / len(trans_arr) * 100
        trans_10deg = np.sum(trans_arr < 10) / len(trans_arr) * 100
        trans_15deg = np.sum(trans_arr < 15) / len(trans_arr) * 100
        trans_30deg = np.sum(trans_arr < 30) / len(trans_arr) * 100
        
        print(f"\n    Translation direction accuracy:")
        print(f"      < 5°:   {trans_5deg:6.1f}%  ({np.sum(trans_arr < 5)}/{len(trans_arr)})")
        print(f"      < 10°:  {trans_10deg:6.1f}%  ({np.sum(trans_arr < 10)}/{len(trans_arr)})")
        print(f"      < 15°:  {trans_15deg:6.1f}%  ({np.sum(trans_arr < 15)}/{len(trans_arr)})")
        print(f"      < 30°:  {trans_30deg:6.1f}%  ({np.sum(trans_arr < 30)}/{len(trans_arr)})")
        
        # Combined success rates
        success_strict = np.sum((rot_arr < 2) & (trans_arr < 10)) / len(rot_arr) * 100
        success_normal = np.sum((rot_arr < 5) & (trans_arr < 15)) / len(rot_arr) * 100
        success_relaxed = np.sum((rot_arr < 10) & (trans_arr < 30)) / len(rot_arr) * 100
        
        print(f"\n    Combined success rates (R and t):")
        print(f"      Strict  (R<2°, t<10°):   {success_strict:6.1f}%")
        print(f"      Normal  (R<5°, t<15°):   {success_normal:6.1f}%")
        print(f"      Relaxed (R<10°, t<30°):  {success_relaxed:6.1f}%")
        
        print("\n" + "=" * 70)
        print(f">>> OVERALL ACCURACY: {success_normal:.1f}% (R<5deg, t<15deg threshold)")
        print("=" * 70)
    
    # Save results
    output_data = {
        'euroc_sequence': str(euroc_path),
        'num_pairs_evaluated': len(results),
        'num_pairs_requested': args.num_pairs,
        'calibration_file': str(calib_file),
        'summary': {
            'rotation_error': {
                'mean': float(np.mean(rotation_errors)) if rotation_errors else None,
                'median': float(np.median(rotation_errors)) if rotation_errors else None,
                'std': float(np.std(rotation_errors)) if rotation_errors else None,
                'min': float(np.min(rotation_errors)) if rotation_errors else None,
                'max': float(np.max(rotation_errors)) if rotation_errors else None,
            },
            'translation_error': {
                'mean': float(np.mean(translation_errors)) if translation_errors else None,
                'median': float(np.median(translation_errors)) if translation_errors else None,
                'std': float(np.std(translation_errors)) if translation_errors else None,
                'min': float(np.min(translation_errors)) if translation_errors else None,
                'max': float(np.max(translation_errors)) if translation_errors else None,
            },
            'accuracy_percentages': {
                'rotation_under_1deg': float(rot_1deg) if rotation_errors else 0.0,
                'rotation_under_2deg': float(rot_2deg) if rotation_errors else 0.0,
                'rotation_under_5deg': float(rot_5deg) if rotation_errors else 0.0,
                'translation_under_5deg': float(trans_5deg) if rotation_errors else 0.0,
                'translation_under_10deg': float(trans_10deg) if rotation_errors else 0.0,
                'translation_under_15deg': float(trans_15deg) if rotation_errors else 0.0,
                'combined_strict': float(success_strict) if rotation_errors else 0.0,
                'combined_normal': float(success_normal) if rotation_errors else 0.0,
                'combined_relaxed': float(success_relaxed) if rotation_errors else 0.0,
            }
        },
        'per_pair_results': results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
