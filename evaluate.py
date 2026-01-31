"""
Evaluation script for Visual Odometry on KITTI Sequence 05
Runs on 100 consecutive frame pairs and calculates total error and correctness.
"""

import numpy as np
import cv2
from pathlib import Path
from visual_odometry import VisualOdometry


def load_poses(path):
    """Load KITTI ground truth poses."""
    poses = []
    with open(path, 'r') as f:
        for line in f:
            values = np.array([float(x) for x in line.strip().split()])
            if len(values) == 12:
                T = np.eye(4)
                T[:3, :] = values.reshape(3, 4)
                poses.append(T)
    return poses


def compute_errors(R_est, t_est, R_gt, t_gt):
    """Compute rotation and translation errors in degrees."""
    # Rotation error
    R_diff = R_est @ R_gt.T
    rot_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    
    # Translation error
    t_est_flat = t_est.flatten()
    t_gt_flat = t_gt.flatten()
    t_est_norm = t_est_flat / (np.linalg.norm(t_est_flat) + 1e-10)
    t_gt_norm = t_gt_flat / (np.linalg.norm(t_gt_flat) + 1e-10)
    cos_angle = np.clip(np.dot(t_est_norm, t_gt_norm), -1, 1)
    trans_err = np.degrees(np.arccos(np.abs(cos_angle)))
    
    return rot_err, trans_err


def evaluate_sequence(sequence="05", num_pairs=100):
    """Evaluate visual odometry on a KITTI sequence."""
    
    # Paths
    kitti_path = Path("d:/monocular-visual-odometry/dataset")
    image_dir = kitti_path / "sequences" / sequence / "image_0"
    calib_path = kitti_path / "sequences" / sequence / "calibration.json"
    poses_path = kitti_path / "poses" / f"{sequence}.txt"
    
    # Check paths exist
    if not image_dir.exists():
        print(f"Error: Image directory not found: {image_dir}")
        return
    if not calib_path.exists():
        print(f"Error: Calibration file not found: {calib_path}")
        return
    if not poses_path.exists():
        print(f"Error: Poses file not found: {poses_path}")
        return
    
    # Initialize
    vo = VisualOdometry(str(calib_path))
    poses = load_poses(poses_path)
    images = sorted(image_dir.glob("*.png"))
    
    # Limit to available frames
    num_pairs = min(num_pairs, len(images) - 1, len(poses) - 1)
    
    print("=" * 70)
    print(f"KITTI Sequence {sequence} Evaluation")
    print(f"Evaluating {num_pairs} consecutive frame pairs")
    print("=" * 70)
    print()
    
    # Run evaluation
    rot_errors = []
    trans_errors = []
    failures = []
    
    for i in range(num_pairs):
        img1_path = str(images[i])
        img2_path = str(images[i + 1])
        
        # Estimate motion
        result = vo.estimate_motion(img1_path, img2_path)
        
        if not result.success:
            failures.append((i, "estimation failed"))
            continue
        
        # Compute ground truth
        T1, T2 = poses[i], poses[i + 1]
        R1 = T1[:3, :3]
        R_gt = R1.T @ T2[:3, :3]
        delta_world = T2[:3, 3] - T1[:3, 3]
        t_gt = R1.T @ delta_world
        t_gt = t_gt / (np.linalg.norm(t_gt) + 1e-10)
        
        # Compute errors
        rot_err, trans_err = compute_errors(result.R, result.t, R_gt, t_gt)
        rot_errors.append(rot_err)
        trans_errors.append(trans_err)
        
        # Print progress every 20 frames
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_pairs} frames...")
    
    # Calculate statistics
    rot_errors = np.array(rot_errors)
    trans_errors = np.array(trans_errors)
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Frames evaluated: {len(rot_errors)}/{num_pairs}")
    print()
    
    print("Rotation Error:")
    print(f"  Mean:   {np.mean(rot_errors):.3f}°")
    print(f"  Median: {np.median(rot_errors):.3f}°")
    print(f"  Std:    {np.std(rot_errors):.3f}°")
    print(f"  Min:    {np.min(rot_errors):.3f}°")
    print(f"  Max:    {np.max(rot_errors):.3f}°")
    print()
    
    print("Translation Error:")
    print(f"  Mean:   {np.mean(trans_errors):.3f}°")
    print(f"  Median: {np.median(trans_errors):.3f}°")
    print(f"  Std:    {np.std(trans_errors):.3f}°")
    print(f"  Min:    {np.min(trans_errors):.3f}°")
    print(f"  Max:    {np.max(trans_errors):.3f}°")
    print()
    
    # Success rates
    print("Success Rates:")
    thresholds = [(1, 5), (2, 10), (5, 15), (10, 30)]
    for r_thresh, t_thresh in thresholds:
        success = np.sum((rot_errors < r_thresh) & (trans_errors < t_thresh))
        rate = 100 * success / len(rot_errors)
        print(f"  R<{r_thresh}°, t<{t_thresh}°: {rate:.1f}% ({success}/{len(rot_errors)})")
    
    print()
    
    # Overall correctness
    correct = np.sum((rot_errors < 5) & (trans_errors < 15))
    correctness = 100 * correct / len(rot_errors)
    
    print("=" * 70)
    print(f"CORRECTNESS: {correctness:.1f}% ({correct}/{len(rot_errors)} frames correct)")
    print("=" * 70)
    
    if failures:
        print()
        print(f"Failures ({len(failures)}):")
        for frame, reason in failures:
            print(f"  Frame {frame}: {reason}")


if __name__ == "__main__":
    evaluate_sequence(sequence="05", num_pairs=100)
