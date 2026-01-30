#!/usr/bin/env python3
"""Diverse KITTI evaluation - samples from multiple sequences."""

import os
import sys
import numpy as np
import json

# Suppress all output from the estimator
import io
from contextlib import redirect_stdout, redirect_stderr

from camera_motion_estimator import CameraMotionEstimator

def load_kitti_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            T = np.eye(4)
            T[:3, :] = np.array(values).reshape(3, 4)
            poses.append(T)
    return poses

def compute_relative_pose(T1, T2):
    T_rel = np.linalg.inv(T1) @ T2
    return T_rel[:3, :3], T_rel[:3, 3]

def rotation_error(R_est, R_gt):
    R_diff = R_est.T @ R_gt
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    return np.degrees(np.arccos((trace - 1) / 2))

def translation_error(t_est, t_gt):
    t_est_n = t_est.flatten() / (np.linalg.norm(t_est) + 1e-10)
    t_gt_n = t_gt.flatten() / (np.linalg.norm(t_gt) + 1e-10)
    cos_angle = np.clip(np.dot(t_est_n, t_gt_n), -1.0, 1.0)
    return np.degrees(np.arccos(np.abs(cos_angle)))

def get_calibration(seq_path):
    """Create or load calibration JSON for sequence."""
    calib_json = os.path.join(seq_path, 'calibration.json')
    if os.path.exists(calib_json):
        return calib_json
    
    # Parse KITTI calib.txt
    calib_txt = os.path.join(seq_path, 'calib.txt')
    K = None
    with open(calib_txt, 'r') as f:
        for line in f:
            if line.startswith('P0:'):
                values = [float(x) for x in line.split()[1:]]
                P = np.array(values).reshape(3, 4)
                K = P[:3, :3]
                break
    
    if K is None:
        raise ValueError(f"Could not find P0 in {calib_txt}")
    
    calib_data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": [0, 0, 0, 0, 0],
        "image_width": 1241,
        "image_height": 376
    }
    with open(calib_json, 'w') as f:
        json.dump(calib_data, f, indent=2)
    return calib_json

def main():
    kitti_path = 'kitti_odometry'
    
    # Sequences 00-10 have ground truth poses
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    pairs_per_seq = 5  # Sample 5 pairs from each sequence (faster)
    
    print(f"KITTI Multi-Sequence Evaluation")
    print(f"================================")
    print(f"Testing {pairs_per_seq} pairs from each available sequence...")
    print()
    
    all_rot_errors = []
    all_trans_errors = []
    seq_results = {}
    
    for seq in sequences:
        seq_path = os.path.join(kitti_path, 'sequences', seq)
        image_dir = os.path.join(seq_path, 'image_0')
        poses_file = os.path.join(kitti_path, 'poses', f'{seq}.txt')
        
        # Check if sequence exists
        if not os.path.exists(image_dir) or not os.path.exists(poses_file):
            continue
        
        images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if len(images) < pairs_per_seq + 1:
            continue
            
        poses = load_kitti_poses(poses_file)
        calib_json = get_calibration(seq_path)
        
        # Initialize estimator (suppress output)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            estimator = CameraMotionEstimator(calib_json, preserve_aspect_ratio=True)
        
        # Sample pairs spread across the sequence
        n_images = len(images)
        step = max(1, (n_images - 1) // pairs_per_seq)
        indices = list(range(0, min(n_images - 1, step * pairs_per_seq), step))[:pairs_per_seq]
        
        rot_errors, trans_errors = [], []
        
        for idx in indices:
            img1 = os.path.join(image_dir, images[idx])
            img2 = os.path.join(image_dir, images[idx + 1])
            
            # Suppress verbose output
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = estimator.estimate(img1, img2)
            
            if result.R is not None:
                R_gt, t_gt = compute_relative_pose(poses[idx], poses[idx + 1])
                rot_errors.append(rotation_error(result.R, R_gt))
                trans_errors.append(translation_error(result.t, t_gt))
        
        if rot_errors:
            all_rot_errors.extend(rot_errors)
            all_trans_errors.extend(trans_errors)
            
            r5 = 100 * sum(1 for e in rot_errors if e < 5) / len(rot_errors)
            combined = 100 * sum(1 for r, t in zip(rot_errors, trans_errors) if r < 5 and t < 15) / len(rot_errors)
            seq_results[seq] = {'pairs': len(rot_errors), 'r5': r5, 'combined': combined}
            print(f"  Seq {seq}: {len(rot_errors):2d} pairs | R<5°: {r5:5.1f}% | Combined: {combined:5.1f}%")
    
    # Final summary
    if all_rot_errors:
        n = len(all_rot_errors)
        print()
        print(f"{'='*50}")
        print(f"TOTAL: {n} pairs across {len(seq_results)} sequences")
        print(f"{'='*50}")
        print(f"Mean Rotation Error:    {np.mean(all_rot_errors):6.2f}°")
        print(f"Mean Translation Error: {np.mean(all_trans_errors):6.2f}°")
        print()
        print(f"R < 1°:  {100*sum(1 for e in all_rot_errors if e<1)/n:5.1f}%")
        print(f"R < 5°:  {100*sum(1 for e in all_rot_errors if e<5)/n:5.1f}%")
        print(f"t < 15°: {100*sum(1 for e in all_trans_errors if e<15)/n:5.1f}%")
        print(f"Combined (R<5°, t<15°): {100*sum(1 for r,t in zip(all_rot_errors,all_trans_errors) if r<5 and t<15)/n:5.1f}%")
    else:
        print("No sequences found with ground truth!")

if __name__ == '__main__':
    main()
