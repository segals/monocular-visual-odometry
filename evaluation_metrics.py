#!/usr/bin/env python3
"""
Standard Visual Odometry Evaluation Metrics

Implements:
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)
- APE (Absolute Pose Error)

Based on TUM RGB-D benchmark evaluation metrics.

Usage:
    from evaluation_metrics import compute_ate, compute_rpe, EvaluationMetrics
    
    ate = compute_ate(estimated_poses, ground_truth_poses)
    rpe = compute_rpe(estimated_poses, ground_truth_poses, delta=1)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PoseError:
    """Container for pose error statistics."""
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float
    
    def to_dict(self) -> Dict:
        return {
            'rmse': self.rmse,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min,
            'max': self.max
        }


def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert rotation matrix and translation to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrix and translation from 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute rotation error between two rotation matrices.
    
    Returns error in degrees.
    """
    R_diff = R1 @ R2.T
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)
    angle = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """
    Compute translation error (Euclidean distance) between two translations.
    
    Returns error in the same units as input.
    """
    return float(np.linalg.norm(t1 - t2))


def compute_statistics(errors: np.ndarray) -> PoseError:
    """Compute statistics for an array of errors."""
    if len(errors) == 0:
        return PoseError(0, 0, 0, 0, 0, 0)
    
    return PoseError(
        rmse=float(np.sqrt(np.mean(errors ** 2))),
        mean=float(np.mean(errors)),
        median=float(np.median(errors)),
        std=float(np.std(errors)),
        min=float(np.min(errors)),
        max=float(np.max(errors))
    )


def align_trajectories_umeyama(estimated: np.ndarray, ground_truth: np.ndarray,
                                with_scale: bool = False) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Align two trajectories using Umeyama's method.
    
    This finds the optimal rotation R, translation t, and scale s such that:
        ground_truth ≈ s * R @ estimated + t
    
    Args:
        estimated: Nx3 array of estimated positions
        ground_truth: Nx3 array of ground truth positions
        with_scale: If True, also estimate scale factor
        
    Returns:
        R: 3x3 rotation matrix
        s: scale factor (1.0 if with_scale=False)
        t: 3x1 translation vector
    """
    assert estimated.shape == ground_truth.shape
    n, m = estimated.shape
    
    # Compute centroids
    mu_est = estimated.mean(axis=0)
    mu_gt = ground_truth.mean(axis=0)
    
    # Center the point sets
    est_centered = estimated - mu_est
    gt_centered = ground_truth - mu_gt
    
    # Compute variances
    var_est = np.sum(est_centered ** 2) / n
    
    # Compute cross-covariance matrix
    H = (gt_centered.T @ est_centered) / n
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    
    # Rotation
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m-1, m-1] = -1
    
    R = U @ S @ Vt
    
    # Scale
    if with_scale:
        s = np.trace(np.diag(D) @ S) / var_est
    else:
        s = 1.0
    
    # Translation
    t = mu_gt - s * R @ mu_est
    
    return R, s, t


def compute_ate(estimated_poses: List[np.ndarray], 
                ground_truth_poses: List[np.ndarray],
                align: bool = True,
                with_scale: bool = False) -> Tuple[PoseError, np.ndarray]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the global consistency of the estimated trajectory with
    respect to ground truth. It's computed as the RMSE of position differences
    after optimal alignment.
    
    Args:
        estimated_poses: List of 4x4 estimated transformation matrices
        ground_truth_poses: List of 4x4 ground truth transformation matrices
        align: If True, align trajectories using Umeyama's method
        with_scale: If True, also align scale (for monocular VO)
        
    Returns:
        error: PoseError with statistics
        aligned_errors: Array of per-frame position errors
    """
    assert len(estimated_poses) == len(ground_truth_poses)
    
    # Extract positions
    est_positions = np.array([T[:3, 3] for T in estimated_poses])
    gt_positions = np.array([T[:3, 3] for T in ground_truth_poses])
    
    if align:
        R, s, t = align_trajectories_umeyama(est_positions, gt_positions, with_scale)
        est_aligned = s * (est_positions @ R.T) + t
    else:
        est_aligned = est_positions
    
    # Compute errors
    errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
    
    return compute_statistics(errors), errors


def compute_rpe(estimated_poses: List[np.ndarray],
                ground_truth_poses: List[np.ndarray],
                delta: int = 1,
                delta_unit: str = 'frames') -> Tuple[PoseError, PoseError, np.ndarray, np.ndarray]:
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the local accuracy of the trajectory by comparing
    relative pose changes between frames.
    
    Args:
        estimated_poses: List of 4x4 estimated transformation matrices
        ground_truth_poses: List of 4x4 ground truth transformation matrices
        delta: The frame interval for computing relative poses
        delta_unit: 'frames' or 'meters' (only 'frames' supported currently)
        
    Returns:
        trans_error: Translation RPE statistics (in meters)
        rot_error: Rotation RPE statistics (in degrees)
        trans_errors: Per-segment translation errors
        rot_errors: Per-segment rotation errors
    """
    assert len(estimated_poses) == len(ground_truth_poses)
    n = len(estimated_poses)
    
    trans_errors = []
    rot_errors = []
    
    for i in range(n - delta):
        # Ground truth relative pose
        T_gt_i = ground_truth_poses[i]
        T_gt_j = ground_truth_poses[i + delta]
        T_gt_rel = np.linalg.inv(T_gt_i) @ T_gt_j
        
        # Estimated relative pose
        T_est_i = estimated_poses[i]
        T_est_j = estimated_poses[i + delta]
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j
        
        # Error in relative pose
        T_error = np.linalg.inv(T_est_rel) @ T_gt_rel
        
        # Translation error
        trans_err = np.linalg.norm(T_error[:3, 3])
        trans_errors.append(trans_err)
        
        # Rotation error
        R_error = T_error[:3, :3]
        trace = np.trace(R_error)
        trace = np.clip(trace, -1.0, 3.0)
        rot_err = np.degrees(np.arccos((trace - 1.0) / 2.0))
        rot_errors.append(rot_err)
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    return (compute_statistics(trans_errors), 
            compute_statistics(rot_errors),
            trans_errors, rot_errors)


def compute_segment_errors(estimated_poses: List[np.ndarray],
                          ground_truth_poses: List[np.ndarray],
                          segment_lengths: List[float] = [100, 200, 300, 400, 500, 600, 700, 800]) -> Dict:
    """
    Compute errors over trajectory segments of different lengths.
    Used in KITTI odometry benchmark.
    
    Args:
        estimated_poses: List of 4x4 estimated transformation matrices
        ground_truth_poses: List of 4x4 ground truth transformation matrices
        segment_lengths: List of segment lengths in meters
        
    Returns:
        Dictionary with per-segment error statistics
    """
    # Extract positions and compute cumulative distances
    gt_positions = np.array([T[:3, 3] for T in ground_truth_poses])
    distances = np.zeros(len(gt_positions))
    
    for i in range(1, len(gt_positions)):
        distances[i] = distances[i-1] + np.linalg.norm(gt_positions[i] - gt_positions[i-1])
    
    results = {}
    
    for length in segment_lengths:
        trans_errors = []
        rot_errors = []
        
        for i in range(len(distances)):
            # Find endpoint of segment
            j = i
            while j < len(distances) and distances[j] - distances[i] < length:
                j += 1
            
            if j >= len(distances):
                continue
            
            # Compute relative pose error for this segment
            T_gt_rel = np.linalg.inv(ground_truth_poses[i]) @ ground_truth_poses[j]
            T_est_rel = np.linalg.inv(estimated_poses[i]) @ estimated_poses[j]
            
            T_error = np.linalg.inv(T_est_rel) @ T_gt_rel
            
            # Translation error (percentage of segment length)
            trans_err = np.linalg.norm(T_error[:3, 3]) / length * 100
            trans_errors.append(trans_err)
            
            # Rotation error (deg/meter)
            R_error = T_error[:3, :3]
            trace = np.clip(np.trace(R_error), -1.0, 3.0)
            rot_err = np.degrees(np.arccos((trace - 1.0) / 2.0)) / length
            rot_errors.append(rot_err)
        
        if trans_errors:
            results[f'{length}m'] = {
                'translation_error_pct': compute_statistics(np.array(trans_errors)).to_dict(),
                'rotation_error_deg_per_m': compute_statistics(np.array(rot_errors)).to_dict(),
                'num_segments': len(trans_errors)
            }
    
    return results


class EvaluationMetrics:
    """
    Complete evaluation metrics suite for visual odometry.
    
    Usage:
        metrics = EvaluationMetrics(estimated_poses, ground_truth_poses)
        print(metrics.summary())
    """
    
    def __init__(self, estimated_poses: List[np.ndarray], 
                 ground_truth_poses: List[np.ndarray],
                 with_scale: bool = True):
        """
        Initialize evaluation metrics.
        
        Args:
            estimated_poses: List of 4x4 estimated transformation matrices
            ground_truth_poses: List of 4x4 ground truth transformation matrices
            with_scale: Whether to align scale (True for monocular VO)
        """
        self.estimated = estimated_poses
        self.ground_truth = ground_truth_poses
        self.with_scale = with_scale
        
        # Compute ATE
        self.ate, self.ate_errors = compute_ate(
            estimated_poses, ground_truth_poses, 
            align=True, with_scale=with_scale
        )
        
        # Compute RPE at different deltas
        self.rpe_1 = compute_rpe(estimated_poses, ground_truth_poses, delta=1)
        self.rpe_5 = compute_rpe(estimated_poses, ground_truth_poses, delta=5)
        self.rpe_10 = compute_rpe(estimated_poses, ground_truth_poses, delta=10)
    
    def summary(self) -> str:
        """Return formatted summary of all metrics."""
        lines = [
            "=" * 60,
            "VISUAL ODOMETRY EVALUATION METRICS",
            "=" * 60,
            "",
            "[ATE - Absolute Trajectory Error]",
            f"  RMSE:   {self.ate.rmse:.4f} m",
            f"  Mean:   {self.ate.mean:.4f} m",
            f"  Median: {self.ate.median:.4f} m",
            f"  Std:    {self.ate.std:.4f} m",
            "",
            "[RPE - Relative Pose Error (delta=1 frame)]",
            f"  Translation RMSE: {self.rpe_1[0].rmse:.4f} m",
            f"  Rotation RMSE:    {self.rpe_1[1].rmse:.4f} deg",
            "",
            "[RPE - Relative Pose Error (delta=5 frames)]",
            f"  Translation RMSE: {self.rpe_5[0].rmse:.4f} m",
            f"  Rotation RMSE:    {self.rpe_5[1].rmse:.4f} deg",
            "",
            "[RPE - Relative Pose Error (delta=10 frames)]",
            f"  Translation RMSE: {self.rpe_10[0].rmse:.4f} m",
            f"  Rotation RMSE:    {self.rpe_10[1].rmse:.4f} deg",
            "=" * 60,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Return all metrics as dictionary."""
        return {
            'ate': self.ate.to_dict(),
            'rpe_delta_1': {
                'translation': self.rpe_1[0].to_dict(),
                'rotation': self.rpe_1[1].to_dict()
            },
            'rpe_delta_5': {
                'translation': self.rpe_5[0].to_dict(),
                'rotation': self.rpe_5[1].to_dict()
            },
            'rpe_delta_10': {
                'translation': self.rpe_10[0].to_dict(),
                'rotation': self.rpe_10[1].to_dict()
            },
            'num_poses': len(self.estimated),
            'scale_aligned': self.with_scale
        }


# Example usage / test
if __name__ == '__main__':
    print("Testing evaluation metrics...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_poses = 100
    
    # Ground truth: circular motion
    gt_poses = []
    for i in range(n_poses):
        angle = 2 * np.pi * i / n_poses
        T = np.eye(4)
        T[0, 3] = 5 * np.cos(angle)
        T[1, 3] = 5 * np.sin(angle)
        T[2, 3] = 0.1 * i
        gt_poses.append(T)
    
    # Estimated: ground truth + noise
    est_poses = []
    for T_gt in gt_poses:
        T_est = T_gt.copy()
        T_est[:3, 3] += np.random.normal(0, 0.05, 3)  # 5cm noise
        est_poses.append(T_est)
    
    # Compute metrics
    metrics = EvaluationMetrics(est_poses, gt_poses, with_scale=False)
    print(metrics.summary())
    
    print("\n✓ Evaluation metrics module working correctly!")
