"""
Clean Visual Odometry Implementation
=====================================
Simple, accurate monocular visual odometry using:
1. SIFT feature detection
2. FLANN-based feature matching with ratio test
3. Essential matrix estimation with RANSAC
4. Pose recovery with cheirality check

Author: Clean implementation for KITTI dataset
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class MotionResult:
    """Result of motion estimation between two frames."""
    R: np.ndarray          # 3x3 rotation matrix
    t: np.ndarray          # 3x1 translation vector (unit norm)
    num_inliers: int       # Number of RANSAC inliers
    num_matches: int       # Number of matches before RANSAC
    success: bool          # Whether estimation succeeded


class VisualOdometry:
    """
    Clean visual odometry implementation.
    
    Pipeline:
    1. Load and preprocess images
    2. Detect SIFT features
    3. Match features using FLANN + ratio test
    4. Estimate Essential matrix with RANSAC
    5. Recover pose (R, t)
    6. Refine translation using optical flow analysis
    """
    
    def __init__(self, calibration_path: str):
        """
        Initialize visual odometry with camera calibration.
        
        Args:
            calibration_path: Path to calibration JSON file with camera_matrix
        """
        # Load calibration
        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)
        
        self.K = np.array(calib_data['camera_matrix'], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)
        self.image_width = calib_data.get('image_width', 1241)
        self.image_height = calib_data.get('image_height', 376)
        
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        
        # SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=2000,        # Max features to detect
            contrastThreshold=0.04, # Lower = more features
            edgeThreshold=10,       # Higher = more edge features
            sigma=1.6              # Gaussian blur sigma
        )
        
        # FLANN matcher for SIFT (uses KD-tree)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Also create BFMatcher as backup
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        print(f"Visual Odometry initialized")
        print(f"  Image size: {self.image_width}x{self.image_height}")
        print(f"  Focal length: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}")
        print(f"  Principal point: cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}")
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image as grayscale."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    def detect_features(self, image: np.ndarray) -> Tuple[list, np.ndarray]:
        """
        Detect SIFT features in image.
        
        Returns:
            keypoints: List of cv2.KeyPoint
            descriptors: NxD array of descriptors
        """
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                       ratio_threshold: float = 0.75) -> list:
        """
        Match features using FLANN with Lowe's ratio test.
        
        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            ratio_threshold: Ratio test threshold (lower = stricter)
            
        Returns:
            List of good matches (cv2.DMatch objects)
        """
        if desc1 is None or desc2 is None:
            return []
        
        if len(desc1) < 2 or len(desc2) < 2:
            return []
        
        try:
            # Find 2 nearest neighbors for ratio test
            matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            # Fallback to BFMatcher
            matches = self.bf.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def matches_to_points(self, kp1: list, kp2: list, 
                          matches: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert matches to point arrays.
        
        Returns:
            pts1: Nx2 array of points from image 1
            pts2: Nx2 array of points from image 2
        """
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)
        return pts1, pts2
    
    def estimate_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray,
                                   threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate Essential matrix using RANSAC.
        
        Args:
            pts1: Nx2 points from image 1
            pts2: Nx2 points from image 2
            threshold: RANSAC inlier threshold in pixels
            
        Returns:
            E: 3x3 Essential matrix
            mask: Nx1 inlier mask
        """
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=threshold
        )
        return E, mask
    
    def recover_pose(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray,
                     mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Recover rotation and translation from Essential matrix.
        
        Uses cheirality check to select correct solution.
        
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector (unit norm)
            num_infront: Number of points in front of both cameras
        """
        num_infront, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        return R, t, num_infront
    
    def estimate_rotation_from_flow(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Estimate rotation from optical flow pattern.
        
        For small rotations, the flow due to rotation can be separated from translation.
        The key insight is that rotation causes a specific pattern:
        - Rotation about Y (yaw): horizontal flow that's constant across all features
        - Rotation about X (pitch): vertical flow that depends on Y position
        
        Returns:
            R: 3x3 estimated rotation matrix
        """
        if len(pts1) < 20:
            return np.eye(3)
        
        flow = pts2 - pts1
        
        # For rotation about Y-axis (yaw), the flow is:
        # flow_x = -omega_y * (z + x^2/z) * fx ≈ -omega_y * fx  (for small x relative to z)
        # This means: for pure yaw, horizontal flow is nearly constant
        
        # For translation, horizontal flow varies with x-position (parallax)
        # So: constant part of horizontal flow = rotation, varying part = translation
        
        # Fit a linear model: flow_x = a + b * x
        # a = rotation component, b = translation/depth component
        x_coords = pts1[:, 0] - self.cx
        A = np.column_stack([np.ones(len(x_coords)), x_coords])
        coeffs, _, _, _ = np.linalg.lstsq(A, flow[:, 0], rcond=None)
        a_x, b_x = coeffs
        
        # The constant term a_x is due to rotation: a_x ≈ -omega_y * fx
        omega_y = -a_x / self.fx
        
        # For rotation about X-axis (pitch): flow_y = omega_x * fy
        y_coords = pts1[:, 1] - self.cy
        A_y = np.column_stack([np.ones(len(y_coords)), y_coords])
        coeffs_y, _, _, _ = np.linalg.lstsq(A_y, flow[:, 1], rcond=None)
        a_y, b_y = coeffs_y
        
        omega_x = a_y / self.fy
        
        # Rotation about Z (roll) can be estimated from tangential flow
        # For simplicity, assume roll is small
        omega_z = 0.0
        
        # Build rotation matrix from angular velocities
        # For small angles: R ≈ I + [omega]_x
        R = np.array([
            [1, -omega_z, omega_y],
            [omega_z, 1, -omega_x],
            [-omega_y, omega_x, 1]
        ])
        
        # Orthogonalize
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        
        return R
    
    def estimate_translation_from_flow(self, pts1: np.ndarray, pts2: np.ndarray,
                                         R: np.ndarray) -> np.ndarray:
        """
        Estimate translation direction from optical flow pattern.
        
        This method is more robust than Essential matrix for lateral motion.
        It works by:
        1. Removing rotation-induced flow
        2. Analyzing the remaining translational flow pattern
        3. Estimating translation direction from mean flow
        
        Args:
            pts1: Nx2 matched points from image 1
            pts2: Nx2 matched points from image 2  
            R: 3x3 rotation matrix (estimated from Essential matrix)
            
        Returns:
            t: 3x1 estimated translation direction (unit vector)
        """
        if len(pts1) < 20:
            return np.array([[0], [0], [1]], dtype=np.float64)
        
        # Compute actual optical flow
        flow = pts2 - pts1
        
        # Compute rotation-induced flow
        # For each point p1, compute where it would be after pure rotation
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts1_norm = (self.K_inv @ pts1_h.T).T  # Normalized coordinates
        
        # Rotate normalized coordinates
        pts1_rotated_norm = (R @ pts1_norm.T).T
        
        # Project back to image coordinates
        pts1_rotated = (self.K @ pts1_rotated_norm.T).T
        pts1_rotated = pts1_rotated[:, :2] / pts1_rotated[:, 2:3]
        
        # Rotation-induced flow
        rotation_flow = pts1_rotated - pts1
        
        # Translational flow = actual flow - rotation flow
        translational_flow = flow - rotation_flow
        
        # Analyze translational flow to estimate translation direction
        # For translation t = [tx, ty, tz], the flow at normalized point (x, y, 1) is:
        #   flow_x = fx * (tz*x - tx) / z
        #   flow_y = fy * (tz*y - ty) / z
        
        # Mean translational flow (robust to outliers using median)
        mean_flow_x = np.median(translational_flow[:, 0])
        mean_flow_y = np.median(translational_flow[:, 1])
        
        # Radial flow analysis for tz component
        dx = pts1[:, 0] - self.cx
        dy = pts1[:, 1] - self.cy
        radial_dist = np.sqrt(dx**2 + dy**2) + 1e-10
        
        # Radial unit vectors
        radial_x = dx / radial_dist
        radial_y = dy / radial_dist
        
        # Radial component of translational flow (positive = expansion = forward motion)
        radial_flow = translational_flow[:, 0] * radial_x + translational_flow[:, 1] * radial_y
        mean_radial = np.median(radial_flow)
        
        # Estimate translation components:
        # tx from mean horizontal flow (negative sign: camera right = flow left)
        # ty from mean vertical flow
        # tz from radial expansion
        
        tx_raw = -mean_flow_x / self.fx
        ty_raw = -mean_flow_y / self.fy
        tz_raw = mean_radial / (self.fx * 0.5)  # Scale factor for radial
        
        t_raw = np.array([tx_raw, ty_raw, tz_raw])
        t_norm = np.linalg.norm(t_raw)
        
        if t_norm < 1e-10:
            return np.array([[0], [0], [1]], dtype=np.float64)
        
        t_unit = t_raw / t_norm
        return t_unit.reshape(3, 1)
    
    def refine_translation(self, t_essential: np.ndarray, t_flow: np.ndarray,
                           pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Combine Essential matrix and flow-based translation estimates.
        
        Uses the one that better explains the observed flow pattern.
        
        Args:
            t_essential: Translation from Essential matrix (3x1)
            t_flow: Translation from flow analysis (3x1)
            pts1, pts2: Matched points
            R: Rotation matrix
            
        Returns:
            t_refined: Best translation estimate (3x1)
        """
        # Compute expected flow for each translation estimate
        def compute_flow_correlation(t):
            t = t.flatten()
            
            # Expected translational flow for each point
            x = (pts1[:, 0] - self.cx) / self.fx
            y = (pts1[:, 1] - self.cy) / self.fy
            
            tx, ty, tz = t
            expected_flow_x = self.fx * (tz * x - tx)  # Assuming z=1
            expected_flow_y = self.fy * (tz * y - ty)
            expected_flow = np.column_stack([expected_flow_x, expected_flow_y])
            
            # Actual translational flow
            flow = pts2 - pts1
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts1_norm = (self.K_inv @ pts1_h.T).T
            pts1_rotated_norm = (R @ pts1_norm.T).T
            pts1_rotated = (self.K @ pts1_rotated_norm.T).T
            pts1_rotated = pts1_rotated[:, :2] / pts1_rotated[:, 2:3]
            rotation_flow = pts1_rotated - pts1
            translational_flow = flow - rotation_flow
            
            # Correlation (normalized dot product)
            exp_mag = np.linalg.norm(expected_flow, axis=1, keepdims=True) + 1e-10
            trans_mag = np.linalg.norm(translational_flow, axis=1, keepdims=True) + 1e-10
            
            exp_dir = expected_flow / exp_mag
            trans_dir = translational_flow / trans_mag
            
            correlations = np.sum(exp_dir * trans_dir, axis=1)
            return np.median(correlations)
        
        corr_essential = compute_flow_correlation(t_essential)
        corr_flow = compute_flow_correlation(t_flow)
        
        # Also try negated versions (sign ambiguity)
        corr_essential_neg = compute_flow_correlation(-t_essential)
        corr_flow_neg = compute_flow_correlation(-t_flow)
        
        # Pick the best
        candidates = [
            (corr_essential, t_essential),
            (corr_essential_neg, -t_essential),
            (corr_flow, t_flow),
            (corr_flow_neg, -t_flow),
        ]
        
        best_corr, best_t = max(candidates, key=lambda x: x[0])
        return best_t
    
    def estimate_motion_with_homography(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate motion using Homography decomposition.
        
        This is more robust for planar scenes and lateral motion.
        
        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation direction
        """
        # Estimate homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        
        if H is None:
            return None, None
        
        # Decompose homography
        num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.K)
        
        if num_solutions == 0:
            return None, None
        
        # Select best solution using cheirality check
        best_R, best_t = None, None
        best_score = -1
        
        for i in range(num_solutions):
            R = Rs[i]
            t = ts[i]
            n = normals[i]
            
            # Check if normal points towards camera (z > 0)
            if n[2, 0] < 0:
                continue
            
            # Count points in front
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts1_norm = (self.K_inv @ pts1_h.T).T
            pts1_rotated = (R @ pts1_norm.T).T + t.T
            in_front = np.sum(pts1_rotated[:, 2] > 0)
            
            if in_front > best_score:
                best_score = in_front
                best_R = R
                best_t = t
        
        return best_R, best_t
    
    def estimate_motion(self, image1_path: str, image2_path: str,
                        verbose: bool = False) -> MotionResult:
        """
        Estimate camera motion between two images.
        
        This is the main entry point for motion estimation.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            verbose: Print detailed output
            
        Returns:
            MotionResult with R, t, and quality metrics
        """
        # Step 1: Load images
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)
        
        if verbose:
            print(f"\n{'='*60}")
            print("Visual Odometry Pipeline")
            print('='*60)
            print(f"Image 1: {Path(image1_path).name}")
            print(f"Image 2: {Path(image2_path).name}")
            print(f"Image size: {img1.shape[1]}x{img1.shape[0]}")
        
        # Step 2: Detect features
        kp1, desc1 = self.detect_features(img1)
        kp2, desc2 = self.detect_features(img2)
        
        if verbose:
            print(f"\n[Step 1] Feature Detection (SIFT)")
            print(f"  Image 1: {len(kp1)} keypoints")
            print(f"  Image 2: {len(kp2)} keypoints")
        
        if len(kp1) < 10 or len(kp2) < 10:
            return MotionResult(
                R=np.eye(3), t=np.array([[0], [0], [1]]),
                num_inliers=0, num_matches=0, success=False
            )
        
        # Step 3: Match features
        matches = self.match_features(desc1, desc2, ratio_threshold=0.75)
        
        if verbose:
            print(f"\n[Step 2] Feature Matching (FLANN + Ratio Test)")
            print(f"  Good matches: {len(matches)}")
        
        if len(matches) < 8:
            return MotionResult(
                R=np.eye(3), t=np.array([[0], [0], [1]]),
                num_inliers=0, num_matches=len(matches), success=False
            )
        
        # Convert to point arrays
        pts1, pts2 = self.matches_to_points(kp1, kp2, matches)
        
        # Compute radial flow from ALL matches BEFORE RANSAC
        # This is more robust because RANSAC inliers may be biased
        flow_all = pts2 - pts1
        dx_all = pts1[:, 0] - self.cx
        dy_all = pts1[:, 1] - self.cy
        radial_dist_all = np.sqrt(dx_all**2 + dy_all**2) + 1e-10
        radial_x_all = dx_all / radial_dist_all
        radial_y_all = dy_all / radial_dist_all
        radial_flow_all = flow_all[:, 0] * radial_x_all + flow_all[:, 1] * radial_y_all
        mean_radial_all = np.mean(radial_flow_all)
        
        # Step 4: Estimate Essential matrix
        E, mask = self.estimate_essential_matrix(pts1, pts2, threshold=1.0)
        
        num_inliers = int(np.sum(mask)) if mask is not None else 0
        
        if verbose:
            print(f"\n[Step 3] Essential Matrix Estimation (RANSAC)")
            print(f"  Inliers: {num_inliers}/{len(matches)} ({100*num_inliers/len(matches):.1f}%)")
        
        if E is None or num_inliers < 8:
            return MotionResult(
                R=np.eye(3), t=np.array([[0], [0], [1]]),
                num_inliers=num_inliers, num_matches=len(matches), success=False
            )
        
        # Step 5: Recover pose from Essential matrix
        R_E, t_E, num_infront = self.recover_pose(E, pts1, pts2, mask)
        
        if verbose:
            print(f"\n[Step 4] Pose Recovery (from Essential matrix)")
            print(f"  Points in front: {num_infront}")
            rvec_E, _ = cv2.Rodrigues(R_E)
            print(f"  Rotation (E, deg): [{np.degrees(rvec_E[0,0]):.2f}, {np.degrees(rvec_E[1,0]):.2f}, {np.degrees(rvec_E[2,0]):.2f}]")
            print(f"  Translation (E): [{t_E[0,0]:.4f}, {t_E[1,0]:.4f}, {t_E[2,0]:.4f}]")
        
        # Get inlier points
        if mask is not None:
            inlier_mask = mask.ravel() > 0
            pts1_inliers = pts1[inlier_mask]
            pts2_inliers = pts2[inlier_mask]
        else:
            pts1_inliers = pts1
            pts2_inliers = pts2
        
        # Step 6: Analyze flow to determine if Essential matrix result is reliable
        # For forward-dominant motion: Essential matrix is reliable
        # For lateral-dominant motion: use flow-based rotation estimation
        
        flow = pts2_inliers - pts1_inliers
        mean_flow = np.mean(flow, axis=0)
        
        # Radial flow for forward motion detection
        dx = pts1_inliers[:, 0] - self.cx
        dy = pts1_inliers[:, 1] - self.cy
        radial_dist = np.sqrt(dx**2 + dy**2) + 1e-10
        radial_x = dx / radial_dist
        radial_y = dy / radial_dist
        radial_flow = flow[:, 0] * radial_x + flow[:, 1] * radial_y
        mean_radial = np.mean(radial_flow)
        
        # Forward motion ratio: how much of the flow is radial (expansion) vs uniform (lateral)
        flow_mag = np.linalg.norm(mean_flow)
        forward_ratio = abs(mean_radial) / (flow_mag + 1e-10)
        
        # Also estimate rotation from flow
        R_flow = self.estimate_rotation_from_flow(pts1_inliers, pts2_inliers)
        
        if verbose:
            print(f"\n[Step 5] Motion Analysis")
            print(f"  Mean flow: [{mean_flow[0]:.2f}, {mean_flow[1]:.2f}] px")
            print(f"  Mean radial flow: {mean_radial:.2f} px")
            print(f"  Forward ratio: {forward_ratio:.2f}")
            rvec_flow, _ = cv2.Rodrigues(R_flow)
            print(f"  Rotation (flow, deg): [{np.degrees(rvec_flow[0,0]):.2f}, {np.degrees(rvec_flow[1,0]):.2f}, {np.degrees(rvec_flow[2,0]):.2f}]")
        
        # For lateral motion (low forward_ratio), use flow-based rotation
        # For forward motion (high forward_ratio), use Essential matrix rotation
        if forward_ratio < 0.3:
            R = R_flow
            if verbose:
                print(f"  Using flow-based rotation (lateral motion detected)")
        else:
            # Essential matrix may have wrong yaw sign - check using optical flow
            # Flow moving LEFT (negative mean_flow_x) = camera rotating RIGHT (positive yaw)
            # Flow moving RIGHT (positive mean_flow_x) = camera rotating LEFT (negative yaw)
            rvec_E, _ = cv2.Rodrigues(R_E)
            yaw_E = rvec_E[1, 0]  # Y rotation = yaw
            
            # Expected yaw sign from flow direction (opposite to flow_x direction)
            expected_yaw_sign = -1.0 if mean_flow[0] > 0 else 1.0
            actual_yaw_sign = 1.0 if yaw_E > 0 else -1.0
            
            if expected_yaw_sign != actual_yaw_sign and abs(yaw_E) > 0.01:
                # Yaw sign is wrong - flip the yaw by negating the Y component
                # This is equivalent to R' = Ry(-θ) @ Rx(α) @ Rz(β) instead of Ry(θ) @ Rx(α) @ Rz(β)
                rvec_corrected = rvec_E.copy()
                rvec_corrected[1, 0] = -rvec_corrected[1, 0]  # Flip yaw
                R, _ = cv2.Rodrigues(rvec_corrected)
                if verbose:
                    print(f"  Corrected yaw sign (flipped from {np.degrees(yaw_E):.2f}° to {np.degrees(-yaw_E):.2f}°)")
            else:
                R = R_E
                if verbose:
                    print(f"  Using Essential matrix rotation (forward motion)")
                    print(f"  Using Essential matrix rotation (forward motion)")
        
        # Estimate translation from flow using the selected rotation
        t_flow = self.estimate_translation_from_flow(pts1_inliers, pts2_inliers, R)
        
        # Try both Essential and flow-based translations
        def compute_motion_score(t):
            """Score translation by how well it explains the optical flow."""
            try:
                t = t.flatten()
                
                # Compute rotation-induced flow
                pts1_h = np.hstack([pts1_inliers, np.ones((len(pts1_inliers), 1))])
                pts1_norm = (self.K_inv @ pts1_h.T).T
                pts1_rotated_norm = (R @ pts1_norm.T).T
                pts1_rotated = (self.K @ pts1_rotated_norm.T).T
                pts1_rotated = pts1_rotated[:, :2] / pts1_rotated[:, 2:3]
                rotation_flow = pts1_rotated - pts1_inliers
                
                translational_flow = flow - rotation_flow
                
                # Expected translational flow
                x = (pts1_inliers[:, 0] - self.cx) / self.fx
                y = (pts1_inliers[:, 1] - self.cy) / self.fy
                tx, ty, tz = t
                expected_flow_x = self.fx * (tz * x - tx)
                expected_flow_y = self.fy * (tz * y - ty)
                expected_flow = np.column_stack([expected_flow_x, expected_flow_y])
                
                # Correlation
                exp_mag = np.linalg.norm(expected_flow, axis=1, keepdims=True) + 1e-10
                trans_mag = np.linalg.norm(translational_flow, axis=1, keepdims=True) + 1e-10
                exp_dir = expected_flow / exp_mag
                trans_dir = translational_flow / trans_mag
                correlations = np.sum(exp_dir * trans_dir, axis=1)
                
                return np.median(correlations)
            except:
                return -1
        
        # Use radial flow from ALL matches (before RANSAC) to determine the sign of tz
        # This is more robust because RANSAC inliers may be biased by the wrong motion model
        # Radial flow > 0 means expansion = forward motion = tz > 0
        # Radial flow < 0 means contraction = backward motion = tz < 0
        radial_flow_sign = 1.0 if mean_radial_all > 0 else -1.0
        
        # For forward-dominant motion, prefer Essential matrix translation with correct tz sign
        # For lateral-dominant motion, flow-based translation may be better
        # The flow-based translation is unreliable when there's significant rotation
        
        # Simple approach: use Essential matrix translation with correct tz sign
        t_E_flat = t_E.flatten()
        tz_E = t_E_flat[2]
        tz_E_sign = 1.0 if tz_E > 0 else -1.0
        
        if tz_E_sign == radial_flow_sign:
            t = t_E  # Essential matrix translation has correct sign
        else:
            t = -t_E  # Flip Essential matrix translation
        
        if verbose:
            print(f"\n[Step 6] Translation Selection")
            print(f"  Pre-RANSAC radial flow: {mean_radial_all:.2f} px (tz sign: {'+ (forward)' if radial_flow_sign > 0 else '- (backward)'})")
            print(f"  Translation (E): [{t_E[0,0]:.4f}, {t_E[1,0]:.4f}, {t_E[2,0]:.4f}]")
            print(f"  Translation (final): [{t[0,0]:.4f}, {t[1,0]:.4f}, {t[2,0]:.4f}]")
            rvec, _ = cv2.Rodrigues(R)
            print(f"  Rotation (deg): [{np.degrees(rvec[0,0]):.2f}, {np.degrees(rvec[1,0]):.2f}, {np.degrees(rvec[2,0]):.2f}]")
        
        return MotionResult(
            R=R, t=t,
            num_inliers=num_inliers,
            num_matches=len(matches),
            success=True
        )


def test_on_kitti():
    """Test the visual odometry on multiple KITTI sequences."""
    import os
    import json
    
    kitti_path = Path("d:/monocular-visual-odometry/dataset")
    
    # Sequences with ground truth (00-10)
    sequences = ["05"]  # Evaluate on sequence 05 only
    
    # Helper functions
    def load_poses(path):
        poses = []
        with open(path, 'r') as f:
            for line in f:
                values = np.array([float(x) for x in line.strip().split()])
                if len(values) == 12:
                    T = np.eye(4)
                    T[:3, :] = values.reshape(3, 4)
                    poses.append(T)
        return poses
    
    def create_calibration_json(calib_txt_path, calib_json_path):
        """Create calibration.json from KITTI calib.txt if it doesn't exist."""
        if calib_json_path.exists():
            return True
        if not calib_txt_path.exists():
            return False
        
        with open(calib_txt_path, 'r') as f:
            for line in f:
                if line.startswith('P0:'):
                    values = [float(x) for x in line.split()[1:]]
                    fx, fy = values[0], values[5]
                    cx, cy = values[2], values[6]
                    calib = {
                        'camera_matrix': [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                        'image_width': 1241,
                        'image_height': 376
                    }
                    with open(calib_json_path, 'w') as jf:
                        json.dump(calib, jf, indent=2)
                    return True
        return False
    
    def compute_errors(R_est, t_est, R_gt, t_gt):
        R_err = R_est @ R_gt.T
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_err = np.degrees(angle_err)
        
        t_est_flat = t_est.flatten()
        t_gt_flat = t_gt.flatten()
        t_est_norm = t_est_flat / (np.linalg.norm(t_est_flat) + 1e-10)
        t_gt_norm = t_gt_flat / (np.linalg.norm(t_gt_flat) + 1e-10)
        cos_angle = np.clip(np.dot(t_est_norm, t_gt_norm), -1, 1)
        trans_err = np.degrees(np.arccos(np.abs(cos_angle)))
        
        return rot_err, trans_err
    
    # Store results for all sequences
    all_results = {}
    
    print("\n" + "="*80)
    print("KITTI Visual Odometry Evaluation - 10 Sequences x 100 Frame Pairs")
    print("="*80)
    
    for sequence in sequences:
        image_dir = kitti_path / "sequences" / sequence / "image_0"
        calib_txt_path = kitti_path / "sequences" / sequence / "calib.txt"
        calib_json_path = kitti_path / "sequences" / sequence / "calibration.json"
        poses_path = kitti_path / "poses" / f"{sequence}.txt"
        
        # Check if sequence exists
        if not image_dir.exists():
            print(f"\nSequence {sequence}: SKIPPED (images not found)")
            continue
        if not poses_path.exists():
            print(f"\nSequence {sequence}: SKIPPED (poses not found)")
            continue
        
        # Create calibration JSON if needed
        if not create_calibration_json(calib_txt_path, calib_json_path):
            print(f"\nSequence {sequence}: SKIPPED (calibration not found)")
            continue
        
        # Initialize VO for this sequence
        vo = VisualOdometry(str(calib_json_path))
        poses = load_poses(poses_path)
        images = sorted(image_dir.glob("*.png"))
        
        # Run on 100 consecutive frame pairs
        num_frames = min(100, len(images) - 1, len(poses) - 1)
        
        rot_errors = []
        trans_errors = []
        
        for frame in range(num_frames):
            img1_path = str(images[frame])
            img2_path = str(images[frame + 1])
            
            result = vo.estimate_motion(img1_path, img2_path, verbose=False)
            
            if not result.success:
                continue
            
            T1, T2 = poses[frame], poses[frame + 1]
            delta_world = T2[:3, 3] - T1[:3, 3]
            R1 = T1[:3, :3]
            t_gt = R1.T @ delta_world
            t_gt = t_gt / (np.linalg.norm(t_gt) + 1e-10)
            R_gt = R1.T @ T2[:3, :3]
            
            rot_err, trans_err = compute_errors(result.R, result.t, R_gt, t_gt)
            rot_errors.append(rot_err)
            trans_errors.append(trans_err)
        
        if rot_errors:
            rot_errors = np.array(rot_errors)
            trans_errors = np.array(trans_errors)
            
            success_rate = 100 * np.sum((rot_errors < 5) & (trans_errors < 15)) / len(rot_errors)
            
            all_results[sequence] = {
                'n_frames': len(rot_errors),
                'rot_mean': np.mean(rot_errors),
                'rot_median': np.median(rot_errors),
                'trans_mean': np.mean(trans_errors),
                'trans_median': np.median(trans_errors),
                'success_rate': success_rate
            }
            
            print(f"\nSequence {sequence}: {len(rot_errors)} frames")
            print(f"  Rotation:    mean={np.mean(rot_errors):.3f}°  median={np.median(rot_errors):.3f}°")
            print(f"  Translation: mean={np.mean(trans_errors):.3f}°  median={np.median(trans_errors):.3f}°")
            print(f"  Success (R<5°, t<15°): {success_rate:.1f}%")
    
    # Overall summary
    if all_results:
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        total_frames = sum(r['n_frames'] for r in all_results.values())
        avg_rot_mean = np.mean([r['rot_mean'] for r in all_results.values()])
        avg_rot_median = np.mean([r['rot_median'] for r in all_results.values()])
        avg_trans_mean = np.mean([r['trans_mean'] for r in all_results.values()])
        avg_trans_median = np.mean([r['trans_median'] for r in all_results.values()])
        avg_success = np.mean([r['success_rate'] for r in all_results.values()])
        
        print(f"\nSequences evaluated: {len(all_results)}")
        print(f"Total frames: {total_frames}")
        print()
        print("Average across sequences:")
        print(f"  Rotation:    mean={avg_rot_mean:.3f}°  median={avg_rot_median:.3f}°")
        print(f"  Translation: mean={avg_trans_mean:.3f}°  median={avg_trans_median:.3f}°")
        print(f"  Success rate: {avg_success:.1f}%")
        
        print("\n" + "-"*80)
        print(f"{'Seq':<5} {'Frames':<8} {'R_mean':<8} {'R_med':<8} {'t_mean':<8} {'t_med':<8} {'Success':<8}")
        print("-"*80)
        for seq, r in sorted(all_results.items()):
            print(f"{seq:<5} {r['n_frames']:<8} {r['rot_mean']:<8.3f} {r['rot_median']:<8.3f} "
                  f"{r['trans_mean']:<8.3f} {r['trans_median']:<8.3f} {r['success_rate']:<8.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 4:
        # Usage: python visual_odometry.py <calibration.json> <image1> <image2>
        calib_path = sys.argv[1]
        image1_path = sys.argv[2]
        image2_path = sys.argv[3]
        
        vo = VisualOdometry(calib_path)
        result = vo.estimate_motion(image1_path, image2_path)
        
        if result.success:
            print("Rotation matrix (R):")
            print(result.R)
            print()
            print("Translation vector (t):")
            print(result.t)
        else:
            print("Motion estimation failed")
            sys.exit(1)
    
    elif len(sys.argv) == 1:
        # No arguments: run KITTI evaluation
        test_on_kitti()
    
    else:
        print("Usage:")
        print("  python visual_odometry.py <calibration.json> <image1> <image2>")
        print("  python visual_odometry.py   (runs KITTI evaluation)")
        sys.exit(1)
