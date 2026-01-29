#!/usr/bin/env python3
"""
6-DOF Camera Motion Estimator
=============================
Estimates rotation matrix (R) and translation vector (t) between two monocular images.

Usage:
    python camera_motion_estimator.py <image1> <image2> <calibration_file> [--output_dir <dir>]

Author: Computer Vision Pipeline
Date: January 2026
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import cv2
import numpy as np
from numpy.typing import NDArray


# =============================================================================
# Configuration Constants
# =============================================================================

# Target resolution for processing
TARGET_WIDTH = 640
TARGET_HEIGHT = 480

# ORB Feature Detection Parameters (optimized for accuracy)
ORB_N_FEATURES = 5000          # High number for better coverage
ORB_SCALE_FACTOR = 1.2         # Pyramid scale factor
ORB_N_LEVELS = 8               # Number of pyramid levels
ORB_EDGE_THRESHOLD = 31        # Border margin for features
ORB_FIRST_LEVEL = 0            # Start from original resolution
ORB_WTA_K = 2                  # Number of points for binary test
ORB_PATCH_SIZE = 31            # Descriptor patch size
ORB_FAST_THRESHOLD = 20        # FAST corner threshold

# Matching Parameters
RATIO_TEST_THRESHOLD = 0.70    # Lowe's ratio test (conservative)
MAX_HAMMING_DISTANCE = 64      # Maximum acceptable Hamming distance

# RANSAC Parameters (optimized for accuracy)
RANSAC_CONFIDENCE = 0.9999     # Very high confidence
RANSAC_THRESHOLD = 1.0         # Strict inlier threshold (pixels)
RANSAC_MAX_ITERS = 10000       # Very high iteration count for accuracy

# Use USAC (Universal RANSAC) if available - much better than regular RANSAC
# USAC methods: USAC_DEFAULT, USAC_PARALLEL, USAC_FM_8PTS, USAC_FAST, USAC_ACCURATE, USAC_PROSAC, USAC_MAGSAC
USE_USAC = True
USAC_METHOD = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.USAC_DEFAULT if hasattr(cv2, 'USAC_DEFAULT') else None

# Essential matrix estimation parameters
ESSENTIAL_THRESHOLD = 1.0      # Threshold for essential matrix estimation

# Quality Thresholds
MIN_FEATURES_WARNING = 100     # Warn if fewer features detected
MIN_MATCHES_WARNING = 50       # Warn if fewer matches found
MIN_INLIERS_WARNING = 20       # Warn if fewer inliers after RANSAC
MIN_INLIERS_CRITICAL = 8       # Minimum required for fundamental matrix

# CLAHE Parameters
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_SIZE = 8


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class QualityMetrics:
    """Stores quality assessment metrics for the estimation."""
    features_image1: int = 0
    features_image2: int = 0
    raw_matches: int = 0
    ratio_test_matches: int = 0
    cross_check_matches: int = 0
    ransac_inliers: int = 0
    inlier_ratio: float = 0.0
    mean_reprojection_error: float = 0.0
    median_reprojection_error: float = 0.0
    max_reprojection_error: float = 0.0
    fundamental_matrix_rank: int = 0
    essential_matrix_condition: float = 0.0
    cheirality_positive_ratio: float = 0.0
    quality_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def compute_quality_score(self):
        """Compute overall quality score (0-100)."""
        score = 100.0
        
        # Penalize for low feature count
        if self.features_image1 < MIN_FEATURES_WARNING:
            score -= 15
        if self.features_image2 < MIN_FEATURES_WARNING:
            score -= 15
            
        # Penalize for low match count
        if self.ratio_test_matches < MIN_MATCHES_WARNING:
            score -= 20
            
        # Penalize for low inlier count
        if self.ransac_inliers < MIN_INLIERS_WARNING:
            score -= 25
        elif self.ransac_inliers < 50:
            score -= 10
            
        # Penalize for low inlier ratio
        if self.inlier_ratio < 0.3:
            score -= 15
        elif self.inlier_ratio < 0.5:
            score -= 5
            
        # Penalize for high reprojection error
        if self.mean_reprojection_error > 2.0:
            score -= 15
        elif self.mean_reprojection_error > 1.0:
            score -= 5
            
        # Penalize for poor cheirality
        if self.cheirality_positive_ratio < 0.8:
            score -= 10
            
        self.quality_score = max(0.0, min(100.0, score))
        return self.quality_score


@dataclass
class MotionEstimationResult:
    """Complete result of motion estimation."""
    R: np.ndarray                          # 3x3 rotation matrix
    t: np.ndarray                          # 3x1 translation vector
    F: np.ndarray                          # 3x3 fundamental matrix
    E: np.ndarray                          # 3x3 essential matrix
    inlier_points1: np.ndarray             # Inlier points in image 1
    inlier_points2: np.ndarray             # Inlier points in image 2
    metrics: QualityMetrics                # Quality metrics
    processing_time_ms: float = 0.0        # Total processing time


# =============================================================================
# Image Preprocessing Module
# =============================================================================

class ImagePreprocessor:
    """Handles image loading, enhancement, and downsampling."""
    
    def __init__(self, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT):
        self.target_width = target_width
        self.target_height = target_height
        self.scale_x = 1.0
        self.scale_y = 1.0
        
    def load_image(self, path: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
    
    def compute_scale_factors(self, original_width: int, original_height: int) -> Tuple[float, float]:
        """Compute scale factors for downsampling."""
        self.scale_x = self.target_width / original_width
        self.scale_y = self.target_height / original_height
        return self.scale_x, self.scale_y
    
    def downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample image using Gaussian pyramid for anti-aliasing."""
        h, w = image.shape[:2]
        self.compute_scale_factors(w, h)
        
        # Use INTER_AREA for downsampling (best quality)
        # Apply slight Gaussian blur before to reduce aliasing
        if self.scale_x < 1.0 or self.scale_y < 1.0:
            # Calculate appropriate blur kernel size
            blur_size = max(3, int(1.0 / min(self.scale_x, self.scale_y)) | 1)
            if blur_size > 1:
                image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        
        return cv2.resize(image, (self.target_width, self.target_height), 
                         interpolation=cv2.INTER_AREA)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_clahe(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for contrast enhancement."""
        clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE)
        )
        return clahe.apply(gray_image)
    
    def apply_bilateral_filter(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving noise reduction."""
        return cv2.bilateralFilter(gray_image, d=5, sigmaColor=50, sigmaSpace=50)
    
    def check_image_quality(self, gray_image: np.ndarray) -> Tuple[bool, List[str]]:
        """Check image quality and return warnings."""
        warnings = []
        is_ok = True
        
        # Check for motion blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        if laplacian_var < 100:
            warnings.append(f"Possible motion blur detected (Laplacian variance: {laplacian_var:.1f})")
            
        # Check for overexposure
        mean_val = np.mean(gray_image)
        if mean_val > 240:
            warnings.append(f"Image may be overexposed (mean: {mean_val:.1f})")
        elif mean_val < 15:
            warnings.append(f"Image may be underexposed (mean: {mean_val:.1f})")
            
        # Check for low contrast
        std_val = np.std(gray_image)
        if std_val < 20:
            warnings.append(f"Low contrast detected (std: {std_val:.1f})")
            
        return is_ok, warnings
    
    def preprocess(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """Complete preprocessing pipeline."""
        # Load
        color_img = self.load_image(image_path)
        
        # Downsample
        color_downsampled = self.downsample(color_img)
        
        # Convert to grayscale
        gray = self.to_grayscale(color_downsampled)
        
        # Check quality
        _, warnings = self.check_image_quality(gray)
        
        # Apply CLAHE
        enhanced = self.apply_clahe(gray)
        
        # Apply bilateral filter
        filtered = self.apply_bilateral_filter(enhanced)
        
        return filtered, warnings


# =============================================================================
# Calibration Matrix Handler
# =============================================================================

class CalibrationHandler:
    """Handles loading and scaling of camera calibration matrix."""
    
    def __init__(self):
        self.K_original = None
        self.K_scaled = None
        
    def load_calibration(self, path_str: str) -> np.ndarray:
        """Load calibration matrix from file (supports .npy, .json, .txt)."""
        path = Path(path_str)
        
        if path.suffix == '.npy':
            self.K_original = np.load(str(path))
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if 'K' in data:
                    self.K_original = np.array(data['K'])
                elif 'camera_matrix' in data:
                    self.K_original = np.array(data['camera_matrix'])
                else:
                    # Assume the JSON is just the matrix
                    self.K_original = np.array(data)
        elif path.suffix in ['.txt', '.csv']:
            self.K_original = np.loadtxt(str(path), delimiter=None)
        else:
            # Try numpy format
            try:
                self.K_original = np.load(str(path))
            except:
                self.K_original = np.loadtxt(str(path))
                
        # Validate shape
        if self.K_original.shape != (3, 3):
            raise ValueError(f"Calibration matrix must be 3x3, got {self.K_original.shape}")
            
        return self.K_original
    
    def scale_calibration(self, scale_x: float, scale_y: float) -> np.ndarray:
        """Scale calibration matrix for downsampled images."""
        if self.K_original is None:
            raise ValueError("Calibration matrix not loaded")
            
        self.K_scaled = self.K_original.copy().astype(np.float64)
        
        # Scale focal lengths and principal point
        self.K_scaled[0, 0] *= scale_x  # fx
        self.K_scaled[1, 1] *= scale_y  # fy
        self.K_scaled[0, 2] *= scale_x  # cx
        self.K_scaled[1, 2] *= scale_y  # cy
        
        return self.K_scaled


# =============================================================================
# Feature Detection and Matching Module
# =============================================================================

class FeatureProcessor:
    """Handles ORB feature detection, description, and matching."""
    
    def __init__(self):
        # Create ORB detector with optimized parameters
        self.orb = cv2.ORB_create(
            nfeatures=ORB_N_FEATURES,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_N_LEVELS,
            edgeThreshold=ORB_EDGE_THRESHOLD,
            firstLevel=ORB_FIRST_LEVEL,
            WTA_K=ORB_WTA_K,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=ORB_PATCH_SIZE,
            fastThreshold=ORB_FAST_THRESHOLD
        )
        
        # Create brute-force matcher with Hamming distance
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
        
    def detect_and_compute(self, image: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:
        """Detect keypoints and compute descriptors."""
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def spatial_bucketing(self, keypoints: Any, 
                          image_shape: Tuple[int, int],
                          grid_size: Tuple[int, int] = (8, 6),
                          max_per_cell: int = 100) -> List[int]:
        """
        Apply spatial bucketing to ensure features are distributed across the image.
        Returns indices of selected keypoints.
        """
        h, w = image_shape[:2]
        cell_h = h / grid_size[1]
        cell_w = w / grid_size[0]
        
        # Create buckets
        buckets: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        for idx, kp in enumerate(keypoints):
            pt = kp.pt  # type: ignore
            response = kp.response  # type: ignore
            cell_x = min(int(pt[0] / cell_w), grid_size[0] - 1)
            cell_y = min(int(pt[1] / cell_h), grid_size[1] - 1)
            cell_key = (cell_x, cell_y)
            
            if cell_key not in buckets:
                buckets[cell_key] = []
            buckets[cell_key].append((idx, response))
        
        # Select top keypoints from each bucket
        selected_indices = []
        for cell_key, kp_list in buckets.items():
            # Sort by response (descending)
            kp_list.sort(key=lambda x: x[1], reverse=True)
            # Take top N
            for idx, _ in kp_list[:max_per_cell]:
                selected_indices.append(idx)
                
        return selected_indices
    
    def match_features(self, desc1: Optional[np.ndarray], desc2: Optional[np.ndarray]) -> List[Any]:
        """Match features using kNN with ratio test."""
        if desc1 is None or desc2 is None:
            return []
        if len(desc1) < 2 or len(desc2) < 2:
            return []
            
        # Find 2 nearest neighbors for ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        return matches
    
    def apply_ratio_test(self, matches: List[Any], threshold: float = RATIO_TEST_THRESHOLD) -> List[Any]:
        """Apply Lowe's ratio test to filter matches."""
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < threshold * n.distance:
                    good_matches.append(m)
        return good_matches
    
    def apply_cross_check(self, desc1: np.ndarray, desc2: np.ndarray,
                          forward_matches: List[Any]) -> List[Any]:
        """Apply cross-check validation."""
        if len(forward_matches) == 0:
            return []
            
        # Match in reverse direction
        reverse_matches = self.matcher.knnMatch(desc2, desc1, k=2)
        reverse_good = self.apply_ratio_test(reverse_matches)
        
        # Build reverse lookup
        reverse_lookup = {m.queryIdx: m.trainIdx for m in reverse_good}  # type: ignore
        
        # Keep only mutual matches
        cross_checked: List[Any] = []
        for m in forward_matches:
            if m.trainIdx in reverse_lookup:  # type: ignore
                if reverse_lookup[m.trainIdx] == m.queryIdx:  # type: ignore
                    cross_checked.append(m)
                    
        return cross_checked
    
    def filter_by_distance(self, matches: List[Any], 
                           max_distance: int = MAX_HAMMING_DISTANCE) -> List[Any]:
        """Filter matches by maximum Hamming distance."""
        return [m for m in matches if m.distance <= max_distance]  # type: ignore
    
    def extract_matched_points(self, kp1: Any, kp2: Any,
                               matches: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract matched point coordinates."""
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)  # type: ignore
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)  # type: ignore
        return pts1, pts2


# =============================================================================
# Geometry Estimation Module
# =============================================================================

class GeometryEstimator:
    """Handles fundamental matrix, essential matrix, and motion estimation."""
    
    def __init__(self, K: np.ndarray):
        self.K = K.astype(np.float64)
        self.K_inv = np.linalg.inv(self.K)
        
    def estimate_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate fundamental matrix using RANSAC/USAC.
        Returns F and inlier mask.
        """
        # Minimum points check
        if len(pts1) < 8 or len(pts2) < 8:
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
        
        # Check for degenerate case: points too close together
        pts_diff = np.linalg.norm(pts1 - pts2, axis=1)
        median_motion = np.median(pts_diff)
        if median_motion < 0.5:  # Less than 0.5 pixel median motion
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
        
        try:
            # Try USAC first (better than regular RANSAC)
            if USE_USAC and USAC_METHOD is not None:
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=USAC_METHOD,
                    ransacReprojThreshold=RANSAC_THRESHOLD,
                    confidence=RANSAC_CONFIDENCE,
                    maxIters=RANSAC_MAX_ITERS
                )
            else:
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=RANSAC_THRESHOLD,
                    confidence=RANSAC_CONFIDENCE,
                    maxIters=RANSAC_MAX_ITERS
                )
            
            if F is None or F.shape != (3, 3):
                # Fallback to regular RANSAC with relaxed threshold
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=2.0,
                    confidence=0.999,
                    maxIters=RANSAC_MAX_ITERS
                )
                
            if F is None or F.shape != (3, 3):
                # Further fallback: LMEDS (Least Median of Squares) - no threshold needed
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_LMEDS,
                    confidence=0.99
                )
                
            if F is None or F.shape != (3, 3):
                # Last resort: 8-point algorithm
                F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_8POINT)
                if mask is None:
                    mask = np.ones((len(pts1), 1), dtype=np.uint8)
                    
            if F is None:
                return None, np.zeros((len(pts1), 1), dtype=np.uint8)
                
            if mask is None:
                mask = np.ones((len(pts1), 1), dtype=np.uint8)
            
            # Enforce rank-2 constraint on F
            U, S, Vt = np.linalg.svd(F)
            S[2] = 0  # Set smallest singular value to 0
            F = U @ np.diag(S) @ Vt
                
        except cv2.error:
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
                
        return F, mask
    
    def estimate_essential_matrix_direct(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate essential matrix directly using calibrated points.
        This is more accurate than computing E from F.
        Returns E and inlier mask.
        """
        # Minimum points check
        if len(pts1) < 5 or len(pts2) < 5:
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
        
        # Check for degenerate case
        pts_diff = np.linalg.norm(pts1 - pts2, axis=1)
        median_motion = np.median(pts_diff)
        if median_motion < 0.5:
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
        
        try:
            # Try USAC first
            if USE_USAC and USAC_METHOD is not None:
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.K,
                    method=USAC_METHOD,
                    prob=RANSAC_CONFIDENCE,
                    threshold=ESSENTIAL_THRESHOLD,
                    maxIters=RANSAC_MAX_ITERS
                )
            else:
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.K,
                    method=cv2.RANSAC,
                    prob=RANSAC_CONFIDENCE,
                    threshold=ESSENTIAL_THRESHOLD,
                    maxIters=RANSAC_MAX_ITERS
                )
            
            if E is None or E.shape != (3, 3):
                # Fallback to LMEDS
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.K,
                    method=cv2.LMEDS,
                    prob=0.999
                )
            
            if E is None:
                return None, np.zeros((len(pts1), 1), dtype=np.uint8)
            
            if mask is None:
                mask = np.ones((len(pts1), 1), dtype=np.uint8)
            
            # Enforce essential matrix constraints
            E = self.enforce_essential_constraints(E)
                
        except cv2.error:
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
                
        return E, mask
    
    def compute_essential_matrix(self, F: np.ndarray) -> np.ndarray:
        """Compute essential matrix from fundamental matrix: E = K^T * F * K"""
        E = self.K.T @ F @ self.K
        return E
    
    def enforce_essential_constraints(self, E: np.ndarray) -> np.ndarray:
        """
        Enforce essential matrix constraints using SVD.
        E must have two equal singular values and one zero.
        """
        U, S, Vt = np.linalg.svd(E)
        
        # Enforce constraint: singular values should be (σ, σ, 0)
        sigma = (S[0] + S[1]) / 2.0
        S_corrected = np.array([sigma, sigma, 0.0])
        
        # Reconstruct E
        E_corrected = U @ np.diag(S_corrected) @ Vt
        
        return E_corrected
    
    def decompose_essential_matrix(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Decompose essential matrix into R and t.
        Uses cheirality check to select correct solution.
        Returns R, t, and number of points passing cheirality.
        """
        # Minimum points check
        if len(pts1) < 5 or len(pts2) < 5:
            return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
        
        try:
            # Use OpenCV's recoverPose which handles decomposition and cheirality
            retval, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
            
            # Validate outputs
            if R is None or t is None:
                return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
            
            # Ensure R is a valid rotation matrix
            det_R = np.linalg.det(R)
            if abs(det_R - 1.0) > 0.1:
                # Invalid rotation - try to fix it
                U, S, Vt = np.linalg.svd(R)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    R = -R
            
            # Count points passing cheirality check
            if mask is not None:
                num_positive = int(np.sum(mask > 0))
            else:
                num_positive = len(pts1)
            
            return R, t, num_positive
            
        except cv2.error as e:
            # OpenCV error during decomposition
            # Try manual decomposition as fallback
            try:
                return self._manual_decompose_essential(E, pts1, pts2)
            except:
                return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
        except Exception as e:
            return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
    
    def _manual_decompose_essential(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Manual decomposition of essential matrix as fallback.
        """
        U, S, Vt = np.linalg.svd(E)
        
        # Ensure proper rotation
        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vt) < 0:
            Vt = -Vt
        
        # W matrix for decomposition
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        
        # Four possible solutions
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2:3]  # Third column of U
        
        # Pick the solution with most points in front
        best_R, best_t, best_count = R1, t, 0
        
        for R in [R1, R2]:
            for sign in [1, -1]:
                t_test = sign * t
                count = self._count_points_in_front(R, t_test, pts1, pts2)
                if count > best_count:
                    best_R, best_t, best_count = R, t_test, count
        
        return best_R, best_t, best_count
    
    def _count_points_in_front(self, R: np.ndarray, t: np.ndarray, 
                                pts1: np.ndarray, pts2: np.ndarray) -> int:
        """
        Count how many triangulated points are in front of both cameras.
        """
        try:
            P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.K @ np.hstack([R, t])
            
            pts1_h = pts1.T
            pts2_h = pts2.T
            
            points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            points_3d = points_4d[:3] / (points_4d[3:] + 1e-10)
            
            # Check z > 0 in camera 1
            in_front_1 = points_3d[2, :] > 0
            
            # Transform to camera 2 and check z > 0
            points_cam2 = R @ points_3d + t
            in_front_2 = points_cam2[2, :] > 0
            
            return int(np.sum(in_front_1 & in_front_2))
        except:
            return 0
    
    def refine_pose_with_all_solutions(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Try all four decomposition solutions and pick the best one based on
        cheirality check AND reprojection error.
        """
        if len(pts1) < 5 or len(pts2) < 5:
            return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
        
        try:
            # SVD decomposition
            U, S, Vt = np.linalg.svd(E)
            
            # Ensure proper rotation (det should be +1)
            if np.linalg.det(U) < 0:
                U = -U
            if np.linalg.det(Vt) < 0:
                Vt = -Vt
            
            # W matrix for decomposition
            W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
            
            # Four possible solutions
            R1 = U @ W @ Vt
            R2 = U @ W.T @ Vt
            t_base = U[:, 2:3]
            
            # Evaluate all four solutions
            candidates = []
            for R in [R1, R2]:
                # Ensure R is valid rotation
                if np.linalg.det(R) < 0:
                    R = -R
                    
                for sign in [1, -1]:
                    t = sign * t_base
                    
                    # Count points in front
                    count = self._count_points_in_front(R, t, pts1, pts2)
                    
                    # Compute reprojection error for valid points
                    if count > len(pts1) * 0.5:  # At least 50% of points in front
                        reproj_err = self.compute_reprojection_error(R, t, pts1, pts2)
                        median_err = np.median(reproj_err) if len(reproj_err) > 0 else float('inf')
                    else:
                        median_err = float('inf')
                    
                    candidates.append((R, t, count, median_err))
            
            # Pick the best: most points in front, then lowest reprojection error
            candidates.sort(key=lambda x: (-x[2], x[3]))
            
            if candidates and candidates[0][2] > 0:
                return candidates[0][0], candidates[0][1], candidates[0][2]
            
        except Exception as e:
            pass
        
        return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
    
    def compute_sampson_error(self, F: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Compute Sampson distance for each point correspondence.
        This is a first-order approximation to geometric error.
        """
        if len(pts1) == 0 or len(pts2) == 0:
            return np.array([0.0])
        
        try:
            # Convert to homogeneous coordinates
            pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
            pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
            
            errors = []
            for p1, p2 in zip(pts1_h, pts2_h):
                # Epipolar constraint: p2^T * F * p1
                Fp1 = F @ p1
                Ftp2 = F.T @ p2
                p2tFp1 = p2 @ F @ p1
                
                # Sampson error with safe denominator
                denom = Fp1[0]**2 + Fp1[1]**2 + Ftp2[0]**2 + Ftp2[1]**2 + 1e-10
                error = (p2tFp1 ** 2) / denom
                errors.append(np.sqrt(max(0, error)))  # Ensure non-negative
                
            result = np.array(errors)
            # Replace any inf/nan with median or 0
            valid = np.isfinite(result)
            if np.any(valid):
                median_val = np.median(result[valid])
                result[~valid] = median_val
            else:
                result = np.zeros_like(result)
            
            return result
        except Exception as e:
            return np.array([0.0])
    
    def compute_reprojection_error(self, R: np.ndarray, t: np.ndarray,
                                    pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Compute reprojection error by triangulating points and reprojecting.
        """
        # Check for minimum points
        if len(pts1) < 2 or len(pts2) < 2:
            return np.array([0.0])
        
        try:
            # Projection matrices
            P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.K @ np.hstack([R, t])
            
            # Triangulate points
            pts1_h = pts1.T  # 2xN
            pts2_h = pts2.T  # 2xN
            
            points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            
            # Handle division by zero in homogeneous conversion
            w = points_4d[3]
            w[np.abs(w) < 1e-10] = 1e-10  # Prevent division by zero
            points_3d = points_4d[:3] / w  # Convert from homogeneous
            
            # Check for invalid points (inf/nan)
            valid_mask = np.all(np.isfinite(points_3d), axis=0)
            if not np.any(valid_mask):
                return np.array([0.0])
            
            # Reproject to camera 2
            points_3d_h = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
            projected = P2 @ points_3d_h
            
            # Handle division by zero in projection
            z = projected[2]
            z[np.abs(z) < 1e-10] = 1e-10
            projected = projected[:2] / z
            
            # Compute reprojection error only for valid points
            errors = np.sqrt(np.sum((projected.T - pts2) ** 2, axis=1))
            
            # Filter out invalid errors
            errors = errors[np.isfinite(errors)]
            if len(errors) == 0:
                return np.array([0.0])
            
            return errors
            
        except Exception as e:
            # Return zero error on any exception
            return np.array([0.0])


# =============================================================================
# Main Pipeline
# =============================================================================

class CameraMotionEstimator:
    """Main pipeline for 6-DOF camera motion estimation."""
    
    def __init__(self, calibration_path: str):
        """Initialize with calibration matrix."""
        self.preprocessor = ImagePreprocessor()
        self.calibration = CalibrationHandler()
        self.calibration.load_calibration(calibration_path)
        self.feature_processor = FeatureProcessor()
        self.geometry_estimator = None  # Initialized after scaling K
        
    def estimate(self, image1_path: str, image2_path: str) -> MotionEstimationResult:
        """
        Complete motion estimation pipeline.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            MotionEstimationResult containing R, t, and quality metrics
        """
        start_time = time.time()
        metrics = QualityMetrics()
        
        # Wrap entire pipeline in try-except for robustness
        try:
            return self._estimate_internal(image1_path, image2_path, start_time, metrics)
        except Exception as e:
            # Catch any unexpected errors
            metrics.warnings.append(f"Pipeline exception: {str(e)}")
            return self._create_identity_result(metrics, start_time)
    
    def _estimate_internal(self, image1_path: str, image2_path: str, 
                           start_time: float, metrics: QualityMetrics) -> MotionEstimationResult:
        """
        Internal motion estimation - called by estimate() with exception handling.
        """
        
        print("=" * 70)
        print("6-DOF Camera Motion Estimation Pipeline")
        print("=" * 70)
        
        # ---------------------------------------------------------------------
        # Stage 1: Image Preprocessing
        # ---------------------------------------------------------------------
        print("\n[Stage 1] Image Preprocessing...")
        
        img1, warnings1 = self.preprocessor.preprocess(image1_path)
        metrics.warnings.extend([f"Image 1: {w}" for w in warnings1])
        
        img2, warnings2 = self.preprocessor.preprocess(image2_path)
        metrics.warnings.extend([f"Image 2: {w}" for w in warnings2])
        
        # Scale calibration matrix
        K_scaled = self.calibration.scale_calibration(
            self.preprocessor.scale_x,
            self.preprocessor.scale_y
        )
        self.geometry_estimator = GeometryEstimator(K_scaled)
        
        print(f"  Original resolution: {int(TARGET_WIDTH/self.preprocessor.scale_x)}x{int(TARGET_HEIGHT/self.preprocessor.scale_y)}")
        print(f"  Downsampled to: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"  Scale factors: ({self.preprocessor.scale_x:.4f}, {self.preprocessor.scale_y:.4f})")
        
        if warnings1 or warnings2:
            print("  Warnings detected:")
            for w in warnings1 + warnings2:
                print(f"    - {w}")
        
        # ---------------------------------------------------------------------
        # Stage 2: Feature Detection
        # ---------------------------------------------------------------------
        print("\n[Stage 2] Feature Detection (ORB)...")
        
        kp1, desc1 = self.feature_processor.detect_and_compute(img1)
        kp2, desc2 = self.feature_processor.detect_and_compute(img2)
        
        metrics.features_image1 = len(kp1)
        metrics.features_image2 = len(kp2)
        
        print(f"  Image 1: {len(kp1)} features detected")
        print(f"  Image 2: {len(kp2)} features detected")
        
        if len(kp1) < MIN_FEATURES_WARNING:
            metrics.warnings.append(f"Low feature count in image 1: {len(kp1)}")
        if len(kp2) < MIN_FEATURES_WARNING:
            metrics.warnings.append(f"Low feature count in image 2: {len(kp2)}")
        
        # Check for critical failure
        if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
            print("  [CRITICAL] Insufficient features detected!")
            return self._create_identity_result(metrics, start_time)
        
        # ---------------------------------------------------------------------
        # Stage 3: Feature Matching
        # ---------------------------------------------------------------------
        print("\n[Stage 3] Feature Matching...")
        
        # Initial matching
        raw_matches = self.feature_processor.match_features(desc1, desc2)
        metrics.raw_matches = len(raw_matches)
        print(f"  Raw kNN matches: {len(raw_matches)}")
        
        # Ratio test
        ratio_matches = self.feature_processor.apply_ratio_test(raw_matches)
        metrics.ratio_test_matches = len(ratio_matches)
        print(f"  After ratio test (threshold={RATIO_TEST_THRESHOLD}): {len(ratio_matches)}")
        
        # Cross-check
        cross_checked = self.feature_processor.apply_cross_check(desc1, desc2, ratio_matches)
        metrics.cross_check_matches = len(cross_checked)
        print(f"  After cross-check: {len(cross_checked)}")
        
        # Distance filter
        filtered_matches = self.feature_processor.filter_by_distance(cross_checked)
        print(f"  After distance filter (max={MAX_HAMMING_DISTANCE}): {len(filtered_matches)}")
        
        if len(filtered_matches) < MIN_MATCHES_WARNING:
            metrics.warnings.append(f"Low match count: {len(filtered_matches)}")
            
        # Check for critical failure
        if len(filtered_matches) < MIN_INLIERS_CRITICAL:
            print("  [CRITICAL] Insufficient matches!")
            return self._create_identity_result(metrics, start_time)
        
        # Extract point coordinates
        pts1, pts2 = self.feature_processor.extract_matched_points(kp1, kp2, filtered_matches)
        
        # ---------------------------------------------------------------------
        # Stage 4: Essential Matrix Estimation (Direct Method with USAC)
        # ---------------------------------------------------------------------
        print("\n[Stage 4] Essential Matrix Estimation (Direct with USAC)...")
        
        # Try direct essential matrix estimation first (more accurate)
        E_direct, inlier_mask_E = self.geometry_estimator.estimate_essential_matrix_direct(pts1, pts2)
        
        if E_direct is not None:
            print("  Using direct Essential matrix estimation")
            E = E_direct
            inlier_mask = inlier_mask_E
            
            # Also compute F for quality metrics
            F = self.geometry_estimator.K_inv.T @ E @ self.geometry_estimator.K_inv
        else:
            # Fallback to Fundamental matrix approach
            print("  Falling back to Fundamental matrix estimation")
            F, inlier_mask = self.geometry_estimator.estimate_fundamental_matrix(pts1, pts2)
            
            if F is None:
                print("  [CRITICAL] Failed to estimate fundamental matrix!")
                return self._create_identity_result(metrics, start_time)
            
            E = self.geometry_estimator.compute_essential_matrix(F)
            E = self.geometry_estimator.enforce_essential_constraints(E)
        
        # Extract inliers
        inlier_mask = inlier_mask.ravel().astype(bool)
        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        
        metrics.ransac_inliers = len(inlier_pts1)
        metrics.inlier_ratio = len(inlier_pts1) / len(pts1) if len(pts1) > 0 else 0
        
        print(f"  Inliers: {metrics.ransac_inliers} / {len(pts1)} ({metrics.inlier_ratio*100:.1f}%)")
        
        if metrics.ransac_inliers < MIN_INLIERS_WARNING:
            metrics.warnings.append(f"Low inlier count after RANSAC: {metrics.ransac_inliers}")
            
        if metrics.ransac_inliers < MIN_INLIERS_CRITICAL:
            print("  [CRITICAL] Insufficient inliers!")
            return self._create_identity_result(metrics, start_time)
        
        # Compute Sampson errors using F
        try:
            # Compute F from E if we used direct method
            if E_direct is not None:
                F = self.geometry_estimator.K_inv.T @ E @ self.geometry_estimator.K_inv
            metrics.fundamental_matrix_rank = np.linalg.matrix_rank(F)
            sampson_errors = self.geometry_estimator.compute_sampson_error(F, inlier_pts1, inlier_pts2)
            print(f"  Sampson error - Mean: {np.mean(sampson_errors):.4f}, Median: {np.median(sampson_errors):.4f}")
        except:
            F = np.eye(3)  # Fallback
            metrics.fundamental_matrix_rank = 3
        
        # ---------------------------------------------------------------------
        # Stage 5: Essential Matrix Verification
        # ---------------------------------------------------------------------
        print("\n[Stage 5] Essential Matrix Verification...")
        
        # Check singular values
        U, S, Vt = np.linalg.svd(E)
        print(f"  Singular values: [{S[0]:.6f}, {S[1]:.6f}, {S[2]:.6f}]")
        
        metrics.essential_matrix_condition = S[0] / (S[1] + 1e-10)
        
        # ---------------------------------------------------------------------
        # Stage 6: Motion Decomposition with Refinement
        # ---------------------------------------------------------------------
        print("\n[Stage 6] Motion Decomposition (R, t) with Refinement...")
        
        # First try OpenCV's recoverPose
        R, t, num_positive = self.geometry_estimator.decompose_essential_matrix(
            E, inlier_pts1, inlier_pts2
        )
        
        # If cheirality is poor, try refined pose selection
        cheirality_ratio = num_positive / len(inlier_pts1) if len(inlier_pts1) > 0 else 0
        if cheirality_ratio < 0.7:
            print("  Low cheirality ratio, trying refined pose selection...")
            R_refined, t_refined, num_refined = self.geometry_estimator.refine_pose_with_all_solutions(
                E, inlier_pts1, inlier_pts2
            )
            if num_refined > num_positive:
                R, t, num_positive = R_refined, t_refined, num_refined
                print(f"  Refined pose selected with {num_positive} points in front")
        
        metrics.cheirality_positive_ratio = num_positive / len(inlier_pts1) if len(inlier_pts1) > 0 else 0
        print(f"  Cheirality check: {num_positive}/{len(inlier_pts1)} points in front ({metrics.cheirality_positive_ratio*100:.1f}%)")
        
        # Verify rotation matrix
        det_R = np.linalg.det(R)
        print(f"  det(R) = {det_R:.6f}")
        
        if abs(det_R - 1.0) > 0.01:
            metrics.warnings.append(f"Rotation matrix determinant deviation: {det_R:.6f}")
            # Fix rotation matrix
            U_R, S_R, Vt_R = np.linalg.svd(R)
            R = U_R @ Vt_R
            if np.linalg.det(R) < 0:
                R = -R
            print(f"  Fixed det(R) = {np.linalg.det(R):.6f}")
            
        # Compute reprojection error
        reproj_errors = self.geometry_estimator.compute_reprojection_error(R, t, inlier_pts1, inlier_pts2)
        metrics.mean_reprojection_error = float(np.mean(reproj_errors))
        metrics.median_reprojection_error = float(np.median(reproj_errors))
        metrics.max_reprojection_error = float(np.max(reproj_errors))
        
        print(f"  Reprojection error - Mean: {metrics.mean_reprojection_error:.4f} px, "
              f"Median: {metrics.median_reprojection_error:.4f} px, Max: {metrics.max_reprojection_error:.4f} px")
        
        # Ensure t is unit vector
        t = t / (np.linalg.norm(t) + 1e-10)
        
        # Reshape t to column vector if needed
        t = t.reshape(3, 1)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Compute quality score
        metrics.compute_quality_score()
        
        # Create result
        result = MotionEstimationResult(
            R=R,
            t=t,
            F=F,
            E=E,
            inlier_points1=inlier_pts1,
            inlier_points2=inlier_pts2,
            metrics=metrics,
            processing_time_ms=processing_time
        )
        
        return result
    
    def _create_identity_result(self, metrics: QualityMetrics, start_time: float) -> MotionEstimationResult:
        """Create a fallback identity result when pipeline fails."""
        metrics.warnings.append("Pipeline failed - returning identity transformation")
        metrics.quality_score = 0.0
        
        return MotionEstimationResult(
            R=np.eye(3),
            t=np.zeros((3, 1)),
            F=np.eye(3),
            E=np.eye(3),
            inlier_points1=np.array([]),
            inlier_points2=np.array([]),
            metrics=metrics,
            processing_time_ms=(time.time() - start_time) * 1000
        )


# =============================================================================
# Output Functions
# =============================================================================

def print_result(result: MotionEstimationResult):
    """Print estimation results to console."""
    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    
    print("\n[Rotation Matrix R (3x3)]:")
    print(np.array2string(result.R, precision=8, suppress_small=True, 
                          formatter={'float_kind': lambda x: f"{x:12.8f}"}))
    
    print("\n[Translation Vector t (3x1, unit norm)]:")
    print(np.array2string(result.t.flatten(), precision=8, suppress_small=True,
                          formatter={'float_kind': lambda x: f"{x:12.8f}"}))
    
    # Convert rotation to axis-angle for interpretability
    angle = np.arccos(np.clip((np.trace(result.R) - 1) / 2, -1, 1))
    print(f"\n[Rotation Angle]: {np.degrees(angle):.4f} degrees")
    
    print(f"\n[Processing Time]: {result.processing_time_ms:.2f} ms")
    
    print("\n[Quality Metrics]:")
    m = result.metrics
    print(f"  Features detected:     Image1={m.features_image1}, Image2={m.features_image2}")
    print(f"  Matches:               Raw={m.raw_matches}, After ratio test={m.ratio_test_matches}")
    print(f"  Cross-checked matches: {m.cross_check_matches}")
    print(f"  RANSAC inliers:        {m.ransac_inliers} ({m.inlier_ratio*100:.1f}%)")
    print(f"  Reprojection error:    Mean={m.mean_reprojection_error:.4f} px, Median={m.median_reprojection_error:.4f} px")
    print(f"  Cheirality ratio:      {m.cheirality_positive_ratio*100:.1f}%")
    print(f"  Quality Score:         {m.quality_score:.1f}/100")
    
    if m.warnings:
        print("\n[Warnings]:")
        for warning in m.warnings:
            print(f"  ⚠ {warning}")
    
    if m.quality_score < 50:
        print("\n" + "!" * 70)
        print("WARNING: Low quality estimation! Results may be unreliable.")
        print("!" * 70)


def save_results(result: MotionEstimationResult, output_dir: str, prefix: str = "motion"):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save rotation matrix
    R_path = output_path / f"{prefix}_R.npy"
    np.save(str(R_path), result.R)
    
    # Save translation vector
    t_path = output_path / f"{prefix}_t.npy"
    np.save(str(t_path), result.t)
    
    # Save fundamental matrix
    F_path = output_path / f"{prefix}_F.npy"
    np.save(str(F_path), result.F)
    
    # Save essential matrix
    E_path = output_path / f"{prefix}_E.npy"
    np.save(str(E_path), result.E)
    
    # Save human-readable summary
    summary_path = output_path / f"{prefix}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("6-DOF Camera Motion Estimation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Rotation Matrix R:\n")
        f.write(np.array2string(result.R, precision=10) + "\n\n")
        
        f.write("Translation Vector t (unit norm):\n")
        f.write(np.array2string(result.t.flatten(), precision=10) + "\n\n")
        
        angle = np.arccos(np.clip((np.trace(result.R) - 1) / 2, -1, 1))
        f.write(f"Rotation Angle: {np.degrees(angle):.6f} degrees\n\n")
        
        f.write("Quality Metrics:\n")
        f.write(f"  Quality Score: {result.metrics.quality_score:.1f}/100\n")
        f.write(f"  Features: {result.metrics.features_image1}, {result.metrics.features_image2}\n")
        f.write(f"  RANSAC Inliers: {result.metrics.ransac_inliers}\n")
        f.write(f"  Inlier Ratio: {result.metrics.inlier_ratio*100:.1f}%\n")
        f.write(f"  Mean Reprojection Error: {result.metrics.mean_reprojection_error:.4f} px\n")
        f.write(f"  Processing Time: {result.processing_time_ms:.2f} ms\n\n")
        
        if result.metrics.warnings:
            f.write("Warnings:\n")
            for warning in result.metrics.warnings:
                f.write(f"  - {warning}\n")
    
    # Save complete result as JSON (for programmatic access)
    json_path = output_path / f"{prefix}_result.json"
    json_data = {
        "R": result.R.tolist(),
        "t": result.t.flatten().tolist(),
        "F": result.F.tolist(),
        "E": result.E.tolist(),
        "rotation_angle_degrees": float(np.degrees(np.arccos(np.clip((np.trace(result.R) - 1) / 2, -1, 1)))),
        "metrics": {
            "quality_score": result.metrics.quality_score,
            "features_image1": result.metrics.features_image1,
            "features_image2": result.metrics.features_image2,
            "ransac_inliers": result.metrics.ransac_inliers,
            "inlier_ratio": result.metrics.inlier_ratio,
            "mean_reprojection_error": result.metrics.mean_reprojection_error,
            "median_reprojection_error": result.metrics.median_reprojection_error,
            "processing_time_ms": result.processing_time_ms
        },
        "warnings": result.metrics.warnings
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n[Output Files Saved]:")
    print(f"  Rotation matrix:    {R_path}")
    print(f"  Translation vector: {t_path}")
    print(f"  Fundamental matrix: {F_path}")
    print(f"  Essential matrix:   {E_path}")
    print(f"  Summary:            {summary_path}")
    print(f"  JSON result:        {json_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6-DOF Camera Motion Estimation between two monocular images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python camera_motion_estimator.py image1.jpg image2.jpg calibration.npy
  python camera_motion_estimator.py img1.jpg img2.jpg calib.json --output_dir results/
  
Calibration file formats supported:
  - NumPy (.npy): 3x3 array
  - JSON (.json): {"K": [[...], [...], [...]]} or {"camera_matrix": ...}
  - Text (.txt): Space or comma separated 3x3 matrix
        """
    )
    
    parser.add_argument("image1", type=str, help="Path to first image")
    parser.add_argument("image2", type=str, help="Path to second image")
    parser.add_argument("calibration", type=str, help="Path to camera calibration matrix file")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save output files (default: ./output)")
    
    args = parser.parse_args()
    
    # Validate input files
    for path in [args.image1, args.image2, args.calibration]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    try:
        # Create estimator
        estimator = CameraMotionEstimator(args.calibration)
        
        # Run estimation
        result = estimator.estimate(args.image1, args.image2)
        
        # Print results
        print_result(result)
        
        # Save results
        save_results(result, args.output_dir)
        
        # Exit with appropriate code
        if result.metrics.quality_score < 20:
            sys.exit(2)  # Very low quality
        elif result.metrics.quality_score < 50:
            sys.exit(1)  # Low quality warning
        else:
            sys.exit(0)  # Success
            
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
