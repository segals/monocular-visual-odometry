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

# Feature Detection Parameters (Shi-Tomasi corners for optical flow)
MAX_CORNERS = 2000             # Maximum corners to detect
CORNER_QUALITY = 0.01          # Quality level for corner detection
MIN_CORNER_DISTANCE = 7        # Minimum distance between corners
CORNER_BLOCK_SIZE = 7          # Block size for corner detection

# Optical Flow Parameters (Lucas-Kanade)
LK_WIN_SIZE = (21, 21)         # Window size for optical flow
LK_MAX_LEVEL = 3               # Maximum pyramid level
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
LK_MIN_EIG_THRESHOLD = 0.001   # Minimum eigenvalue threshold

# Flow Validation Parameters
FORWARD_BACKWARD_THRESHOLD = 1.0  # Max pixel error for forward-backward check
MIN_FLOW_MAGNITUDE = 0.5       # Minimum flow to consider valid

# RANSAC Parameters (optimized for accuracy)
RANSAC_CONFIDENCE = 0.9999     # Very high confidence
RANSAC_THRESHOLD = 0.75        # Strict inlier threshold (pixels)
RANSAC_MAX_ITERS = 15000       # Very high iteration count for accuracy

# Use USAC (Universal RANSAC) if available - much better than regular RANSAC
# USAC methods: USAC_DEFAULT, USAC_PARALLEL, USAC_FM_8PTS, USAC_FAST, USAC_ACCURATE, USAC_PROSAC, USAC_MAGSAC
USE_USAC = True
USAC_METHOD = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.USAC_DEFAULT if hasattr(cv2, 'USAC_DEFAULT') else None

# Essential matrix estimation parameters
ESSENTIAL_THRESHOLD = 0.75     # Threshold for essential matrix estimation

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
    
    def __init__(self, target_width: int = TARGET_WIDTH, target_height: int = TARGET_HEIGHT,
                 preserve_aspect_ratio: bool = True):
        self.target_width = target_width
        self.target_height = target_height
        self.preserve_aspect_ratio = preserve_aspect_ratio
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
        """Downsample image based on mode:
        
        If preserve_aspect_ratio=True (default for KITTI, etc.):
          - Keep native resolution if smaller than target
          - Scale proportionally if larger, preserving aspect ratio
          
        If preserve_aspect_ratio=False (for DJI Tello with fixed calibration):
          - Always resize to exact target dimensions
        """
        h, w = image.shape[:2]
        
        if self.preserve_aspect_ratio:
            # Mode 1: Preserve aspect ratio (for KITTI, etc.)
            # If image is already at or below target size, keep native
            if w <= self.target_width and h <= self.target_height:
                self.scale_x = 1.0
                self.scale_y = 1.0
                return image
            
            # Calculate scale to fit within target while preserving aspect ratio
            scale = min(self.target_width / w, self.target_height / h)
            
            if scale >= 1.0:
                self.scale_x = 1.0
                self.scale_y = 1.0
                return image
            
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            self.scale_x = new_w / w
            self.scale_y = new_h / h
        else:
            # Mode 2: Force exact target dimensions (for DJI Tello)
            new_w = self.target_width
            new_h = self.target_height
            self.scale_x = new_w / w
            self.scale_y = new_h / h
        
        # Apply slight Gaussian blur before downsampling to reduce aliasing
        if self.scale_x < 1.0 or self.scale_y < 1.0:
            blur_size = max(3, int(1.0 / min(self.scale_x, self.scale_y)) | 1)
            if blur_size > 1:
                image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
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
# Feature Detection and Optical Flow Tracking Module
# =============================================================================

class FeatureProcessor:
    """Handles feature detection and optical flow tracking."""
    
    def __init__(self):
        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=LK_WIN_SIZE,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA,
            minEigThreshold=LK_MIN_EIG_THRESHOLD
        )
        
        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=MAX_CORNERS,
            qualityLevel=CORNER_QUALITY,
            minDistance=MIN_CORNER_DISTANCE,
            blockSize=CORNER_BLOCK_SIZE
        )
    
    def detect_features(self, image: np.ndarray) -> np.ndarray:
        """
        Detect good features to track using Shi-Tomasi corner detection.
        Returns Nx1x2 array of corner coordinates.
        """
        corners = cv2.goodFeaturesToTrack(image, mask=None, **self.feature_params)
        if corners is None:
            return np.array([], dtype=np.float32).reshape(0, 1, 2)
        return corners
    
    def track_features(self, img1: np.ndarray, img2: np.ndarray, 
                       pts1: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features from img1 to img2 using Lucas-Kanade optical flow.
        
        Args:
            img1: First grayscale image
            img2: Second grayscale image
            pts1: Points to track (Nx1x2 or Nx2)
            
        Returns:
            pts1_good: Successfully tracked points in image 1
            pts2_good: Corresponding points in image 2
            status: Tracking status for each point
        """
        if len(pts1) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Ensure correct shape for optical flow
        if pts1.ndim == 2:
            pts1 = pts1.reshape(-1, 1, 2)
        pts1 = pts1.astype(np.float32)
        
        # Forward optical flow: img1 -> img2
        pts2, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(
            img1, img2, pts1, None, **self.lk_params
        )
        
        if pts2 is None:
            return np.array([]), np.array([]), np.array([])
        
        # Backward optical flow: img2 -> img1 (for validation)
        pts1_back, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(
            img2, img1, pts2, None, **self.lk_params
        )
        
        # Forward-backward consistency check
        if pts1_back is not None:
            fb_error = np.linalg.norm(pts1 - pts1_back, axis=2).flatten()
            fb_valid = fb_error < FORWARD_BACKWARD_THRESHOLD
        else:
            fb_valid = np.ones(len(pts1), dtype=bool)
        
        # Combine all validity checks
        status_fwd = status_fwd.flatten().astype(bool)
        status_bwd = status_bwd.flatten().astype(bool) if status_bwd is not None else np.ones(len(pts1), dtype=bool)
        
        # Check for minimum flow magnitude
        flow = pts2.reshape(-1, 2) - pts1.reshape(-1, 2)
        flow_mag = np.linalg.norm(flow, axis=1)
        flow_valid = flow_mag >= MIN_FLOW_MAGNITUDE
        
        # Combined validity mask
        valid = status_fwd & status_bwd & fb_valid & flow_valid
        
        # Extract valid points
        pts1_good = pts1.reshape(-1, 2)[valid]
        pts2_good = pts2.reshape(-1, 2)[valid]
        
        return pts1_good, pts2_good, valid
    
    def compute_flow_with_detection(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Complete pipeline: detect features in img1 and track to img2.
        
        Returns:
            pts1: Matched points in image 1 (Nx2)
            pts2: Matched points in image 2 (Nx2)
            num_detected: Number of features initially detected
        """
        # Detect features in first image
        corners = self.detect_features(img1)
        num_detected = len(corners)
        
        if num_detected == 0:
            return np.array([]), np.array([]), 0
        
        # Track to second image
        pts1_good, pts2_good, _ = self.track_features(img1, img2, corners)
        
        return pts1_good, pts2_good, num_detected


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
                # Further fallback: LMEDS (Least Median of Squares)
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_LMEDS,
                    ransacReprojThreshold=1.0,
                    confidence=0.99,
                    maxIters=5000
                )
                
            if F is None or F.shape != (3, 3):
                # Last resort: 8-point algorithm
                F, mask = cv2.findFundamentalMat(
                    pts1, pts2,
                    method=cv2.FM_8POINT,
                    ransacReprojThreshold=1.0,
                    confidence=0.99,
                    maxIters=1000
                )
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
            # Primary method: USAC_MAGSAC (best accuracy)
            if USE_USAC and USAC_METHOD is not None:
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.K,
                    method=USAC_METHOD,
                    prob=RANSAC_CONFIDENCE,
                    threshold=ESSENTIAL_THRESHOLD
                )
                
                if E is not None and E.shape == (3, 3) and mask is not None:
                    E = self.enforce_essential_constraints(E)
                    return E, mask
            
            # Fallback: Standard RANSAC
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.RANSAC,
                prob=RANSAC_CONFIDENCE,
                threshold=ESSENTIAL_THRESHOLD
            )
            
            if E is not None and E.shape == (3, 3) and mask is not None:
                E = self.enforce_essential_constraints(E)
                return E, mask
            
            # Last fallback: LMEDS
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.LMEDS,
                prob=0.999
            )
            
            if E is not None:
                E = self.enforce_essential_constraints(E)
                if mask is None:
                    mask = np.ones((len(pts1), 1), dtype=np.uint8)
                return E, mask
            
            return None, np.zeros((len(pts1), 1), dtype=np.uint8)
                
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
    
    def check_homography_degeneracy(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[bool, float]:
        """
        Check if the scene is degenerate (planar or nearly planar).
        If homography fits well compared to fundamental matrix, translation is unreliable.
        
        Returns: (is_degenerate, homography_ratio)
        """
        if len(pts1) < 10:
            return False, 0.0
        
        try:
            # Estimate homography
            H, mask_H = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
            if H is None or mask_H is None:
                return False, 0.0
            
            # Estimate fundamental matrix
            F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            if F is None or mask_F is None:
                return False, 0.0
            
            # Count inliers for each
            h_inliers = np.sum(mask_H)
            f_inliers = np.sum(mask_F)
            
            # Compute homography ratio
            if f_inliers > 0:
                ratio = h_inliers / f_inliers
            else:
                ratio = 1.0
            
            # If homography explains points almost as well as fundamental matrix,
            # the scene is likely planar and translation direction is unreliable
            is_degenerate = ratio > 0.85 and h_inliers > len(pts1) * 0.6
            
            return is_degenerate, ratio
            
        except Exception:
            return False, 0.0
    
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
    
    def _evaluate_pose_quality(self, R: np.ndarray, t: np.ndarray, 
                               pts1: np.ndarray, pts2: np.ndarray) -> Tuple[int, float, float]:
        """
        Comprehensive pose quality evaluation using:
        1. Cheirality (points in front of both cameras)
        2. Parallax angle (critical for translation accuracy)
        3. Reprojection error
        4. Depth consistency
        
        Returns: (num_valid, mean_reproj_error, depth_std_ratio)
        """
        try:
            P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.K @ np.hstack([R, t])
            
            pts1_h = pts1.T
            pts2_h = pts2.T
            
            # Triangulate
            points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
            points_3d = points_4d[:3] / (points_4d[3:] + 1e-10)
            
            # Check z > 0 in camera 1 (with margin to avoid numerical issues)
            in_front_1 = points_3d[2, :] > 0.01
            
            # Transform to camera 2 and check z > 0
            points_cam2 = R @ points_3d + t
            in_front_2 = points_cam2[2, :] > 0.01
            
            # Also check for reasonable depth (not too far, not too close)
            reasonable_depth = (points_3d[2, :] > 0.1) & (points_3d[2, :] < 1000)
            
            valid_mask = in_front_1 & in_front_2 & reasonable_depth
            num_valid = int(np.sum(valid_mask))
            
            if num_valid < 5:
                return num_valid, float('inf'), float('inf')
            
            # Compute reprojection error for valid points
            valid_3d = points_3d[:, valid_mask]
            valid_pts1 = pts1[valid_mask]
            valid_pts2 = pts2[valid_mask]
            
            # Project back to image 1
            proj1 = P1 @ np.vstack([valid_3d, np.ones((1, valid_3d.shape[1]))])
            proj1 = proj1[:2] / (proj1[2:] + 1e-10)
            
            # Project back to image 2
            proj2 = P2 @ np.vstack([valid_3d, np.ones((1, valid_3d.shape[1]))])
            proj2 = proj2[:2] / (proj2[2:] + 1e-10)
            
            # Compute errors
            err1 = np.sqrt(np.sum((proj1.T - valid_pts1)**2, axis=1))
            err2 = np.sqrt(np.sum((proj2.T - valid_pts2)**2, axis=1))
            
            # Filter out outlier reprojection errors
            all_errs = np.concatenate([err1, err2])
            median_err = np.median(all_errs)
            good_errs = all_errs[all_errs < median_err * 3]
            mean_reproj = float(np.mean(good_errs)) if len(good_errs) > 0 else float(np.mean(all_errs))
            
            # Depth consistency: ratio of std to mean depth (lower is better)
            depths = valid_3d[2, :]
            if np.mean(depths) > 1e-6:
                depth_consistency = float(np.std(depths) / np.mean(depths))
            else:
                depth_consistency = float('inf')
            
            # Compute parallax angles - critical for translation accuracy
            # Camera center 1 is at origin, camera center 2 is at -R.T @ t
            C2 = -R.T @ t.flatten()
            
            parallax_angles = []
            for i in range(valid_3d.shape[1]):
                p = valid_3d[:, i]
                # Ray from camera 1 to point
                ray1 = p / (np.linalg.norm(p) + 1e-10)
                # Ray from camera 2 to point  
                ray2 = (p - C2) / (np.linalg.norm(p - C2) + 1e-10)
                # Parallax angle
                cos_angle = np.clip(np.dot(ray1, ray2), -1, 1)
                parallax_angles.append(np.arccos(cos_angle))
            
            mean_parallax = np.mean(parallax_angles)
            median_parallax = np.median(parallax_angles)
            
            # Use median parallax for robustness against outliers
            # Parallax below ~1 degree means translation is poorly constrained
            # But we don't want to over-penalize or we might select wrong solution
            parallax_factor = 1.0
            if median_parallax < np.radians(0.5):
                parallax_factor = 2.0  # Increase error weight for very low parallax
            elif median_parallax < np.radians(1.0):
                parallax_factor = 1.5
            
            # Apply parallax factor to reprojection error (makes it less reliable metric)
            adjusted_reproj = mean_reproj * parallax_factor
            
            return num_valid, adjusted_reproj, depth_consistency
            
        except Exception:
            return 0, float('inf'), float('inf')
    
    def refine_pose_with_all_solutions(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Try all four decomposition solutions and pick the best one using
        comprehensive quality metrics: cheirality, reprojection error, depth consistency.
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
            
            # Evaluate all four solutions with comprehensive quality metrics
            candidates = []
            for R in [R1, R2]:
                # Ensure R is valid rotation
                if np.linalg.det(R) < 0:
                    R = -R
                    
                for sign in [1, -1]:
                    t = sign * t_base
                    
                    # Comprehensive evaluation
                    num_valid, reproj_err, depth_consistency = self._evaluate_pose_quality(
                        R, t, pts1, pts2
                    )
                    
                    # Score: prioritize num_valid, then reproj_err
                    if num_valid >= len(pts1) * 0.2:  # At least 20% valid
                        score = (num_valid, -reproj_err, -depth_consistency)
                    else:
                        score = (num_valid, -float('inf'), -float('inf'))
                    
                    candidates.append((R, t.copy(), num_valid, reproj_err, depth_consistency, score))
            
            # Sort by score (descending)
            candidates.sort(key=lambda x: x[5], reverse=True)
            
            if not candidates or candidates[0][2] == 0:
                return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
            
            best = candidates[0]
            R_best, t_best = best[0], best[1]
            best_num = best[2]
            
            # Check if there's ambiguity between top 2 solutions
            if len(candidates) >= 2:
                second = candidates[1]
                # If second solution has similar quality, we need to be careful
                if second[2] >= best[2] * 0.85 and second[3] < best[3] * 1.5:
                    # Use flow-based estimation to break tie
                    t_flow = self.estimate_translation_from_flow(pts1, pts2, R_best)
                    if t_flow is not None:
                        t1_norm = t_best.flatten() / (np.linalg.norm(t_best) + 1e-10)
                        t2_norm = second[1].flatten() / (np.linalg.norm(second[1]) + 1e-10)
                        t_flow_norm = t_flow.flatten() / (np.linalg.norm(t_flow) + 1e-10)
                        
                        # Pick translation closer to flow estimate
                        dot1 = abs(np.dot(t1_norm, t_flow_norm))
                        dot2 = abs(np.dot(t2_norm, t_flow_norm))
                        
                        if dot2 > dot1 + 0.1:  # Second is significantly closer to flow
                            t_best = second[1]
                            best_num = second[2]
            
            # Refine translation direction with finer local optimization
            t_refined = self._refine_translation_direction(R_best, t_best, pts1, pts2)
            
            # Validate refinement improved things
            num_refined, err_refined, _ = self._evaluate_pose_quality(R_best, t_refined, pts1, pts2)
            num_orig, err_orig, _ = self._evaluate_pose_quality(R_best, t_best, pts1, pts2)
            
            # Only use refinement if it clearly improved things
            if err_refined > err_orig * 1.2 or num_refined < num_orig * 0.8:
                t_refined = t_best
            else:
                best_num = max(best_num, num_refined)
                
            return R_best, t_refined, best_num
            
        except Exception as e:
            pass
        
        return np.eye(3), np.array([[0], [0], [1]], dtype=np.float64), 0
    
    def _refine_translation_direction(self, R: np.ndarray, t_init: np.ndarray, 
                                       pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """
        Refine translation direction by minimizing reprojection error
        while keeping rotation fixed. Does coarse-to-fine local search.
        """
        t = t_init.flatten()
        t = t / (np.linalg.norm(t) + 1e-10)  # Normalize
        
        # Convert to spherical coordinates
        theta = np.arctan2(t[1], t[0])
        phi = np.arccos(np.clip(t[2], -1, 1))
        
        best_t = t_init.copy()
        _, best_err, _ = self._evaluate_pose_quality(R, t_init, pts1, pts2)
        
        # Local refinement with coarse-to-fine search (finer grid for better accuracy)
        for scale in [0.2, 0.1, 0.05, 0.02, 0.01]:
            for d_theta in np.linspace(-scale, scale, 5):
                for d_phi in np.linspace(-scale, scale, 5):
                    new_theta = theta + d_theta
                    new_phi = np.clip(phi + d_phi, 0.01, np.pi - 0.01)
                    
                    new_t = np.array([
                        np.sin(new_phi) * np.cos(new_theta),
                        np.sin(new_phi) * np.sin(new_theta),
                        np.cos(new_phi)
                    ]).reshape(3, 1)
                    
                    _, err, _ = self._evaluate_pose_quality(R, new_t, pts1, pts2)
                    
                    if err < best_err:
                        best_err = err
                        best_t = new_t.copy()
                        theta = new_theta
                        phi = new_phi
        
        return best_t
    
    def estimate_translation_from_flow(self, pts1: np.ndarray, pts2: np.ndarray, 
                                        R: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate translation direction from optical flow pattern after removing rotation.
        Uses a robust median-based approach to find the focus of expansion.
        """
        if len(pts1) < 20:
            return None
        
        try:
            # Convert to normalized coordinates
            pts1_norm = (self.K_inv @ np.hstack([pts1, np.ones((len(pts1), 1))]).T).T[:, :2]
            pts2_norm = (self.K_inv @ np.hstack([pts2, np.ones((len(pts2), 1))]).T).T[:, :2]
            
            # Remove rotation from the flow by rotating pts1 with R
            pts1_h = np.hstack([pts1_norm, np.ones((len(pts1_norm), 1))])
            pts1_rotated = (R.T @ pts1_h.T).T[:, :2]
            
            # Flow after rotation removal - should be radial from/to FOE
            flow = pts2_norm - pts1_rotated
            flow_mag = np.linalg.norm(flow, axis=1)
            
            # Filter out very small flows (might be noise)
            valid = flow_mag > 1e-4
            if np.sum(valid) < 10:
                return None
            
            pts_valid = pts2_norm[valid]
            flow_valid = flow[valid]
            flow_mag_valid = flow_mag[valid]
            
            # Normalize flow directions
            flow_dirs = flow_valid / (flow_mag_valid.reshape(-1, 1) + 1e-10)
            
            # For forward motion, FOE is in the direction the flow radiates from
            # Estimate FOE using weighted median of flow line intersections
            # Each point + flow direction defines a line; FOE is where they intersect
            
            # Use RANSAC-like approach: sample pairs and find intersection
            n_pts = len(pts_valid)
            if n_pts < 10:
                return None
            
            foe_candidates = []
            n_samples = min(50, n_pts * (n_pts - 1) // 2)
            
            for _ in range(n_samples):
                i, j = np.random.choice(n_pts, 2, replace=False)
                
                # Two lines: p_i + t * d_i and p_j + s * d_j
                p1, d1 = pts_valid[i], flow_dirs[i]
                p2, d2 = pts_valid[j], flow_dirs[j]
                
                # Solve for intersection using least squares
                # p1 + t*d1 = p2 + s*d2
                # [d1, -d2] * [t, s]^T = p2 - p1
                A = np.column_stack([d1, -d2])
                b = p2 - p1
                
                try:
                    if abs(np.linalg.det(A)) > 1e-6:
                        ts = np.linalg.solve(A, b)
                        foe = p1 + ts[0] * d1
                        foe_candidates.append(foe)
                except:
                    continue
            
            if len(foe_candidates) < 5:
                # Fallback: use weighted average flow direction
                # Weight by flow magnitude (stronger flow = more reliable)
                weights = flow_mag_valid / (np.sum(flow_mag_valid) + 1e-10)
                avg_flow = np.sum(flow_valid * weights.reshape(-1, 1), axis=0)
                t_approx = np.array([-avg_flow[0], -avg_flow[1], 1.0])
            else:
                # Use robust median with outlier rejection
                foe_array = np.array(foe_candidates)
                
                # Remove outliers using MAD
                foe_median_init = np.median(foe_array, axis=0)
                distances = np.linalg.norm(foe_array - foe_median_init, axis=1)
                mad = np.median(distances)
                inliers = distances < mad * 3  # 3 MAD threshold
                
                if np.sum(inliers) >= 3:
                    foe_median = np.median(foe_array[inliers], axis=0)
                else:
                    foe_median = foe_median_init
                
                # Translation points towards FOE (for forward motion)
                t_approx = np.array([foe_median[0], foe_median[1], 1.0])
            
            t_approx = t_approx / (np.linalg.norm(t_approx) + 1e-10)
            return t_approx.reshape(3, 1)
            
        except Exception:
            return None
    
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
    
    def __init__(self, calibration_path: str, preserve_aspect_ratio: bool = True):
        """Initialize with calibration matrix.
        
        Args:
            calibration_path: Path to calibration JSON file
            preserve_aspect_ratio: If True (default), preserves aspect ratio when
                                   downsampling (good for KITTI, datasets with different
                                   aspect ratios). If False, forces exact target dimensions
                                   (use for DJI Tello with fixed calibration).
        """
        self.preprocessor = ImagePreprocessor(preserve_aspect_ratio=preserve_aspect_ratio)
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
        # Stage 2: Feature Detection and Optical Flow Tracking
        # ---------------------------------------------------------------------
        print("\n[Stage 2] Feature Detection (Shi-Tomasi corners)...")
        print("\n[Stage 3] Feature Tracking (Lucas-Kanade Optical Flow)...")
        
        # Detect features in image 1 and track to image 2
        pts1, pts2, num_detected = self.feature_processor.compute_flow_with_detection(img1, img2)
        
        metrics.features_image1 = num_detected
        metrics.features_image2 = len(pts2)  # Successfully tracked features
        metrics.raw_matches = num_detected
        metrics.ratio_test_matches = len(pts1)  # After forward-backward check
        metrics.cross_check_matches = len(pts1)
        
        print(f"  Features detected in image 1: {num_detected}")
        print(f"  Successfully tracked to image 2: {len(pts1)}")
        print(f"  Tracking success rate: {100*len(pts1)/max(1,num_detected):.1f}%")
        
        if num_detected < MIN_FEATURES_WARNING:
            metrics.warnings.append(f"Low feature count: {num_detected}")
        
        if len(pts1) < MIN_MATCHES_WARNING:
            metrics.warnings.append(f"Low tracking count: {len(pts1)}")
        
        # Check for critical failure
        if len(pts1) < MIN_INLIERS_CRITICAL:
            print("  [CRITICAL] Insufficient tracked features!")
            return self._create_identity_result(metrics, start_time)
        
        # Compute motion magnitude (average optical flow)
        flow = pts2 - pts1
        flow_magnitudes = np.linalg.norm(flow, axis=1)
        mean_flow = np.mean(flow_magnitudes)
        median_flow = np.median(flow_magnitudes)
        
        print(f"  Average optical flow: {mean_flow:.2f} px (median: {median_flow:.2f} px)")
        
        # Flag for small motion (translation direction may be unreliable)
        small_motion = median_flow < 3.0  # Less than 3 pixels median displacement
        if small_motion:
            print("  [WARNING] Small motion detected - translation direction may be less reliable")
            metrics.warnings.append("Small camera motion detected")
        
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
        
        # Secondary epipolar filtering - remove any remaining outliers
        # using Sampson error threshold
        if len(inlier_pts1) > MIN_INLIERS_CRITICAL:
            F_temp = self.geometry_estimator.K_inv.T @ E @ self.geometry_estimator.K_inv if E is not None else F
            sampson_errs = self.geometry_estimator.compute_sampson_error(F_temp, inlier_pts1, inlier_pts2)
            
            # Use adaptive threshold based on median absolute deviation
            median_err = np.median(sampson_errs)
            mad = np.median(np.abs(sampson_errs - median_err))
            threshold = median_err + 3.0 * mad * 1.4826  # 1.4826 converts MAD to std
            threshold = max(threshold, 1.5)  # Minimum threshold
            
            epipolar_good = sampson_errs < threshold
            if np.sum(epipolar_good) >= MIN_INLIERS_CRITICAL:
                inlier_pts1 = inlier_pts1[epipolar_good]
                inlier_pts2 = inlier_pts2[epipolar_good]
                print(f"  After epipolar refinement: {len(inlier_pts1)} points")
        
        # Apply cv2.correctMatches for sub-pixel refinement
        # This optimally triangulates points by adjusting them to satisfy epipolar constraint
        if len(inlier_pts1) >= MIN_INLIERS_CRITICAL and F is not None:
            try:
                pts1_refined, pts2_refined = cv2.correctMatches(
                    F, 
                    inlier_pts1.reshape(1, -1, 2),
                    inlier_pts2.reshape(1, -1, 2)
                )
                if pts1_refined is not None and pts2_refined is not None:
                    inlier_pts1 = pts1_refined.reshape(-1, 2)
                    inlier_pts2 = pts2_refined.reshape(-1, 2)
                    print(f"  Applied optimal triangulation correction")
            except cv2.error:
                pass  # Keep original points if correction fails
        
        # Re-estimate Essential matrix using cleaned inliers for more accuracy
        if len(inlier_pts1) >= 20:
            E_refined, mask_refined = self.geometry_estimator.estimate_essential_matrix_direct(
                inlier_pts1, inlier_pts2
            )
            if E_refined is not None and np.sum(mask_refined) >= MIN_INLIERS_CRITICAL:
                E = E_refined
                # Apply refined mask
                mask_refined = mask_refined.ravel().astype(bool)
                inlier_pts1 = inlier_pts1[mask_refined]
                inlier_pts2 = inlier_pts2[mask_refined]
                print(f"  Re-estimated E with {len(inlier_pts1)} cleaned inliers")
        
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
        # Stage 6: Motion Decomposition with Comprehensive Pose Selection
        # ---------------------------------------------------------------------
        print("\n[Stage 6] Motion Decomposition (R, t) with Comprehensive Selection...")
        
        # ALWAYS use comprehensive pose selection with triangulation quality
        # This evaluates all 4 solutions and picks the one with:
        # 1. Maximum points with positive depth
        # 2. Minimum reprojection error
        # 3. Best depth consistency
        R, t, num_positive = self.geometry_estimator.refine_pose_with_all_solutions(
            E, inlier_pts1, inlier_pts2
        )
        
        # Verify with OpenCV's recoverPose as sanity check
        R_cv, t_cv, num_cv = self.geometry_estimator.decompose_essential_matrix(
            E, inlier_pts1, inlier_pts2
        )
        
        # Compare results - take the better one
        if num_cv > num_positive:
            # OpenCV got more positive depth points
            qual_cv = self.geometry_estimator._evaluate_pose_quality(R_cv, t_cv, inlier_pts1, inlier_pts2)
            qual_ours = self.geometry_estimator._evaluate_pose_quality(R, t, inlier_pts1, inlier_pts2)
            
            # Compare using comprehensive quality: (num_valid, -reproj_err, -depth_consistency)
            score_cv = (qual_cv[0], -qual_cv[1], -qual_cv[2])
            score_ours = (qual_ours[0], -qual_ours[1], -qual_ours[2])
            
            if score_cv > score_ours:
                R, t, num_positive = R_cv, t_cv, num_cv
                print(f"  Using OpenCV pose (better quality)")
            else:
                print(f"  Keeping refined pose (better quality)")
        
        # ---------------------------------------------------------------------
        # Stage 6b: Translation Validation using Flow-Based Cross-Check
        # ---------------------------------------------------------------------
        t_flow = self.geometry_estimator.estimate_translation_from_flow(inlier_pts1, inlier_pts2, R)
        if t_flow is not None:
            # Check consistency between Essential matrix translation and flow-based
            t_flat = t.flatten()
            t_flow_flat = t_flow.flatten()
            
            # Cosine similarity - should be high if consistent
            cos_sim = abs(np.dot(t_flat, t_flow_flat) / (np.linalg.norm(t_flat) * np.linalg.norm(t_flow_flat) + 1e-10))
            
            print(f"  Translation validation: E-based vs flow-based cos_sim = {cos_sim:.4f}")
            
            # If low consistency, try negating translation (sign ambiguity)
            if cos_sim < 0.7:
                t_neg = -t
                cos_sim_neg = abs(np.dot(t_neg.flatten(), t_flow_flat) / 
                                 (np.linalg.norm(t_neg) * np.linalg.norm(t_flow_flat) + 1e-10))
                
                if cos_sim_neg > cos_sim:
                    # Negated translation matches flow better - check cheirality
                    num_neg = self.geometry_estimator._count_points_in_front(R, t_neg, inlier_pts1, inlier_pts2)
                    if num_neg >= num_positive * 0.9:  # Strict cheirality requirement
                        print(f"  Flipping translation sign (cos_sim: {cos_sim:.3f} -> {cos_sim_neg:.3f}, cheirality OK)")
                        t = t_neg
                        num_positive = num_neg
                    else:
                        print(f"  Translation flip rejected (cheirality failed: {num_neg} < {num_positive*0.9:.0f})")
        
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
