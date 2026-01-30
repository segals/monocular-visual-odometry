#!/usr/bin/env python3
"""
Lightweight 6-DOF Camera Motion Estimator CLI for DJI Tello / Raspberry Pi Zero

Usage:
    python estimate_motion.py <image1> <image2> <calibration.json>
    python estimate_motion.py <image1> <image2> <calibration.json> --json
    python estimate_motion.py <image1> <image2> <calibration.json> --verbose

Output (default):
    R: rx,ry,rz (Rodrigues rotation vector)
    t: tx,ty,tz (normalized translation direction)
    
Output (--json):
    {"R": [...], "t": [...], "quality": ..., "inliers": ...}

Example:
    python estimate_motion.py frame001.jpg frame002.jpg tello_calib.json
    
Raspberry Pi Zero one-liner:
    python3 estimate_motion.py "$IMG1" "$IMG2" calib.json

Author: Camera Motion Estimation Project
License: MIT
"""

import sys
import json
import argparse
import numpy as np

# Minimal imports for faster startup on Pi Zero
try:
    import cv2
except ImportError:
    print("Error: OpenCV not installed. Run: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# CONSTANTS (matching camera_motion_estimator.py)
# =============================================================================
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = 8
RATIO_TEST_THRESHOLD = 0.6
RANSAC_THRESHOLD = 0.75
MIN_INLIERS = 15


def load_calibration(calib_path: str) -> tuple:
    """Load camera calibration from JSON file."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    K = np.array(calib['camera_matrix'], dtype=np.float64)
    dist = np.array(calib.get('dist_coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)
    orig_w = calib.get('image_width', 960)
    orig_h = calib.get('image_height', 720)
    
    return K, dist, orig_w, orig_h


def preprocess_image(image_path: str, K: np.ndarray, dist: np.ndarray,
                     orig_w: int, orig_h: int, preserve_aspect: bool = False) -> tuple:
    """Load and preprocess image for feature detection."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Calculate downsampling scale
    if preserve_aspect:
        # Maintain aspect ratio - scale to fit within target dimensions
        scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
        scale_x = scale_y = scale
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        # Force exact target dimensions (Tello mode)
        scale_x = TARGET_WIDTH / w
        scale_y = TARGET_HEIGHT / h
        new_w = TARGET_WIDTH
        new_h = TARGET_HEIGHT
    
    # Apply anti-aliasing blur before downsampling
    if scale_x < 1.0 or scale_y < 1.0:
        blur_size = max(3, int(1.0 / min(scale_x, scale_y)) | 1)
        if blur_size > 1:
            img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    
    # Downsample
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILE_SIZE, CLAHE_TILE_SIZE))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Scale calibration matrix
    K_scaled = K.copy()
    K_scaled[0, :] *= scale_x
    K_scaled[1, :] *= scale_y
    
    return gray, K_scaled


def detect_and_match(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """Detect ORB features and match with ratio test."""
    # ORB detector
    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        patchSize=31,
        fastThreshold=20
    )
    
    # Detect and compute
    kp1, desc1 = orb.detectAndCompute(img1, None)
    kp2, desc2 = orb.detectAndCompute(img2, None)
    
    if desc1 is None or desc2 is None or len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS:
        return None, None, 0
    
    # BFMatcher with ratio test
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < MIN_INLIERS:
        return None, None, 0
    
    # Extract matched points
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)
    
    return pts1, pts2, len(good_matches)


def estimate_motion(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> tuple:
    """Estimate Essential matrix and recover pose."""
    # Find Essential matrix using USAC_MAGSAC
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.USAC_MAGSAC,
        prob=0.999,
        threshold=RANSAC_THRESHOLD
    )
    
    if E is None:
        return None, None, 0, 0.0
    
    # Count inliers
    inlier_mask = mask.ravel() == 1
    num_inliers = np.sum(inlier_mask)
    
    if num_inliers < MIN_INLIERS:
        return None, None, num_inliers, 0.0
    
    # Recover pose
    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    # Calculate quality score
    quality = min(100.0, (num_inliers / 50.0) * 100.0)
    
    return R, t, num_inliers, quality


def main():
    parser = argparse.ArgumentParser(
        description='6-DOF Camera Motion Estimator for DJI Tello / Raspberry Pi',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json
    python estimate_motion.py img1.png img2.png calib.json --json
    python estimate_motion.py "$IMG1" "$IMG2" calib.json --verbose
        """
    )
    parser.add_argument('image1', type=str, help='Path to first image')
    parser.add_argument('image2', type=str, help='Path to second image')
    parser.add_argument('calibration', type=str, help='Path to calibration JSON file')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--preserve-aspect', '-p', action='store_true', 
                        help='Preserve aspect ratio (use for non-Tello cameras like KITTI)')
    
    args = parser.parse_args()
    
    try:
        # Load calibration
        K, dist, orig_w, orig_h = load_calibration(args.calibration)
        
        if args.verbose:
            print(f"Loaded calibration: {args.calibration}", file=sys.stderr)
            print(f"  Original resolution: {orig_w}x{orig_h}", file=sys.stderr)
        
        # Preprocess images
        img1, K_scaled = preprocess_image(args.image1, K, dist, orig_w, orig_h, args.preserve_aspect)
        img2, _ = preprocess_image(args.image2, K, dist, orig_w, orig_h, args.preserve_aspect)
        
        if args.verbose:
            print(f"Preprocessed images to {TARGET_WIDTH}x{TARGET_HEIGHT}", file=sys.stderr)
        
        # Detect and match features
        pts1, pts2, num_matches = detect_and_match(img1, img2)
        
        if pts1 is None:
            if args.json:
                print(json.dumps({"error": "Not enough features", "matches": num_matches}))
            else:
                print("ERROR: Not enough features detected", file=sys.stderr)
            sys.exit(1)
        
        if args.verbose:
            print(f"Matched features: {num_matches}", file=sys.stderr)
        
        # Estimate motion
        R, t, num_inliers, quality = estimate_motion(pts1, pts2, K_scaled)
        
        if R is None:
            if args.json:
                print(json.dumps({"error": "Motion estimation failed", "inliers": num_inliers}))
            else:
                print("ERROR: Motion estimation failed", file=sys.stderr)
            sys.exit(1)
        
        # Convert rotation matrix to Rodrigues vector
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.flatten()
        tvec = t.flatten()
        
        if args.verbose:
            print(f"Inliers: {num_inliers}, Quality: {quality:.1f}", file=sys.stderr)
        
        # Output
        if args.json:
            result = {
                "R": rvec.tolist(),
                "t": tvec.tolist(),
                "R_matrix": R.tolist(),
                "quality": quality,
                "inliers": int(num_inliers),
                "matches": num_matches
            }
            print(json.dumps(result))
        else:
            # Compact output for piping
            print(f"R: {rvec[0]:.6f},{rvec[1]:.6f},{rvec[2]:.6f}")
            print(f"t: {tvec[0]:.6f},{tvec[1]:.6f},{tvec[2]:.6f}")
            print(f"Q: {quality:.1f}")
            print(f"I: {num_inliers}")
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
