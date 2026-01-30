# quick_test.py - Quick evaluation test
import sys
import os
sys.path.insert(0, r"C:\Users\gilad\OneDrive\Documents\Etgar\Dan'sLab")
os.chdir(r"C:\Users\gilad\OneDrive\Documents\Etgar\Dan'sLab")

import numpy as np
import cv2
from pathlib import Path
from camera_motion_estimator import CameraMotionEstimator

# Calibration file path
calib_path = 'euroc_test/machine_hall/MH_01_easy/calibration_cam0.json'

print("Initializing estimator with calibration file:", calib_path)

# Initialize estimator with calibration file path
estimator = CameraMotionEstimator(calib_path)

# Load two images
img_dir = Path('euroc_test/machine_hall/MH_01_easy/mav0/cam0/data')
imgs = sorted(list(img_dir.glob('*.png')))
print(f"Found {len(imgs)} images")

# Test pair
idx1, idx2 = 125, 130
img1_path = str(imgs[idx1])
img2_path = str(imgs[idx2])

print(f"Testing images {idx1} and {idx2}")

# Use the estimate() method with image paths
result = estimator.estimate(img1_path, img2_path)
print(f"Quality score: {result.metrics.quality_score:.1f}")
print(f"Inliers: {result.metrics.ransac_inliers}")
print(f"Mean reproj error: {result.metrics.mean_reprojection_error:.2f}")

# Rotation angle
angle = np.arccos(np.clip((np.trace(result.R) - 1) / 2, -1, 1))
print(f"Rotation angle: {np.degrees(angle):.3f} degrees")
print(f"Translation: {result.t.flatten()}")
