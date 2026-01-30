#!/usr/bin/env python3
"""
Unified Test Runner for Camera Motion Estimator

Runs quick validation tests on all available datasets and generates a summary report.

Usage:
    python quick_test.py                    # Run all available tests
    python quick_test.py --euroc            # Run only EuRoC test
    python quick_test.py --kitti            # Run only KITTI test
    python quick_test.py --synthetic        # Run synthetic test
    python quick_test.py --all --verbose    # Run all with detailed output
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, details: str = ""):
    """Print test result with status icon."""
    icon = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{icon}] {name}: {status}")
    if details:
        print(f"      {details}")


def test_opencv_version() -> bool:
    """Test OpenCV version compatibility."""
    version = cv2.__version__
    major, minor = map(int, version.split('.')[:2])
    
    # Check for USAC_MAGSAC support (requires 4.5+)
    has_magsac = hasattr(cv2, 'USAC_MAGSAC')
    
    print(f"  OpenCV version: {version}")
    print(f"  USAC_MAGSAC support: {'Yes' if has_magsac else 'No'}")
    
    return major >= 4 and minor >= 5


def test_imports() -> bool:
    """Test that all required modules can be imported."""
    try:
        from camera_motion_estimator import CameraMotionEstimator
        from evaluation_metrics import compute_ate, compute_rpe
        return True
    except ImportError as e:
        print(f"  Import error: {e}")
        return False


def test_synthetic() -> dict:
    """Run synthetic test with known motion."""
    print_header("Synthetic Test (Known Ground Truth)")
    
    results = {'name': 'synthetic', 'passed': False, 'details': {}}
    
    try:
        from camera_motion_estimator import CameraMotionEstimator, ImagePreprocessor
        
        # Create synthetic calibration
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Create synthetic 3D points (random scene)
        np.random.seed(42)
        n_points = 100
        points_3d = np.random.rand(n_points, 3) * 10
        points_3d[:, 2] += 5  # Push points away from camera
        
        # Camera 1: identity
        R1 = np.eye(3)
        t1 = np.zeros(3)
        
        # Camera 2: known rotation and translation
        angle = np.radians(5)  # 5 degree rotation around Y
        R2 = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        t2 = np.array([0.1, 0.0, 0.05])  # Small translation
        
        # Project points to both cameras
        def project(pts_3d, R, t, K):
            pts_cam = (R @ pts_3d.T).T + t
            pts_2d = (K @ pts_cam.T).T
            pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]
            return pts_2d
        
        pts1 = project(points_3d, R1, t1, K)
        pts2 = project(points_3d, R2, t2, K)
        
        # Filter points within image bounds
        valid = (pts1[:, 0] > 0) & (pts1[:, 0] < 640) & \
                (pts1[:, 1] > 0) & (pts1[:, 1] < 480) & \
                (pts2[:, 0] > 0) & (pts2[:, 0] < 640) & \
                (pts2[:, 1] > 0) & (pts2[:, 1] < 480)
        
        pts1 = pts1[valid].astype(np.float32)
        pts2 = pts2[valid].astype(np.float32)
        
        print(f"  Valid point correspondences: {len(pts1)}")
        
        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.USAC_MAGSAC,
                                        prob=0.999, threshold=1.0)
        
        if E is None:
            print_result("Essential Matrix", False, "Failed to estimate")
            return results
        
        # Recover pose
        _, R_est, t_est, _ = cv2.recoverPose(E, pts1, pts2, K)
        t_est = t_est.flatten()
        
        # Ground truth relative pose
        R_gt = R2 @ R1.T
        t_gt = t2 - R_gt @ t1
        t_gt = t_gt / np.linalg.norm(t_gt)  # Normalize
        
        # Compute errors
        R_diff = R_est @ R_gt.T
        trace = np.clip(np.trace(R_diff), -1, 3)
        r_err = np.degrees(np.arccos((trace - 1) / 2))
        
        t_err = np.degrees(np.arccos(np.clip(np.abs(np.dot(t_est, t_gt)), -1, 1)))
        
        print(f"  Rotation error: {r_err:.3f}°")
        print(f"  Translation error: {t_err:.3f}°")
        
        passed = r_err < 1.0 and t_err < 5.0
        print_result("Synthetic Motion", passed, 
                    f"R_err={r_err:.2f}°, t_err={t_err:.2f}°")
        
        results['passed'] = passed
        results['details'] = {
            'rotation_error': r_err,
            'translation_error': t_err,
            'num_points': len(pts1)
        }
        
    except Exception as e:
        print_result("Synthetic Test", False, str(e))
        results['error'] = str(e)
    
    return results


def test_euroc() -> dict:
    """Run quick EuRoC evaluation."""
    print_header("EuRoC MAV Dataset Test")
    
    results = {'name': 'euroc', 'passed': False, 'details': {}}
    
    euroc_path = Path("euroc_test/machine_hall/MH_01_easy/mav0")
    
    if not euroc_path.exists():
        print("  Dataset not found, skipping...")
        results['skipped'] = True
        return results
    
    try:
        from camera_motion_estimator import CameraMotionEstimator
        
        # Find calibration
        calib_path = euroc_path.parent / "calibration_cam0.json"
        if not calib_path.exists():
            print(f"  Calibration not found: {calib_path}")
            results['error'] = "Calibration not found"
            return results
        
        # Initialize estimator
        estimator = CameraMotionEstimator(str(calib_path), preserve_aspect_ratio=False)
        
        # Get a few image pairs
        image_dir = euroc_path / "cam0" / "data"
        images = sorted(image_dir.glob("*.png"))[:20]  # First 20 images
        
        if len(images) < 2:
            print("  Not enough images found")
            results['error'] = "Not enough images"
            return results
        
        print(f"  Found {len(images)} images for quick test")
        
        # Test a few pairs
        successes = 0
        total = min(5, len(images) - 1)
        
        for i in range(total):
            try:
                result = estimator.estimate(str(images[i]), str(images[i+1]))
                if result.quality_score > 50:
                    successes += 1
                print(f"    Pair {i+1}: Q={result.quality_score:.1f}, inliers={result.num_inliers}")
            except Exception as e:
                print(f"    Pair {i+1}: Error - {str(e)[:40]}")
        
        success_rate = successes / total * 100
        passed = success_rate >= 60
        
        print_result("EuRoC Quick Test", passed, 
                    f"{successes}/{total} pairs succeeded ({success_rate:.0f}%)")
        
        results['passed'] = passed
        results['details'] = {
            'successes': successes,
            'total': total,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print_result("EuRoC Test", False, str(e))
        results['error'] = str(e)
    
    return results


def test_kitti() -> dict:
    """Run quick KITTI evaluation."""
    print_header("KITTI Odometry Dataset Test")
    
    results = {'name': 'kitti', 'passed': False, 'details': {}}
    
    kitti_path = Path("kitti_odometry")
    
    if not kitti_path.exists() or not (kitti_path / "sequences").exists():
        print("  Dataset not found, skipping...")
        results['skipped'] = True
        return results
    
    # Check for sequence 00
    seq_path = kitti_path / "sequences" / "00"
    if not seq_path.exists():
        print("  Sequence 00 not found, skipping...")
        results['skipped'] = True
        return results
    
    try:
        from camera_motion_estimator import CameraMotionEstimator
        from evaluate_kitti import load_kitti_calibration, create_kitti_calibration_json
        
        # Find images
        image_dir = None
        for d in ['image_0', 'image_1', 'image_2', 'image_3']:
            if (seq_path / d).exists():
                image_dir = seq_path / d
                break
        
        if image_dir is None:
            print("  No image directory found")
            results['error'] = "No images"
            return results
        
        images = sorted(image_dir.glob("*.png"))[:20]
        print(f"  Found {len(images)} images for quick test")
        
        # Create calibration
        calib_path = seq_path / "calib.txt"
        calib_json = seq_path / "calibration.json"
        
        sample_img = cv2.imread(str(images[0]))
        h, w = sample_img.shape[:2]
        
        create_kitti_calibration_json(calib_path, calib_json, w, h)
        
        # Initialize estimator
        estimator = CameraMotionEstimator(str(calib_json), preserve_aspect_ratio=True)
        
        # Test a few pairs
        successes = 0
        total = min(5, len(images) - 1)
        
        for i in range(total):
            try:
                result = estimator.estimate(str(images[i]), str(images[i+1]))
                if result.quality_score > 50:
                    successes += 1
                print(f"    Pair {i+1}: Q={result.quality_score:.1f}, inliers={result.num_inliers}")
            except Exception as e:
                print(f"    Pair {i+1}: Error - {str(e)[:40]}")
        
        success_rate = successes / total * 100
        passed = success_rate >= 60
        
        print_result("KITTI Quick Test", passed,
                    f"{successes}/{total} pairs succeeded ({success_rate:.0f}%)")
        
        results['passed'] = passed
        results['details'] = {
            'successes': successes,
            'total': total,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print_result("KITTI Test", False, str(e))
        results['error'] = str(e)
    
    return results


def test_cli_tool() -> dict:
    """Test the lightweight CLI tool."""
    print_header("CLI Tool Test (estimate_motion.py)")
    
    results = {'name': 'cli_tool', 'passed': False, 'details': {}}
    
    cli_path = Path("estimate_motion.py")
    if not cli_path.exists():
        print("  CLI tool not found")
        results['error'] = "Not found"
        return results
    
    try:
        # Just check that it imports correctly
        import importlib.util
        spec = importlib.util.spec_from_file_location("estimate_motion", cli_path)
        module = importlib.util.module_from_spec(spec)
        
        # Check key functions exist
        spec.loader.exec_module(module)
        
        has_funcs = all(hasattr(module, f) for f in 
                       ['load_calibration', 'preprocess_image', 'detect_and_match', 'estimate_motion'])
        
        print_result("CLI Tool Import", has_funcs, "All functions available" if has_funcs else "Missing functions")
        
        results['passed'] = has_funcs
        
    except Exception as e:
        print_result("CLI Tool", False, str(e))
        results['error'] = str(e)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run quick validation tests')
    parser.add_argument('--euroc', action='store_true', help='Run only EuRoC test')
    parser.add_argument('--kitti', action='store_true', help='Run only KITTI test')
    parser.add_argument('--synthetic', action='store_true', help='Run only synthetic test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # If no specific test selected, run all
    run_all = args.all or not (args.euroc or args.kitti or args.synthetic)
    
    print_header("Camera Motion Estimator - Quick Test Suite")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    
    start_time = time.time()
    all_results = []
    
    # Environment checks
    print_header("Environment Check")
    cv_ok = test_opencv_version()
    import_ok = test_imports()
    print_result("OpenCV 4.5+", cv_ok)
    print_result("Module Imports", import_ok)
    
    if not cv_ok or not import_ok:
        print("\n⚠️  Environment issues detected. Some tests may fail.")
    
    # Run tests
    if run_all or args.synthetic:
        all_results.append(test_synthetic())
    
    if run_all or args.euroc:
        all_results.append(test_euroc())
    
    if run_all or args.kitti:
        all_results.append(test_kitti())
    
    if run_all:
        all_results.append(test_cli_tool())
    
    # Summary
    elapsed = time.time() - start_time
    print_header("Test Summary")
    
    passed = sum(1 for r in all_results if r.get('passed', False))
    skipped = sum(1 for r in all_results if r.get('skipped', False))
    failed = len(all_results) - passed - skipped
    
    print(f"  Total:   {len(all_results)}")
    print(f"  Passed:  {passed} ✓")
    print(f"  Failed:  {failed} ✗")
    print(f"  Skipped: {skipped} -")
    print(f"  Time:    {elapsed:.2f}s")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'opencv_version': cv2.__version__,
        'python_version': sys.version.split()[0],
        'results': all_results,
        'summary': {
            'total': len(all_results),
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'elapsed_seconds': elapsed
        }
    }
    
    with open('quick_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: quick_test_results.json")
    
    # Exit code
    if failed > 0:
        print("\n⚠️  Some tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
