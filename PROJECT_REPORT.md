# Monocular Visual Odometry Project Report
 
Lia Bercovitz and Gilad Segal
---

## Abstract

This project presents a monocular visual odometry pipeline for estimating 6 Degrees of Freedom (6-DOF) camera motion between two consecutive images. Given two monocular JPG images and camera calibration parameters, the system computes a 3×3 rotation matrix **R** and a 3×1 translation direction vector **t**. The project evolved from an accuracy-focused implementation using SIFT features to a real-time system using Lucas-Kanade optical flow tracking, driven by the requirement to deploy on a Raspberry Pi. The final system achieves **85% accuracy on KITTI** and **58% on EuRoC** benchmarks while running in under 1 second on embedded hardware.

---

## 1. Introduction

### 1.1 Problem Definition

Visual odometry estimates camera motion by analyzing image sequences. Given two consecutive frames from a monocular camera and the camera intrinsic matrix **K**, we aim to recover:

- **R**: 3×3 rotation matrix describing orientation change
- **t**: 3×1 unit translation vector describing motion direction

The translation magnitude (scale) is unobservable in monocular systems—this is a fundamental limitation we accept.

### 1.2 Project Objectives

1. Develop a robust two-frame motion estimation pipeline
2. Achieve high accuracy on standard benchmarks
3. Enable real-time operation on Raspberry Pi Zero
4. Provide tools for evaluation and visualization

### 1.3 Target Platforms

- **Primary**: DJI Tello drone with Raspberry Pi Zero companion computer
- **Secondary**: Desktop evaluation on KITTI and EuRoC datasets

---

## 2. Algorithm Overview

The pipeline follows the classical feature-based visual odometry approach:

```
Image Pair → Preprocessing → Feature Detection → Optical Flow Tracking → 
Essential Matrix → Motion Decomposition → (R, t)
```

### 2.1 Preprocessing

Images are downsampled to 640×480 with anti-aliasing to reduce computation while preserving geometric information. We apply:

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Improves feature detection in low-contrast regions
- **Bilateral filtering**: Edge-preserving noise reduction

The camera matrix **K** is scaled proportionally to maintain geometric consistency.

### 2.2 Feature Detection and Tracking

We use **Shi-Tomasi corner detection** to find trackable features, followed by **Lucas-Kanade optical flow** for tracking:

**Corner Detection (Shi-Tomasi):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max corners | 2000 | Sufficient coverage |
| Quality level | 0.01 | Sensitive detection |
| Min distance | 7 px | Avoid clustering |

**Optical Flow (Lucas-Kanade):**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window size | 21×21 | Balance accuracy/speed |
| Pyramid levels | 3 | Handle larger motions |
| Forward-backward threshold | 1.0 px | Validation check |

The forward-backward consistency check tracks points from image 1 to image 2, then back to image 1. Points with round-trip error > 1 pixel are rejected as unreliable.

### 2.3 Essential Matrix Estimation

We estimate the Essential matrix **E** directly using **USAC_MAGSAC** (OpenCV 4.5+), which automatically determines optimal inlier thresholds through marginalization. This significantly outperforms classical RANSAC.

| Parameter | Value |
|-----------|-------|
| Method | USAC_MAGSAC |
| Confidence | 0.9999 |
| Max iterations | 15,000 |
| Threshold | 0.75 pixels |

The Essential matrix is constrained to have singular values (σ, σ, 0) via SVD correction.

### 2.4 Motion Decomposition

The Essential matrix yields four possible (R, t) solutions. We select the correct one using:

1. **Cheirality check**: Triangulated points must have positive depth in both cameras
2. **Reprojection error**: Lower error indicates better solution
3. **Depth consistency**: Valid solutions have consistent depth distribution

Additionally, we validate translation direction using optical flow analysis to resolve sign ambiguity.

### 2.5 Quality Scoring

The system computes a quality score (0-100) based on:
- Feature count (penalty if < 100)
- Match count after filtering (penalty if < 50)
- RANSAC inlier ratio (penalty if < 30%)
- Reprojection error (penalty if > 1 pixel)

This score enables downstream systems to assess result reliability.

---

## 3. Development Path

### 3.1 Initial Design: Accuracy First

The project began with accuracy as the sole priority, without considering computational constraints.

**Initial choices:**
- **SIFT features**: 128-dimensional floating-point descriptors, highly distinctive
- **FLANN matching**: Approximate nearest neighbor for efficiency with SIFT
- **High feature count**: 10,000+ features per image
- **Extensive refinement**: Multiple stages of outlier rejection

**Results**: High accuracy on benchmarks, but **processing time exceeded 2 seconds per frame** on desktop hardware.

### 3.2 The Raspberry Pi Constraint

Mid-project, we decided to deploy on a Raspberry Pi Zero for the DJI Tello application. This introduced strict constraints:

- **CPU**: Single-core 1GHz ARM
- **RAM**: 512MB
- **Target**: < 1 second per frame pair

SIFT was completely infeasible—a single frame required 5+ seconds on the Pi.

### 3.3 Migration to Optical Flow

We replaced SIFT descriptor matching with Lucas-Kanade optical flow tracking:

| Aspect | SIFT Matching | Optical Flow |
|--------|---------------|---------------|
| Approach | Detect + Describe + Match | Detect + Track |
| Computation | Heavy (128D descriptors) | Light (local search) |
| Detection time | ~100ms | ~10ms |
| Matching/Tracking time | ~50ms | ~15ms |
| Robustness | Scale/rotation invariant | Requires small motion |

**Key insight**: For consecutive video frames, motion between frames is typically small (< 50 pixels). Optical flow exploits this assumption for faster and often more accurate correspondence.

**Validation**: We use forward-backward consistency checking—track points forward, then backward, and reject points where the round-trip error exceeds 1 pixel. This provides robust outlier rejection without needing descriptor matching.

### 3.4 RANSAC to USAC_MAGSAC

Classical RANSAC with fixed thresholds performed inconsistently across different motion magnitudes. We adopted USAC_MAGSAC, which:

- Marginalizes over possible thresholds
- Adapts to varying noise levels
- Provides ~5% accuracy improvement at similar speed

### 3.5 What Did Not Work

Several approaches were tried and abandoned:

1. **Fundamental matrix first, then E = K'FK**: Less accurate than direct Essential matrix estimation with calibrated points

2. **Aggressive downsampling (320×240)**: Too few features survived, especially in low-texture regions

3. **ORB descriptor matching**: Tested as an intermediate step between SIFT and optical flow. While faster than SIFT, the binary descriptors produced more false matches than optical flow with forward-backward validation

4. **Homography-based degeneracy rejection**: Useful for detection but not for correction; we kept it only as a warning flag

### 3.6 Final Architecture

The final system has two implementations:

1. **Full pipeline** (`camera_motion_estimator.py`): All features, diagnostics, and fallbacks for evaluation
2. **Lightweight CLI** (`estimate_motion.py`): Minimal code for Raspberry Pi deployment

---

## 4. Evaluation

### 4.1 Datasets

| Dataset | Platform | Resolution | Motion Type |
|---------|----------|------------|-------------|
| **KITTI** | Ground vehicle | 1241×376 | Forward driving |
| **EuRoC** | MAV drone | 752×480 | 6-DOF flight |

### 4.2 Metrics

- **Rotation error**: Angle between estimated and ground truth rotation (degrees)
- **Translation error**: Angle between estimated and ground truth translation direction (degrees)
- **Combined accuracy**: Percentage of pairs with R < 5° AND t < 15°

### 4.3 Results

#### KITTI Odometry (Sequence 00)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| Rotation error | 0.56° | 0.39° | 0.65° |
| Translation error | 9.5° | 3.3° | 14.5° |

| Threshold | Rotation | Translation | Combined |
|-----------|----------|-------------|----------|
| Strict (R<2°, t<5°) | 95% | 75% | **75%** |
| Normal (R<5°, t<15°) | 100% | 85% | **85%** |
| Relaxed (R<10°, t<30°) | 100% | 90% | **90%** |

#### EuRoC MAV (MH_01_easy)

| Metric | Mean | Median | Std |
|--------|------|--------|-----|
| Rotation error | 3.29° | 2.90° | 1.91° |
| Translation error | 13.7° | 10.6° | 13.2° |

| Threshold | Rotation | Translation | Combined |
|-----------|----------|-------------|----------|
| Strict (R<2°, t<5°) | 23% | 19% | **15%** |
| Normal (R<5°, t<15°) | 88% | 71% | **58%** |
| Relaxed (R<10°, t<30°) | 100% | 94% | **94%** |

**Observations:**
- KITTI results are excellent due to predominantly forward motion
- EuRoC is more challenging due to aggressive 6-DOF maneuvers
- Translation direction is harder to estimate than rotation (expected for monocular VO)
- **100% robustness**: No pipeline failures on either dataset

### 4.4 Processing Time

| Platform | Time per pair |
|----------|---------------|
| Desktop (Intel i7) | 80-140 ms |
| Raspberry Pi Zero | 500-1000 ms |

The Raspberry Pi target of < 1 second was achieved.

---

## 5. Project Structure

| File | Purpose |
|------|---------|
| `camera_motion_estimator.py` | Main pipeline with full diagnostics |
| `estimate_motion.py` | Lightweight CLI for embedded deployment |
| `evaluate_euroc.py` | EuRoC benchmark evaluation |
| `evaluate_kitti.py` | KITTI benchmark evaluation |
| `evaluation_metrics.py` | ATE/RPE metric implementations |
| `visualize_results.py` | Result plotting tools |
| `quick_test.py` | Automated test suite |

---

## 6. Usage

### Basic Estimation
```bash
python camera_motion_estimator.py image1.jpg image2.jpg calibration.json
```

### Raspberry Pi Deployment
```bash
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json --json
```

### Benchmark Evaluation
```bash
python evaluate_kitti.py kitti_odometry --sequence 00 --num_pairs 100
python evaluate_euroc.py euroc/MH_01_easy/mav0 --num_pairs 50 --flip_z
```

---

## 7. Demonstration

A video demonstrating real-time execution on a Raspberry Pi Zero is included with this submission. The video shows:

1. Live camera feed from DJI Tello
2. Real-time motion estimation at ~1 Hz
3. Rotation and translation output overlay

---

## 8. Conclusion

We developed a monocular visual odometry system that estimates 6-DOF camera motion from image pairs. The project evolved from an accuracy-focused SIFT-based design to a real-time ORB-based system suitable for embedded deployment.

**Key achievements:**
- 85% combined accuracy on KITTI, 58% on EuRoC
- Real-time operation on Raspberry Pi Zero (< 1 second)
- 100% robustness (no failures)
- Comprehensive evaluation and visualization tools

**Key trade-offs:**
- ORB vs SIFT: 10× speed improvement with ~5% accuracy reduction
- Resolution: 640×480 balances feature count and computation
- Ratio test threshold: 0.6 provides optimal precision/recall

The system successfully meets its objectives of accurate, robust, and real-time monocular visual odometry for mobile robotics applications.

---

## References

1. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
2. Nistér, D. (2004). An efficient solution to the five-point relative pose problem. *IEEE TPAMI*.
3. Rublee, E., et al. (2011). ORB: An efficient alternative to SIFT or SURF. *ICCV*.
4. Raguram, R., et al. (2013). USAC: A universal framework for random sample consensus. *IEEE TPAMI*.
5. Geiger, A., et al. (2012). Are we ready for autonomous driving? The KITTI vision benchmark. *CVPR*.
6. Burri, M., et al. (2016). The EuRoC micro aerial vehicle datasets. *IJRR*.
