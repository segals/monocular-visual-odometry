# Monocular Visual Odometry - 6-DOF Camera Motion Estimator

A high-precision Python pipeline for estimating 6 Degrees of Freedom (6-DOF) camera motion between two monocular images. Designed for mobile robotics applications including drones (DJI Tello) and ground vehicles.

## 🎯 Benchmark Results

| Dataset | Sequence | Rotation Accuracy | Translation Accuracy | Combined Accuracy |
|---------|----------|-------------------|---------------------|-------------------|
| **KITTI** | 00 | 100% (<5°) | 85% (<15°) | **85%** (R<5°, t<15°) |
| **EuRoC** | MH_01_easy | 87.5% (<5°) | 70.8% (<15°) | **58.3%** (R<5°, t<15°) |

## Features

- **High Accuracy Focus**: Optimized for precision over speed
- **Robust Preprocessing**: CLAHE enhancement, bilateral filtering, automatic downsampling to 640×480
- **ORB Feature Detection**: Configured for maximum feature coverage (5000 features, 8 pyramid levels)
- **Ratio Test Matching**: Lowe's ratio test (0.6 threshold) with cross-check validation
- **USAC_MAGSAC**: State-of-the-art robust estimation (OpenCV 4.5+) with 15K iterations
- **Essential Matrix Decomposition**: SVD-based with proper constraint enforcement
- **Cheirality Verification**: Automatic selection of correct (R, t) solution from 4 candidates
- **Multi-Dataset Support**: EuRoC MAV, KITTI Odometry, DJI Tello
- **Raspberry Pi Zero Compatible**: Lightweight CLI tool included
- **Quality Scoring**: Automated quality assessment (0-100 scale)
- **Comprehensive Evaluation**: ATE/RPE metrics, visualization tools

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.5+ (for USAC_MAGSAC support)
- NumPy 1.20+
- SciPy (for evaluation scripts)
- matplotlib (optional, for visualization)

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Full Pipeline)

```bash
python camera_motion_estimator.py <image1> <image2> <calibration_file> [--output_dir <dir>]
```

### Raspberry Pi / DJI Tello (Lightweight CLI)

```bash
# Simple output (Rodrigues rotation vector + translation direction)
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json

# JSON output (for parsing)
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json --json

# Verbose mode with quality metrics
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json --verbose
```

### Dataset Evaluation

```bash
# EuRoC MAV Dataset
python evaluate_euroc.py euroc_test/machine_hall/MH_01_easy/mav0 \
    --num_pairs 50 --step 5 --flip_z

# KITTI Odometry Dataset  
python evaluate_kitti.py kitti_odometry --sequence 00 --num_pairs 20 --step 1

# Visualize Results
python visualize_results.py euroc_evaluation.json --show
python visualize_results.py kitti_evaluation.json --output plots/
```

### Quick Testing

```bash
# Run all available tests with summary report
python quick_test.py --all --verbose

# Run specific dataset tests
python quick_test.py --euroc
python quick_test.py --kitti
python quick_test.py --synthetic
```

## Input

### Images
- Two JPEG/PNG images (consecutive frames from video)
- Supports any resolution (automatically downsampled to 640×480)
- Aspect ratio preserved for KITTI-style datasets

### Calibration File (JSON format)

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "image_width": 960,
  "image_height": 720
}
```

Supported formats:
- **JSON** (`.json`): Recommended format with full metadata
- **NumPy** (`.npy`): 3×3 camera matrix array
- **Text** (`.txt`): Space/comma-separated 3×3 matrix

### Example Calibration Files

**DJI Tello (tello_calibration_template.json):**
```json
{
  "camera_matrix": [[925.0, 0.0, 480.0], [0.0, 925.0, 360.0], [0.0, 0.0, 1.0]],
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
  "image_width": 960,
  "image_height": 720
}
```

**KITTI Odometry:**
```json
{
  "camera_matrix": [[718.856, 0.0, 607.1928], [0.0, 718.856, 185.2157], [0.0, 0.0, 1.0]],
  "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
  "image_width": 1241,
  "image_height": 376
}
```

## Output

### Files Generated (with --output_dir)

| File | Description |
|------|-------------|
| `motion_R.npy` | 3×3 Rotation matrix |
| `motion_t.npy` | 3×1 Translation vector (unit norm) |
| `motion_F.npy` | 3×3 Fundamental matrix |
| `motion_E.npy` | 3×3 Essential matrix |
| `motion_summary.txt` | Human-readable summary |
| `motion_result.json` | Complete results in JSON format |

### CLI Output (estimate_motion.py)

**Default format:**
```
R: 0.0123,0.0456,0.0789
t: 0.123,0.456,0.789
```

**JSON format (--json):**
```json
{"R": [...], "t": [...], "quality": 85.0, "inliers": 340}
```

## Detailed Benchmark Results

### KITTI Odometry Dataset (Sequence 00)

| Metric | Threshold | Accuracy |
|--------|-----------|----------|
| **Rotation** | < 1° | **90%** |
| **Rotation** | < 2° | **95%** |
| **Rotation** | < 5° | **100%** |
| **Translation Direction** | < 5° | **75%** |
| **Translation Direction** | < 10° | **75%** |
| **Translation Direction** | < 15° | **85%** |
| **Combined (R<2°, t<5°)** | Strict | **75%** |
| **Combined (R<5°, t<15°)** | Normal | **85%** |
| **Combined (R<10°, t<30°)** | Relaxed | **90%** |

**Error Statistics:**
- Rotation Error: Mean=0.56°, Median=0.39°, Std=0.65°
- Translation Direction Error: Mean=9.5°, Median=3.3°, Std=14.5°

### EuRoC MAV Dataset (MH_01_easy)

| Metric | Threshold | Accuracy |
|--------|-----------|----------|
| **Rotation** | < 1° | **4.2%** |
| **Rotation** | < 2° | **22.9%** |
| **Rotation** | < 5° | **87.5%** |
| **Translation Direction** | < 5° | **18.8%** |
| **Translation Direction** | < 10° | **47.9%** |
| **Translation Direction** | < 15° | **70.8%** |
| **Combined (R<2°, t<5°)** | Strict | **14.6%** |
| **Combined (R<5°, t<15°)** | Normal | **58.3%** |
| **Combined (R<10°, t<30°)** | Relaxed | **93.8%** |

**Error Statistics:**
- Rotation Error: Mean=3.3°, Median=2.9°, Std=1.9°
- Translation Direction Error: Mean=13.7°, Median=10.6°, Std=13.2°
- 100% success rate (no OpenCV exceptions)

*Note: Translation is direction-only; scale is unobservable in monocular VO*

## Algorithm Pipeline

```
Input Images + Calibration Matrix K
              │
              ▼
┌─────────────────────────────────────┐
│   1. IMAGE PREPROCESSING            │
│   • Downsample to 640×480           │
│   • Gaussian blur (anti-aliasing)   │
│   • CLAHE enhancement (clip=2.5)    │
│   • Bilateral filtering (d=5)       │
│   • Scale calibration matrix K      │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   2. FEATURE DETECTION              │
│   • ORB detector (5000 features)    │
│   • 8 pyramid levels (scale=1.2)    │
│   • FAST corner threshold: 20       │
│   • Harris score ranking            │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   3. FEATURE MATCHING               │
│   • Brute-force Hamming distance    │
│   • Lowe's ratio test (0.6)         │
│   • Maximum Hamming distance: 45    │
│   • Cross-check validation          │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   4. ESSENTIAL MATRIX ESTIMATION    │
│   • USAC_MAGSAC (15K iterations)    │
│   • Confidence: 0.9999              │
│   • Threshold: 0.75 px              │
│   • Fallback: F → E if needed       │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│   5. MOTION DECOMPOSITION           │
│   • SVD of E matrix                 │
│   • 4 candidate (R, t) solutions    │
│   • Cheirality check (depth > 0)    │
│   • Select best solution            │
└─────────────────────────────────────┘
              │
              ▼
      Output: R (3×3), t (3×1 unit vector)
```

## Quality Score

The system computes a quality score (0-100) based on:

| Factor | Penalty |
|--------|---------|
| Features < 100 per image | -15 each |
| Matches < 50 after ratio test | -20 |
| RANSAC inliers < 20 | -25 |
| RANSAC inliers < 50 | -10 |
| Inlier ratio < 0.3 | -15 |
| Inlier ratio < 0.5 | -5 |
| Mean reprojection error > 2.0px | -15 |
| Mean reprojection error > 1.0px | -5 |
| Cheirality pass rate < 80% | -10 |

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (quality ≥ 50) |
| 1 | Low quality warning (20 ≤ quality < 50) |
| 2 | Very low quality (quality < 20) |
| 3 | Exception/error |

## Evaluation Metrics

The project includes standard visual odometry evaluation metrics:

- **ATE** (Absolute Trajectory Error): RMSE of aligned trajectory positions
- **RPE** (Relative Pose Error): Frame-to-frame rotation and translation errors
- **Direction Accuracy**: Angular error between estimated and ground truth translation directions

See [evaluation_metrics.py](evaluation_metrics.py) for implementation details.

## Important Notes

### Scale Ambiguity

In monocular vision, **translation is only recoverable up to scale**. The returned translation vector `t` is a unit vector representing direction only. To recover absolute scale, you need:
- Known object sizes in the scene
- Additional sensors (IMU, wheel odometry)
- Multi-frame constraints
- Stereo baseline (if available)

### Degenerate Cases

The pipeline handles degenerate cases gracefully:
- **Pure rotation**: May produce unreliable translation direction
- **Planar scenes**: Essential matrix may be ill-conditioned
- **Low texture**: Fewer features detected, quality warning issued
- **Motion blur**: Reduced feature matching quality

### Coordinate Systems

- **OpenCV convention**: X-right, Y-down, Z-forward (camera looking along +Z)
- **EuRoC convention**: Requires `--flip_z` flag for correct evaluation
- **KITTI convention**: Compatible with OpenCV, no transformation needed

## Hardware Compatibility

| Platform | Resolution | Performance |
|----------|------------|-------------|
| **DJI Tello** | 960×720 → 640×480 | Primary target |
| **Raspberry Pi Zero** | Any → 640×480 | ~0.5-1.0 sec/pair |
| **KITTI cameras** | 1241×376 (aspect preserved) | Full resolution option |
| **Desktop** | Any resolution | < 100ms/pair |

Requirements: Python 3.8+ and OpenCV 4.5+

## Project Structure

```
monocular-visual-odometry/
├── camera_motion_estimator.py  # Main estimation pipeline (full featured)
├── estimate_motion.py          # Lightweight CLI for Pi Zero / Tello
├── evaluate_euroc.py           # EuRoC MAV dataset evaluation
├── evaluate_kitti.py           # KITTI odometry dataset evaluation
├── run_kitti_diverse.py        # Multi-sequence KITTI evaluation
├── evaluation_metrics.py       # ATE/RPE/APE metrics implementation
├── visualize_results.py        # Result visualization (matplotlib)
├── quick_test.py               # Unified test runner
├── requirements.txt            # Python dependencies
├── tello_calibration_template.json  # DJI Tello calibration template
├── euroc_evaluation.json       # Sample EuRoC evaluation results
├── kitti_evaluation.json       # Sample KITTI evaluation results
└── test_data/
    ├── calibration.json        # KITTI calibration sample
    ├── calibration_lia.json    # Alternative calibration
    └── ground_truth.json       # Sample ground truth data
```

## API Usage

```python
from camera_motion_estimator import CameraMotionEstimator

# Initialize estimator
estimator = CameraMotionEstimator(verbose=True)

# Load calibration
K = estimator.load_calibration("calibration.json")

# Estimate motion between two images
result = estimator.estimate("frame1.jpg", "frame2.jpg", K)

# Access results
R = result.R              # 3x3 rotation matrix
t = result.t              # 3x1 translation vector (unit norm)
E = result.E              # 3x3 essential matrix
quality = result.metrics.quality_score  # 0-100
```

## License

MIT License

## Citation

If you use this code in academic work, please cite:

```bibtex
@software{monocular_vo_2026,
  title={Monocular Visual Odometry - 6-DOF Camera Motion Estimator},
  author={Computer Vision Pipeline},
  year={2026},
  url={https://github.com/your-repo/monocular-visual-odometry}
}
```
