# Monocular Visual Odometry - 6-DOF Camera Motion Estimator

A high-precision Python pipeline for estimating 6 Degrees of Freedom (6-DOF) camera motion between two monocular images. Designed for mobile robotics applications including drones (DJI Tello) and ground vehicles.

## 🎯 Current Accuracy

**EuRoC MAV Dataset (MH_01_easy):** **58.3%** combined accuracy (R<5°, t<15°)

## Features

- **High Accuracy Focus**: Optimized for precision over speed
- **Robust Preprocessing**: CLAHE enhancement, bilateral filtering, automatic downsampling
- **ORB Feature Detection**: Configured for maximum feature coverage (2000 features)
- **Ratio Test Matching**: Lowe's ratio test (0.6 threshold)
- **USAC_MAGSAC**: State-of-the-art robust estimation (OpenCV 4.5+)
- **Essential Matrix Decomposition**: SVD-based with proper constraint enforcement
- **Cheirality Verification**: Automatic selection of correct (R, t) solution
- **Multi-Dataset Support**: EuRoC, KITTI, DJI Tello
- **Raspberry Pi Zero Compatible**: Lightweight CLI tool included

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.5+ (for USAC_MAGSAC support)
- NumPy
- matplotlib (optional, for visualization)

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (Full Pipeline)

```bash
python camera_motion_estimator.py <image1> <image2> <calibration_file>
```

### Raspberry Pi / DJI Tello (Lightweight CLI)

```bash
# Simple output
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json

# JSON output (for parsing)
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json --json

# Verbose mode
python estimate_motion.py frame1.jpg frame2.jpg tello_calib.json --verbose
```

### Dataset Evaluation

```bash
# EuRoC MAV Dataset
python evaluate_euroc.py euroc_test/machine_hall/MH_01_easy/mav0 \
    --num_pairs 100 --step 5 --flip_z

# KITTI Odometry Dataset
python evaluate_kitti.py kitti_odometry --sequence 00 --num_pairs 100

# Visualize Results
python visualize_results.py euroc_evaluation.json --show
```

### With Output Directory

```bash
python camera_motion_estimator.py image1.jpg image2.jpg calibration.npy --output_dir results/
```

### Example

```bash
python camera_motion_estimator.py frame_001.jpg frame_002.jpg dji_tello_calibration.json --output_dir ./output
```

## Input

### Images
- Two JPEG/PNG images (consecutive frames from video)
- Supports any resolution (automatically downsampled to 640×480)

### Calibration Matrix
The camera intrinsic matrix K (3×3) for the **original image resolution**.

Supported formats:
- **NumPy** (`.npy`): 3×3 array
- **JSON** (`.json`): 
  ```json
  {"K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
  ```
  or 
  ```json
  {"camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
  ```
- **Text** (`.txt`): Space/comma-separated 3×3 matrix

### Example Calibration File (JSON)

```json
{
  "K": [
    [921.170702, 0.0, 459.904354],
    [0.0, 919.018377, 351.238301],
    [0.0, 0.0, 1.0]
  ]
}
```

## Output

### Files Generated

| File | Description |
|------|-------------|
| `motion_R.npy` | 3×3 Rotation matrix |
| `motion_t.npy` | 3×1 Translation vector (unit norm) |
| `motion_F.npy` | 3×3 Fundamental matrix |
| `motion_E.npy` | 3×3 Essential matrix |
| `motion_summary.txt` | Human-readable summary |
| `motion_result.json` | Complete results in JSON format |

## Benchmark Results: EuRoC MAV Dataset

Evaluated on EuRoC Machine Hall MH_01_easy sequence (3682 images, 20Hz):

### Accuracy Summary (Current Best: 58.3%)

| Metric | Threshold | Accuracy |
|--------|-----------|----------|
| **Rotation** | < 5° | **87.5%** |
| **Rotation** | < 2° | **22.9%** |
| **Translation Direction** | < 15° | **70.8%** |
| **Translation Direction** | < 10° | **47.9%** |
| **Combined (R<5°, t<15°)** | Normal | **58.3%** |
| **Combined (R<2°, t<5°)** | Strict | **14.6%** |
| **Combined (R<10°, t<30°)** | Relaxed | **93.8%** |

### Detailed Statistics

- **Rotation Error**: Mean=3.3°, Median=2.9°, Std=1.9°
- **Translation Direction Error**: Mean=13.7°, Median=10.6°, Std=13.2°
- **100% success rate** (no OpenCV exceptions)

*Note: Translation is direction-only; scale is unobservable in monocular VO*

### Running Evaluation

```bash
# Download EuRoC MH_01_easy and extract
python evaluate_euroc.py euroc_test/machine_hall/MH_01_easy/mav0 \
    --num_pairs 100 --step 5 --flip_z
```

The `--flip_z` flag accounts for coordinate system differences between OpenCV and EuRoC.

### Console Output

The pipeline prints detailed progress including:
- Preprocessing statistics
- Feature detection counts
- Matching statistics at each stage
- RANSAC inlier count and ratio
- Essential matrix singular values
- Cheirality check results
- Reprojection errors
- Final quality score (0-100)

## Algorithm Pipeline

```
Input Images (4K) + Calibration Matrix K
              │
              ▼
┌─────────────────────────────┐
│   1. IMAGE PREPROCESSING    │
│   • Downsample to 640×480   │
│   • CLAHE enhancement       │
│   • Bilateral filtering     │
│   • Scale calibration K     │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   2. FEATURE DETECTION      │
│   • ORB (5000 features)     │
│   • Harris score ranking    │
│   • 8 pyramid levels        │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   3. FEATURE MATCHING       │
│   • Brute-force Hamming     │
│   • Lowe's ratio test (0.7) │
│   • Cross-check validation  │
│   • Distance threshold      │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   4. ESSENTIAL MATRIX       │
│   (Direct estimation)       │
│   • USAC_MAGSAC (10K iters) │
│   • Confidence: 0.9999      │
│   • Threshold: 1.0 px       │
│   • Fallback: F → E         │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   5. MOTION DECOMPOSITION   │
│   • SVD of E matrix         │
│   • 4 candidate solutions   │
│   • Cheirality check        │
│   • Reprojection refinement │
└─────────────────────────────┘
              │
              ▼
      Output: R (3×3), t (3×1)
```

## Quality Score

The system computes a quality score (0-100) based on:
- Feature detection count
- Match count after filtering
- RANSAC inlier count and ratio
- Reprojection error
- Cheirality check pass rate

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (quality ≥ 50) |
| 1 | Low quality warning (20 ≤ quality < 50) |
| 2 | Very low quality (quality < 20) |
| 3 | Exception/error |

## Important Notes

### Scale Ambiguity

In monocular vision, **translation is only recoverable up to scale**. The returned translation vector `t` is a unit vector representing direction only. To recover absolute scale, you need:
- Known object sizes in the scene
- Additional sensors (IMU, wheel odometry)
- Multi-frame constraints

### Degenerate Cases

The pipeline handles degenerate cases gracefully:
- **Pure rotation**: May produce unreliable translation
- **Planar scenes**: Fundamental matrix may be ill-conditioned
- **Low texture**: Fewer features detected, quality warning issued

## Hardware Compatibility

Designed and tested for:
- **DJI Tello Drone** (960×720 → 640×480)
- **Raspberry Pi Zero** (primary deployment target)
- **KITTI-style cameras** (1242×375 with aspect ratio preservation)
- Any system with Python 3.8+ and OpenCV 4.5+

Expected performance on Raspberry Pi Zero:
- ~0.5-1.0 seconds per frame pair

## Project Structure

```
├── camera_motion_estimator.py  # Main estimation pipeline
├── estimate_motion.py          # Lightweight CLI for Pi Zero / Tello
├── evaluate_euroc.py           # EuRoC dataset evaluation
├── evaluate_kitti.py           # KITTI odometry evaluation
├── visualize_results.py        # Result visualization tools
├── evaluation_metrics.py       # ATE/RPE metrics
├── setup_kitti.py              # KITTI dataset setup helper
├── tello_calibration_template.json  # DJI Tello calibration template
└── requirements.txt            # Dependencies
```

## License

MIT License

## Citation

If you use this code in academic work, please cite appropriately.
