# Monocular Visual Odometry - 6-DOF Camera Motion Estimator

A high-precision Python pipeline for estimating 6 Degrees of Freedom (6-DOF) camera motion between two monocular images. Designed for mobile robotics applications including drones and ground vehicles.

## Features

- **High Accuracy Focus**: Optimized for precision over speed
- **Robust Preprocessing**: CLAHE enhancement, bilateral filtering, automatic downsampling from 4K to 640×480
- **ORB Feature Detection**: Configured for maximum feature coverage (5000 features)
- **Multi-Stage Matching**: Ratio test → Cross-check → Distance filtering
- **RANSAC Outlier Rejection**: High-confidence fundamental matrix estimation
- **Essential Matrix Decomposition**: SVD-based with proper constraint enforcement
- **Cheirality Verification**: Automatic selection of correct (R, t) solution
- **Quality Metrics**: Comprehensive scoring and warnings system

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.x
- NumPy

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

```bash
python camera_motion_estimator.py <image1> <image2> <calibration_file>
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
│   4. OUTLIER REJECTION      │
│   • RANSAC (5000 iters)     │
│   • Confidence: 0.9999      │
│   • Threshold: 1.0 px       │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   5. ESSENTIAL MATRIX       │
│   • E = K^T × F × K         │
│   • SVD constraint enforce  │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│   6. MOTION DECOMPOSITION   │
│   • 4 candidate solutions   │
│   • Cheirality check        │
│   • Select valid (R, t)     │
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
- **Raspberry Pi Zero** (primary target)
- Any system with Python 3.8+ and OpenCV

Expected performance on Raspberry Pi Zero:
- ~0.5-1.0 seconds per frame pair

## License

MIT License

## Citation

If you use this code in academic work, please cite appropriately.
