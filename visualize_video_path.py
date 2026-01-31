"""
Visualize camera path from video using motion estimation.
Takes pairs of frames with a specified step and plots a top-down view of the trajectory.
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import tempfile
import os

from camera_motion_estimator import CameraMotionEstimator


def extract_frames(video_path: str, frame_step: int = 10):
    """Extract frames from video with specified step."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_indices = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_step == 0:
            frames.append(frame)
            frame_indices.append(frame_idx)
        
        frame_idx += 1
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames from video (total: {frame_idx} frames, step: {frame_step})")
    return frames, frame_indices


def estimate_trajectory(frames: list, calibration_path: str | None = None):
    """
    Estimate camera trajectory from sequence of frames.
    Returns list of (x, z) positions in top-down view.
    """
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames")
    
    # Create temporary calibration if not provided
    temp_calib_file = None
    if calibration_path is None or not Path(calibration_path).exists():
        # Create default calibration based on frame size
        h, w = frames[0].shape[:2]
        fx = fy = w  # Approximate focal length
        cx, cy = w / 2, h / 2
        
        temp_calib = {
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0],
            "image_width": w,
            "image_height": h
        }
        
        temp_calib_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(temp_calib, temp_calib_file)
        temp_calib_file.close()
        calibration_path = temp_calib_file.name
        print(f"Using default calibration (frame size: {w}x{h})")
    else:
        print(f"Loaded calibration from {calibration_path}")
    
    # Create estimator
    estimator = CameraMotionEstimator(calibration_path, preserve_aspect_ratio=True)
    
    # Initialize trajectory
    positions = [(0.0, 0.0)]  # (x, z) in world coordinates
    current_pose = np.eye(4)  # 4x4 transformation matrix
    
    rotations = []
    translations = []
    
    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp()
    
    print(f"\nEstimating motion for {len(frames)-1} frame pairs...")
    
    try:
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Save frames temporarily
            path1 = os.path.join(temp_dir, f"frame_{i:04d}.png")
            path2 = os.path.join(temp_dir, f"frame_{i+1:04d}.png")
            cv2.imwrite(path1, frame1)
            cv2.imwrite(path2, frame2)
            
            # Estimate motion between frames
            try:
                result = estimator.estimate(path1, path2)
                
                if result is None or result.metrics.quality_score < 50:
                    print(f"  Pair {i}: Low quality estimate")
                    positions.append(positions[-1])
                    continue
                
                R, t = result.R, result.t
                
                # Ensure t is a column vector
                t = t.reshape(3, 1)
                
                # Create relative transformation matrix
                T_rel = np.eye(4)
                T_rel[:3, :3] = R
                T_rel[:3, 3] = t.flatten()
                
                # Update current pose: new_pose = current_pose @ T_rel
                current_pose = current_pose @ T_rel
                
                # Extract position (x, z for top-down view)
                x = current_pose[0, 3]
                z = current_pose[2, 3]
                positions.append((x, z))
                
                rotations.append(R)
                translations.append(t)
                
                # Print progress every 10 pairs
                if (i + 1) % 10 == 0 or i == len(frames) - 2:
                    print(f"  Processed {i+1}/{len(frames)-1} pairs")
                    
            except Exception as e:
                print(f"  Pair {i}: Error - {str(e)[:50]}")
                positions.append(positions[-1])
    
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        if temp_calib_file:
            os.unlink(temp_calib_file.name)
    
    return positions, rotations, translations


def plot_trajectory(positions: list, output_path: str | None = None, title: str = "Camera Trajectory (Top-Down View)"):
    """Plot top-down view of camera trajectory."""
    
    x_coords = [p[0] for p in positions]
    z_coords = [p[1] for p in positions]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Plot trajectory
    ax.plot(x_coords, z_coords, 'b-', linewidth=1.5, label='Path')
    
    # Mark start and end
    ax.scatter(x_coords[0], z_coords[0], c='green', s=150, marker='o', zorder=5, label='Start')
    ax.scatter(x_coords[-1], z_coords[-1], c='red', s=150, marker='s', zorder=5, label='End')
    
    # Plot direction arrows every few points
    arrow_step = max(1, len(positions) // 20)
    for i in range(0, len(positions) - 1, arrow_step):
        dx = x_coords[i+1] - x_coords[i]
        dz = z_coords[i+1] - z_coords[i]
        if abs(dx) > 1e-6 or abs(dz) > 1e-6:
            ax.annotate('', xy=(x_coords[i] + dx*0.5, z_coords[i] + dz*0.5),
                       xytext=(x_coords[i], z_coords[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=1))
    
    ax.set_xlabel('X (right/left)', fontsize=12)
    ax.set_ylabel('Z (forward)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Add scale info
    x_range = max(x_coords) - min(x_coords)
    z_range = max(z_coords) - min(z_coords)
    ax.text(0.02, 0.98, f'X range: {x_range:.2f}\nZ range: {z_range:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {output_path}")
    
    plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize camera path from video (top-down view)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_video_path.py video.mp4
  python visualize_video_path.py video.mp4 --step 5 --calibration calib.json
  python visualize_video_path.py video.mp4 --output path.png
        """
    )
    
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--step', type=int, default=10,
                       help='Frame step (default: 10, meaning frames 0,10,20,...)')
    parser.add_argument('--calibration', '-c', type=str, default=None,
                       help='Path to camera calibration JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for the plot image')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print(f"Processing video: {args.video}")
    print(f"Frame step: {args.step}")
    
    # Extract frames
    frames, frame_indices = extract_frames(args.video, args.step)
    
    if args.max_frames and len(frames) > args.max_frames:
        frames = frames[:args.max_frames]
        frame_indices = frame_indices[:args.max_frames]
        print(f"Limited to {args.max_frames} frames")
    
    if len(frames) < 2:
        print("Error: Not enough frames extracted from video")
        sys.exit(1)
    
    # Estimate trajectory
    positions, rotations, translations = estimate_trajectory(frames, args.calibration)
    
    # Plot
    video_name = Path(args.video).stem
    title = f"Camera Trajectory - {video_name}\n(Top-Down View, step={args.step})"
    
    output_path = args.output
    if output_path is None:
        output_path = f"trajectory_{video_name}.png"
    
    plot_trajectory(positions, output_path, title)
    
    print(f"\nTrajectory statistics:")
    print(f"  Total frames processed: {len(frames)}")
    print(f"  Total motion estimates: {len(positions) - 1}")
    
    x_coords = [p[0] for p in positions]
    z_coords = [p[1] for p in positions]
    print(f"  Final position: ({x_coords[-1]:.3f}, {z_coords[-1]:.3f})")
    print(f"  Total X displacement: {x_coords[-1] - x_coords[0]:.3f}")
    print(f"  Total Z displacement: {z_coords[-1] - z_coords[0]:.3f}")


if __name__ == '__main__':
    main()
