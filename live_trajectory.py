"""
Visual Odometry with Real-Time Visualization
=============================================
Uses the new VisualOdometry class for motion estimation.
Top-down view with video side-by-side.

Usage:
    python live_trajectory.py video.mp4 -c calibration.json
"""

import cv2
import numpy as np
import sys
import os
import argparse
import tempfile
import json
from pathlib import Path

from visual_odometry import VisualOdometry


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_WIDTH = 1400          # Total window width
WINDOW_HEIGHT = 700          # Total window height
TRAIL_LENGTH = 500           # Number of past positions to show


# =============================================================================
# VISUALIZATION CLASS
# =============================================================================

class TrajectoryVisualizer:
    """Real-time trajectory visualization with top-down view."""
    
    def __init__(self, video_width, video_height):
        # Scale video to fit in half the window
        max_video_width = WINDOW_WIDTH // 2 - 10
        max_video_height = WINDOW_HEIGHT - 20
        
        scale = min(max_video_width / video_width, max_video_height / video_height, 1.0)
        self.video_width = int(video_width * scale)
        self.video_height = int(video_height * scale)
        self.video_scale = scale
        
        # Trajectory gets remaining space
        self.traj_width = max(WINDOW_WIDTH - self.video_width - 30, 400)
        self.traj_height = WINDOW_HEIGHT
        
        # Trajectory state
        self.positions = []  # List of (x, y, z) positions
        self.current_pos = np.zeros(3)
        self.current_rot = np.eye(3)
        
        # For auto-scaling the trajectory view
        self.min_x = 0
        self.max_x = 0
        self.min_z = 0
        self.max_z = 0
        
    def update(self, R, t, success):
        """Update trajectory with new pose estimate."""
        if success and R is not None and t is not None:
            # Integrate pose into world frame
            t_flat = t.flatten()
            self.current_pos = self.current_pos + self.current_rot @ t_flat
            self.current_rot = self.current_rot @ R.T
            
        self.positions.append(self.current_pos.copy())
        
        # Update bounds for auto-scaling
        if len(self.positions) > 1:
            x, _, z = self.current_pos
            self.min_x = min(self.min_x, x)
            self.max_x = max(self.max_x, x)
            self.min_z = min(self.min_z, z)
            self.max_z = max(self.max_z, z)
    
    def draw_trajectory(self):
        """Draw top-down trajectory view."""
        # Create trajectory canvas (white background)
        traj_img = np.ones((self.traj_height, self.traj_width, 3), dtype=np.uint8) * 255
        
        if len(self.positions) < 2:
            # Draw start indicator
            center = (self.traj_width // 2, self.traj_height // 2)
            cv2.circle(traj_img, center, 10, (0, 180, 0), -1)
            cv2.putText(traj_img, "START (0,0)", (center[0] + 15, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)
            return traj_img
        
        # Calculate scale and offset to fit trajectory
        margin = 60
        usable_width = self.traj_width - 2 * margin
        usable_height = self.traj_height - 2 * margin
        
        # Get range (include 0,0 to keep origin visible)
        x_range = max(self.max_x - self.min_x, abs(self.max_x), abs(self.min_x), 1) * 2
        z_range = max(self.max_z - self.min_z, abs(self.max_z), abs(self.min_z), 1) * 2
        
        # Scale to fit (maintain aspect ratio)
        scale = min(usable_width / x_range, usable_height / z_range) * 0.8
        
        # Center offset
        center_x = self.traj_width // 2
        center_z = self.traj_height // 2
        
        def world_to_screen(pos):
            """Convert world coordinates to screen coordinates."""
            x, _, z = pos
            # X positive = right on screen
            # Z positive = forward = up on screen (flip for screen coords)
            sx = int(center_x + x * scale)
            sz = int(center_z - z * scale)
            return (sx, sz)
        
        # Draw grid
        grid_color = (230, 230, 230)
        for i in range(0, self.traj_width, 50):
            cv2.line(traj_img, (i, 0), (i, self.traj_height), grid_color, 1)
        for i in range(0, self.traj_height, 50):
            cv2.line(traj_img, (0, i), (self.traj_width, i), grid_color, 1)
        
        # Draw origin axes
        origin = world_to_screen(np.zeros(3))
        cv2.drawMarker(traj_img, origin, (200, 200, 200), cv2.MARKER_CROSS, 20, 1)
        
        # Draw trajectory trail
        positions_to_draw = self.positions[-TRAIL_LENGTH:]
        
        for i in range(1, len(positions_to_draw)):
            pt1 = world_to_screen(positions_to_draw[i-1])
            pt2 = world_to_screen(positions_to_draw[i])
            
            # Color gradient: older = lighter blue, newer = darker blue
            alpha = i / len(positions_to_draw)
            color = (255 - int(200 * alpha), 150 - int(100 * alpha), 50)
            
            cv2.line(traj_img, pt1, pt2, color, 2)
        
        # Draw start point (green)
        start_pt = world_to_screen(np.zeros(3))
        cv2.circle(traj_img, start_pt, 10, (0, 180, 0), -1)
        cv2.putText(traj_img, "Start", (start_pt[0] + 12, start_pt[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 0), 1)
        
        # Draw current position (red) with direction arrow
        current_pt = world_to_screen(self.current_pos)
        cv2.circle(traj_img, current_pt, 8, (0, 0, 255), -1)
        
        # Draw direction arrow (forward in camera frame)
        forward = self.current_rot @ np.array([0, 0, 1])
        arrow_end = (
            int(current_pt[0] + forward[0] * 40),
            int(current_pt[1] - forward[2] * 40)  # Flip Z
        )
        cv2.arrowedLine(traj_img, current_pt, arrow_end, (0, 0, 255), 2, tipLength=0.3)
        
        # Info panel
        cv2.rectangle(traj_img, (5, 5), (200, 100), (240, 240, 240), -1)
        cv2.rectangle(traj_img, (5, 5), (200, 100), (180, 180, 180), 1)
        cv2.putText(traj_img, "TOP-DOWN VIEW", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(traj_img, f"X: {self.current_pos[0]:.2f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(traj_img, f"Z: {self.current_pos[2]:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(traj_img, f"Points: {len(self.positions)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Axes indicator (bottom right)
        axes_origin = (self.traj_width - 70, self.traj_height - 70)
        cv2.arrowedLine(traj_img, axes_origin, (axes_origin[0] + 40, axes_origin[1]),
                       (255, 0, 0), 2, tipLength=0.2)  # X axis (red) = Right
        cv2.arrowedLine(traj_img, axes_origin, (axes_origin[0], axes_origin[1] - 40),
                       (0, 0, 255), 2, tipLength=0.2)  # Z axis (blue) = Forward
        cv2.putText(traj_img, "Right", (axes_origin[0] + 30, axes_origin[1] + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(traj_img, "Fwd", (axes_origin[0] - 25, axes_origin[1] - 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return traj_img
    
    def draw_frame_with_info(self, frame, quality_score, num_inliers, success, skip_count=0):
        """Draw the video frame with tracking info."""
        vis = frame.copy()
        vis = cv2.resize(vis, (self.video_width, self.video_height))
        
        # Info overlay
        color = (0, 255, 0) if success else (0, 0, 255)
        status = "OK" if success else f"SKIP ({skip_count})"
        
        cv2.putText(vis, f"Quality: {quality_score:.0f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Inliers: {num_inliers}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"Status: {status}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis
    
    def create_combined_view(self, frame_vis, frame_idx, total_frames):
        """Create combined visualization."""
        # Draw trajectory
        traj_vis = self.draw_trajectory()
        
        # Match heights
        if frame_vis.shape[0] != traj_vis.shape[0]:
            traj_vis = cv2.resize(traj_vis, (self.traj_width, frame_vis.shape[0]))
        
        # Combine side by side
        combined = np.hstack([frame_vis, traj_vis])
        
        # Frame counter at bottom
        cv2.putText(combined, f"Frame: {frame_idx}/{total_frames}", (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return combined


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_realtime_vo(video_path, calibration_path, frame_step=1):
    """Run VO with real-time visualization using YOUR CameraMotionEstimator."""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"Frame step: {frame_step}")
    
    # Setup calibration
    temp_calib = None
    if calibration_path is None or not Path(calibration_path).exists():
        fx = fy = max(width, height)
        cx, cy = width / 2, height / 2
        calib = {
            "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0],
            "image_width": width,
            "image_height": height
        }
        temp_calib = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(calib, temp_calib)
        temp_calib.close()
        calibration_path = temp_calib.name
        print("Using default calibration")
    else:
        print(f"Calibration: {calibration_path}")
    
    # Create VisualOdometry estimator
    print("\nUsing VisualOdometry (SIFT + FLANN)!")
    vo = VisualOdometry(calibration_path)
    
    # Temp files for frame saving
    temp_dir = tempfile.mkdtemp()
    path1 = os.path.join(temp_dir, "frame1.png")
    path2 = os.path.join(temp_dir, "frame2.png")
    
    # Initialize visualizer
    visualizer = TrajectoryVisualizer(width, height)
    
    # Create window
    cv2.namedWindow("Visual Odometry", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Visual Odometry", WINDOW_WIDTH, WINDOW_HEIGHT)
    
    print(f"\nPress 'q' or ESC to quit, SPACE to pause, 'r' to restart")
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    frame_idx = 1
    last_proc_idx = 0
    paused = False
    skip_count = 0  # Track consecutive skipped frames
    
    # Initial display
    frame_vis = visualizer.draw_frame_with_info(prev_frame, 0, 0, False, 0)
    quality_score = 0
    num_inliers = 0
    success = False
    
    while True:
        if not paused:
            ret, curr_frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 1
                ret, prev_frame = cap.read()
                continue
            
            # Process every Nth frame
            if frame_idx - last_proc_idx >= frame_step:
                # Save frames
                cv2.imwrite(path1, prev_frame)
                cv2.imwrite(path2, curr_frame)
                
                # Run VisualOdometry estimator
                result = vo.estimate_motion(path1, path2, verbose=False)
                
                quality_score = result.num_inliers  # Use inliers as quality indicator
                num_inliers = result.num_inliers
                success = result.success and result.num_inliers >= 20
                
                if success:
                    R, t = result.R, result.t
                    visualizer.update(R, t, True)
                    skip_count = 0  # Reset skip counter on success
                else:
                    visualizer.update(None, None, False)
                    skip_count += 1
                
                # Always update prev_frame so we don't get stuck comparing old frames
                prev_frame = curr_frame.copy()
                last_proc_idx = frame_idx
            
            # Draw current frame with info
            frame_vis = visualizer.draw_frame_with_info(curr_frame, quality_score, num_inliers, success, skip_count)
            frame_idx += 1
        
        # Create combined view
        combined = visualizer.create_combined_view(frame_vis, frame_idx, total_frames)
        
        # Show
        cv2.imshow("Visual Odometry", combined)
        
        # Handle keys
        delay = max(1, int(1000 / fps)) if not paused else 50
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):  # Space to pause
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):  # Restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 1
            visualizer = TrajectoryVisualizer(width, height)
            ret, prev_frame = cap.read()
            print("Restarted")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    if temp_calib:
        os.unlink(temp_calib.name)
    
    # Final stats
    print(f"\nProcessed {frame_idx} frames")
    final_pos = visualizer.current_pos
    print(f"Final position: X={final_pos[0]:.2f}, Y={final_pos[1]:.2f}, Z={final_pos[2]:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visual Odometry with Real-Time Visualization (uses YOUR code)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  q / ESC  - Quit
  SPACE    - Pause/Resume  
  r        - Restart video

Examples:
  python live_trajectory.py video.mp4
  python live_trajectory.py video.mp4 -c calibration.json
  python live_trajectory.py video.mp4 --step 5
        """
    )
    
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('--calibration', '-c', help='Path to calibration JSON')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth frame (default: 1)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    run_realtime_vo(args.video, args.calibration, args.step)


if __name__ == "__main__":
    main()
