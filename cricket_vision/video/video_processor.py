"""
Video Processing Pipeline for Cricket Ball Tracking & Hawk-Eye DRS Analysis

Coordinates:
- Video ingestion and frame buffer management
- Multi-frame temporal deep learning detection
- 3D tracking and pitch bounce identification
- Future trajectory projection (Hawk-Eye)
- High-definition video annotation and output rendering
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import cv2
import pandas as pd

from cricket_vision.models.ball_detector import CricketBallDetector, BallDetection
from cricket_vision.physics.homography import CameraModel
from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.trajectory import BallisticTrajectorySimulator, TrajectoryPoint
from cricket_vision.physics.hawk_eye import HawkEyeDRSSystem, DRSReviewResult
from cricket_vision.tracking.tracker import CricketBallTracker, TrackedBallState
from cricket_vision.video.visualizer import CricketVisualizer


class VideoTrackingPipeline:
    """
    End-to-end video processing engine.
    """

    def __init__(
        self,
        weights_path: Optional[str] = "cricket_vision/models/weights/tracknet_cricket.pth",
        camera: Optional[CameraModel] = None,
        confidence_thresh: float = 0.35,
        match_pitch_model: Optional[object] = None
    ):
        self.camera = camera or CameraModel()
        self.pitch = CricketPitch()
        self.match_pitch_model = match_pitch_model
        self.detector = CricketBallDetector(weights_path=weights_path, confidence_thresh=confidence_thresh)
        self.tracker = CricketBallTracker(camera=self.camera, pitch=self.pitch)
        self.drs_system = HawkEyeDRSSystem(pitch=self.pitch)
        self.visualizer = CricketVisualizer(camera=self.camera, pitch=self.pitch)
        self.simulator = BallisticTrajectorySimulator(pitch=self.pitch, pitch_model=match_pitch_model)

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict:
        """
        Executes complete tracking pipeline over an input video.
        Returns:
            Dictionary containing:
            - 'drs_result': DRSReviewResult dataclass
            - 'telemetry_df': Pandas DataFrame of frame-by-frame kinematics
            - 'trajectory': List of TrackedBallState
            - 'projected_path': List of 3D points
            - 'output_video': Path to output annotated video (if rendered)
            - 'fps': Processing frames per second
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.tracker.reset()
        self.tracker.fps = fps
        self.tracker.dt = 1.0 / fps

        # Read all raw frames
        raw_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        if len(raw_frames) < 3:
            raise ValueError(f"Video {video_path} too short ({len(raw_frames)} frames)")

        # Pass 1: Multi-frame Detection & Tracking
        detections: List[Optional[BallDetection]] = [None] * len(raw_frames)
        tracked_states: List[Optional[TrackedBallState]] = [None] * len(raw_frames)

        start_time = time.time()
        for idx in range(len(raw_frames)):
            # Form 3-frame temporal triplet [idx-1, idx, idx+1]
            prev_idx = max(0, idx - 1)
            next_idx = min(len(raw_frames) - 1, idx + 1)
            triplet = [raw_frames[prev_idx], raw_frames[idx], raw_frames[next_idx]]

            det = self.detector.detect_triplet(triplet)
            detections[idx] = det
            state = self.tracker.process_frame(idx, det)
            tracked_states[idx] = state

            if progress_callback:
                progress_callback(0.5 * (idx / float(len(raw_frames))))

        # Pass 2: Hawk-Eye Trajectory Projection & DRS Evaluation
        trajectory_pts = self.tracker.get_trajectory_points()
        projected_3d_pts: List[np.ndarray] = []
        drs_result: Optional[DRSReviewResult] = None

        if len(trajectory_pts) >= 5:
            drs_result = self.drs_system.evaluate_delivery(trajectory_pts)

            # Project future path beyond impact to stumps
            last_valid = trajectory_pts[-1]
            future_sim = self.simulator.project_virtual_path(last_valid, dt=1.0/fps)
            projected_3d_pts = [p.position for p in future_sim]

        # Register bounce event in match-adaptive model if available
        bounce_obs = None
        if self.tracker.bounce_event and self.match_pitch_model:
            ev = self.tracker.bounce_event
            bounce_obs = self.match_pitch_model.observe_delivery(
                delivery_idx=len(self.match_pitch_model.history) + 1,
                timestamp=ev.timestamp,
                bounce_x=ev.position[0],
                bounce_y=ev.position[1],
                v_in=ev.incoming_velocity,
                v_out_actual=ev.outgoing_velocity
            )

        # Pass 3: High-definition Video Annotation & Writing
        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        annotated_frames: List[np.ndarray] = []
        active_track_history: List[TrackedBallState] = []

        for idx, frame in enumerate(raw_frames):
            out_f = frame.copy()
            curr_st = tracked_states[idx]

            if curr_st is not None:
                active_track_history.append(curr_st)

            # Render 3D pitch lines
            out_f = self.visualizer.draw_pitch_wireframe(out_f)

            # Render trajectory glow ribbon
            if active_track_history:
                # Include projected points once ball has bounced or is near popping crease
                show_proj = projected_3d_pts if (self.tracker.bounce_event and idx >= self.tracker.bounce_event.frame_index) else None
                out_f = self.visualizer.draw_trajectory_ribbon(out_f, active_track_history, show_proj)

            # Render pitch bounce circle
            if self.tracker.bounce_event and idx >= self.tracker.bounce_event.frame_index:
                out_f = self.visualizer.draw_bounce_spot(out_f, self.tracker.bounce_event.position)

            # Draw current ball target reticle
            if curr_st is not None and curr_st.pixel_coords[0] != -1:
                cx, cy = curr_st.pixel_coords
                cv2.circle(out_f, (cx, cy), 8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(out_f, (cx, cy), 2, (0, 0, 255), -1, cv2.LINE_AA)

            # Render telemetry HUD
            out_f = self.visualizer.draw_telemetry_hud(out_f, curr_st, drs_result if idx > len(raw_frames) * 0.6 else None)

            if writer:
                writer.write(out_f)
            annotated_frames.append(out_f)

            if progress_callback:
                progress_callback(0.5 + 0.5 * (idx / float(len(raw_frames))))

        if writer:
            writer.release()

        proc_time = time.time() - start_time
        proc_fps = len(raw_frames) / max(0.001, proc_time)

        # Build Telemetry DataFrame
        records = []
        for s in active_track_history:
            records.append({
                "frame": s.frame_idx,
                "time_s": s.timestamp,
                "x_m": round(float(s.pos_3d[0]), 3),
                "y_m": round(float(s.pos_3d[1]), 3),
                "z_m": round(float(s.pos_3d[2]), 3),
                "vx_mps": round(float(s.vel_3d[0]), 2),
                "vy_mps": round(float(s.vel_3d[1]), 2),
                "vz_mps": round(float(s.vel_3d[2]), 2),
                "speed_kmh": s.speed_kmh,
                "confidence": round(float(s.confidence), 3),
                "is_bounced": s.is_bounced,
            })
        telemetry_df = pd.DataFrame(records)

        return {
            "drs_result": drs_result,
            "telemetry_df": telemetry_df,
            "trajectory": active_track_history,
            "projected_path": projected_3d_pts,
            "output_video": output_video_path,
            "annotated_frames": annotated_frames,
            "fps": proc_fps,
            "total_frames": len(raw_frames),
            "bounce_obs": bounce_obs,
            "match_pitch_model": self.match_pitch_model
        }
