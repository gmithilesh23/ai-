"""
Comprehensive 3D Cricket Ball Tracking Pipeline

Coordinates:
- Camera ray back-projection (2D pixel -> 3D pitch space)
- Ballistic Kalman filtering with Mahalanobis distance gating
- Bounce detection and state discontinuity handling
- Occlusion coasting & tracklet smoothing
- Delivery event logging (Release, Bounce, Impact, Stumps)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from cricket_vision.models.ball_detector import BallDetection
from cricket_vision.physics.homography import CameraModel
from cricket_vision.physics.trajectory import TrajectoryPoint, BallisticTrajectorySimulator, AerodynamicProperties
from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.tracking.kalman_filter import BallisticKalmanFilter
from cricket_vision.tracking.bounce_detector import BounceDetector, BounceEvent


@dataclass
class TrackedBallState:
    """Consolidated state at a single video frame."""
    frame_idx: int
    timestamp: float
    pos_3d: np.ndarray             # [X, Y, Z] in meters
    vel_3d: np.ndarray             # [vx, vy, vz] in m/s
    accel_3d: np.ndarray           # [ax, ay, az] in m/s^2
    pixel_coords: Tuple[int, int]  # (u, v) in original video
    confidence: float              # Detection / filter confidence
    is_measured: bool              # True if updated by detector, False if coasted
    is_bounced: bool
    speed_kmh: float


class CricketBallTracker:
    """
    End-to-end multi-frame tracker for cricket deliveries.
    """

    def __init__(
        self,
        camera: Optional[CameraModel] = None,
        pitch: Optional[CricketPitch] = None,
        fps: float = 60.0,
        max_lost_frames: int = 8
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_lost_frames = max_lost_frames

        self.camera = camera or CameraModel()
        self.pitch = pitch or CricketPitch()

        self.kf = BallisticKalmanFilter(dt=self.dt)
        self.bounce_detector = BounceDetector()
        self.simulator = BallisticTrajectorySimulator(pitch=self.pitch)

        self.is_tracking = False
        self.consecutive_lost = 0
        self.history: List[TrackedBallState] = []
        self.bounce_event: Optional[BounceEvent] = None

    def reset(self):
        """Resets tracker for a new delivery."""
        self.is_tracking = False
        self.consecutive_lost = 0
        self.history.clear()
        self.bounce_detector.reset()
        self.bounce_event = None

    def process_frame(
        self,
        frame_idx: int,
        detection: Optional[BallDetection]
    ) -> Optional[TrackedBallState]:
        """
        Processes a single frame given optional ball detection.
        Returns the updated ball state.
        """
        timestamp = frame_idx * self.dt

        # Case 1: Active Tracking
        if self.is_tracking:
            # Predict step
            pred_pos = self.kf.predict()

            if detection is not None:
                # Reconstruct 3D position using forward pitch depth prior from prediction
                # This allows true elevation Z to be observed directly from the camera viewing ray
                pos_3d = self.camera.reconstruct_3d_point(
                    detection.x, detection.y, depth_y_prior=pred_pos[1]
                )
                pos_3d[2] = max(0.036, pos_3d[2])

                # Kalman update with Mahalanobis gate
                accepted, dist_m = self.kf.update(pos_3d)

                if accepted:
                    self.consecutive_lost = 0
                    is_measured = True
                    conf = detection.confidence
                else:
                    # Gating rejected outlier: coast on prediction
                    self.consecutive_lost += 1
                    is_measured = False
                    conf = max(0.1, 1.0 - (self.consecutive_lost * 0.15))
            else:
                # No detection: coast on prediction
                self.consecutive_lost += 1
                is_measured = False
                conf = max(0.1, 1.0 - (self.consecutive_lost * 0.15))

            if self.consecutive_lost > self.max_lost_frames:
                self.is_tracking = False
                return None

            curr_pos = self.kf.position
            curr_vel = self.kf.velocity
            curr_accel = self.kf.acceleration

            # Check for ground bounce event
            if self.bounce_event is None and len(self.history) >= 2:
                pos_hist = [s.pos_3d for s in self.history] + [curr_pos]
                vel_hist = [s.vel_3d for s in self.history] + [curr_vel]
                acc_hist = [s.accel_3d for s in self.history] + [curr_accel]

                ev = self.bounce_detector.check_bounce(
                    frame_idx, timestamp, pos_hist, vel_hist, acc_hist
                )
                if ev is not None:
                    self.bounce_event = ev
                    self.kf.apply_bounce_event(ev.restitution_ratio)

            # Project current 3D position to image pixels
            u, v, _ = self.camera.world_to_pixel(curr_pos)
            speed = float(np.linalg.norm(curr_vel) * 3.6)

            state = TrackedBallState(
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                pos_3d=curr_pos,
                vel_3d=curr_vel,
                accel_3d=curr_accel,
                pixel_coords=(u, v),
                confidence=conf,
                is_measured=is_measured,
                is_bounced=(self.bounce_event is not None),
                speed_kmh=round(speed, 1)
            )
            self.history.append(state)
            return state

        # Case 2: Not yet tracking, try to initialize
        elif detection is not None and detection.confidence > 0.40:
            # Initial estimate: assume release height ~2.0m
            init_pos = self.camera.reconstruct_3d_point(detection.x, detection.y, height_prior=2.0)
            # Default delivery forward velocity ~35 m/s (~126 km/h)
            init_vel = np.array([0.0, 35.0, -1.5], dtype=np.float64)

            self.kf.initialize(init_pos, init_vel)
            self.is_tracking = True
            self.consecutive_lost = 0

            u, v, _ = self.camera.world_to_pixel(init_pos)
            state = TrackedBallState(
                frame_idx=frame_idx,
                timestamp=round(timestamp, 4),
                pos_3d=init_pos,
                vel_3d=init_vel,
                accel_3d=np.array([0.0, 0.0, -9.8]),
                pixel_coords=(u, v),
                confidence=detection.confidence,
                is_measured=True,
                is_bounced=False,
                speed_kmh=round(float(np.linalg.norm(init_vel) * 3.6), 1)
            )
            self.history.append(state)
            return state

        return None

    def get_trajectory_points(self) -> List[TrajectoryPoint]:
        """
        Converts tracking history into a list of TrajectoryPoint instances
        with phase classification (flight_pre_bounce, bounce, flight_post_bounce).
        """
        if not self.history:
            return []

        pts: List[TrajectoryPoint] = []
        bounced = False

        for state in self.history:
            if self.bounce_event and state.frame_idx == self.bounce_event.frame_index:
                phase = "bounce"
                bounced = True
            elif bounced:
                phase = "flight_post_bounce"
            else:
                phase = "flight_pre_bounce"

            pts.append(TrajectoryPoint(
                time=state.timestamp,
                position=state.pos_3d.copy(),
                velocity=state.vel_3d.copy(),
                acceleration=state.accel_3d.copy(),
                phase=phase
            ))

        return pts
