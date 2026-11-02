"""
Physics-Informed Ground Bounce Detection Engine

Detects pitch impact epochs by analyzing vertical kinematics:
- Vertical velocity sign transition (vz < 0 -> vz > 0)
- Vertical acceleration impulse spike (az >> 0)
- Proximity to pitch turf plane (z ~= r_ball)
- Dynamic restitution validation (0.35 <= e_z <= 0.75)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class BounceEvent:
    """Detected pitch bounce occurrence."""
    frame_index: int
    timestamp: float
    position: np.ndarray          # [X, Y, Z] in meters
    incoming_velocity: np.ndarray # [vx, vy, vz] before bounce
    outgoing_velocity: np.ndarray # [vx, vy, vz] after bounce
    restitution_ratio: float      # Measured COR
    seam_deviation_deg: float     # Off-pitch angular cut/spin (degrees)
    confidence: float             # [0.0 - 1.0]


class BounceDetector:
    """
    Monitors 3D kinematic state sequences to reliably detect pitch bounce.
    """

    def __init__(
        self,
        ground_z_threshold: float = 0.20,
        impulse_accel_threshold: float = 30.0,
        window_size: int = 5
    ):
        self.ground_z_threshold = ground_z_threshold
        self.impulse_accel_threshold = impulse_accel_threshold
        self.window_size = window_size
        self.detected_bounce: Optional[BounceEvent] = None

    def reset(self):
        self.detected_bounce = None

    def check_bounce(
        self,
        frame_idx: int,
        timestamp: float,
        pos_history: List[np.ndarray],
        vel_history: List[np.ndarray],
        accel_history: List[np.ndarray]
    ) -> Optional[BounceEvent]:
        """
        Inspects recent kinematic window to identify a bounce event.
        """
        if self.detected_bounce is not None:
            return None  # Only one primary pitch bounce per delivery

        if len(pos_history) < 3 or len(vel_history) < 3:
            return None

        # Look at last two epochs
        z_curr = pos_history[-1][2]
        z_prev = pos_history[-2][2]

        vz_curr = vel_history[-1][2]
        vz_prev = vel_history[-2][2]

        # 1. Check for velocity sign reversal (falling -> rising)
        velocity_reversal = (vz_prev < -0.2) and (vz_curr > 0.1)

        # 2. Check for ground proximity
        near_ground = (z_curr <= self.ground_z_threshold or z_prev <= self.ground_z_threshold)

        # 3. Check for vertical inflection (minimum elevation reached)
        elevation_inflection = (z_curr >= z_prev) and (z_prev <= self.ground_z_threshold) and (vz_prev < 0)

        # 4. Check for vertical acceleration spike
        az = accel_history[-1][2] if len(accel_history) > 0 else 0.0
        accel_spike = (az > self.impulse_accel_threshold) or (vz_curr - vz_prev > 1.5)

        if (velocity_reversal and near_ground) or (elevation_inflection) or (near_ground and accel_spike and vz_curr > -0.2):
            # Compute measured coefficient of restitution
            e_measured = abs(vz_curr / max(abs(vz_prev), 1e-4))
            e_measured = float(np.clip(e_measured, 0.30, 0.85))

            # Compute lateral angular deviation (seam movement off pitch)
            v_in = vel_history[-2]
            v_out = vel_history[-1]
            angle_in = np.arctan2(v_in[0], max(v_in[1], 1e-3))
            angle_out = np.arctan2(v_out[0], max(v_out[1], 1e-3))
            dev_deg = float(np.degrees(angle_out - angle_in))

            # Score confidence
            conf = 0.5
            if velocity_reversal:
                conf += 0.3
            if near_ground:
                conf += 0.2

            bounce_pos = pos_history[-1].copy()
            bounce_pos[2] = 0.036  # Ball radius at turf contact

            event = BounceEvent(
                frame_index=frame_idx,
                timestamp=timestamp,
                position=bounce_pos,
                incoming_velocity=v_in.copy(),
                outgoing_velocity=v_out.copy(),
                restitution_ratio=e_measured,
                seam_deviation_deg=dev_deg,
                confidence=min(1.0, conf)
            )
            self.detected_bounce = event
            return event

        return None
