"""
Cricket Match Delivery Video Generator

Synthesizes high-definition 60 FPS match deliveries with realistic physics:
- Inswinging Yorker (144 km/h)
- Outswinger Good Length (138 km/h)
- Leg-break Turning Spinner (88 km/h)
- Fast Bouncer (148 km/h)
- LBW Review Candidate (141 km/h)
- Reverse Swing Yorker (146 km/h)
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.trajectory import BallisticTrajectorySimulator, AerodynamicProperties, TrajectoryPoint
from cricket_vision.physics.homography import CameraModel


class CricketDeliveryGenerator:
    """
    Renders realistic full-match delivery video sequences.
    """

    def __init__(
        self,
        resolution: Tuple[int, int] = (1280, 720),
        fps: float = 60.0,
        ball_type: str = "red"
    ):
        self.resolution = resolution
        self.fps = fps
        self.dt = 1.0 / fps
        self.ball_type = ball_type

        self.pitch = CricketPitch()
        self.camera = CameraModel(resolution=resolution)

    def _draw_stadium_and_pitch(self) -> np.ndarray:
        """Draws realistic stadium turf, pitch strip, creases, and stumps."""
        w, h = self.resolution
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Stadium outfield grass (gradient with lawn mower striped texture)
        base_grass = np.full((h, w, 3), [32, 130, 32], dtype=np.float32)
        vignette = 1.0 - 0.25 * ((np.arange(h) - h / 2.0) / (h / 2.0)) ** 2
        grass = (base_grass * vignette[:, None, None]).astype(np.uint8)

        # Lawn mower striping effect
        stripe_width = 30
        for y in range(0, h, stripe_width * 2):
            grass[y:min(h, y+stripe_width), :] = np.clip(
                grass[y:min(h, y+stripe_width), :].astype(np.int16) + 12, 0, 255
            ).astype(np.uint8)

        img = np.ascontiguousarray(grass)

        # Pitch clay strip
        poly_3d = self.pitch.get_pitch_boundary_polygon()
        poly_2d = []
        for pt in poly_3d:
            u, v, valid = self.camera.world_to_pixel(pt)
            if valid:
                poly_2d.append([u, v])

        if len(poly_2d) >= 4:
            pts = np.array(poly_2d, dtype=np.int32)
            # Tan clay color with subtle pitch wear patch
            cv2.fillPoly(img, [pts], (135, 175, 205), cv2.LINE_AA)

            # Central good-length wear zone (scuffed darker patch)
            wear_poly = np.array([
                self.camera.world_to_pixel(np.array([-0.6, 12.0, 0.0]))[0:2],
                self.camera.world_to_pixel(np.array([0.6, 12.0, 0.0]))[0:2],
                self.camera.world_to_pixel(np.array([0.6, 16.5, 0.0]))[0:2],
                self.camera.world_to_pixel(np.array([-0.6, 16.5, 0.0]))[0:2],
            ], dtype=np.int32)
            cv2.fillPoly(img, [wear_poly], (120, 160, 190), cv2.LINE_AA)

        # Crease markings (bright white)
        creases = self.pitch.get_crease_lines()
        for name, line in creases.items():
            u1, v1, val1 = self.camera.world_to_pixel(line[0])
            u2, v2, val2 = self.camera.world_to_pixel(line[1])
            if val1 and val2:
                cv2.line(img, (u1, v1), (u2, v2), (250, 250, 250), 3, cv2.LINE_AA)

        # Batsman Wickets (3 Stumps and Bails)
        wbox = self.pitch.get_wicket_bounding_box(is_batsman_end=True)
        stumps_x = [wbox["off_stump_x"], wbox["middle_stump_x"], wbox["leg_stump_x"]]
        for sx in stumps_x:
            b_u, b_v, _ = self.camera.world_to_pixel(np.array([sx, wbox["y"], 0.0]))
            t_u, t_v, _ = self.camera.world_to_pixel(np.array([sx, wbox["y"], self.pitch.dims.stump_height]))
            # Stump shadow
            s_u, s_v, _ = self.camera.world_to_pixel(np.array([sx + 0.15, wbox["y"] + 0.2, 0.0]))
            cv2.line(img, (b_u, b_v), (s_u, s_v), (70, 90, 110), 3, cv2.LINE_AA)
            # Stump cylinder
            cv2.line(img, (b_u, b_v), (t_u, t_v), (220, 225, 235), 4, cv2.LINE_AA)

        # Bails
        b1_u, b1_v, _ = self.camera.world_to_pixel(np.array([wbox["off_stump_x"], wbox["y"], self.pitch.dims.total_wicket_height]))
        b2_u, b2_v, _ = self.camera.world_to_pixel(np.array([wbox["leg_stump_x"], wbox["y"], self.pitch.dims.total_wicket_height]))
        cv2.line(img, (b1_u, b1_v), (b2_u, b2_v), (200, 210, 220), 3, cv2.LINE_AA)

        return img

    def generate_delivery(
        self,
        delivery_type: str = "inswinging_yorker",
        output_path: Optional[str] = None
    ) -> Tuple[str, List[TrajectoryPoint]]:
        """
        Generates and saves a synthetic match delivery video.
        Supported delivery types:
          - 'inswinging_yorker'
          - 'outswinger_good_length'
          - 'leg_break_spinner'
          - 'bouncer'
          - 'lbw_candidate'
          - 'reverse_swing_yorker'
        """
        # Aerodynamic and kinematic presets for delivery types
        presets = {
            "inswinging_yorker": {
                "r0": np.array([-0.30, 0.0, 2.10]),
                "v0": np.array([0.45, 40.0, -1.95]),  # ~144 km/h
                "swing": 0.35,                         # Inswing towards leg/middle
                "restitution": 0.54,
                "seam_dev": 0.015
            },
            "outswinger_good_length": {
                "r0": np.array([0.15, 0.0, 2.15]),
                "v0": np.array([-0.35, 38.3, -2.25]), # ~138 km/h
                "swing": -0.40,                        # Outswing away from off
                "restitution": 0.56,
                "seam_dev": -0.025
            },
            "leg_break_spinner": {
                "r0": np.array([0.10, 0.0, 1.95]),
                "v0": np.array([0.20, 24.5, -1.80]),  # ~88 km/h
                "swing": 0.10,
                "restitution": 0.52,
                "seam_dev": -0.095                     # Sharp leg break spin turning across off
            },
            "bouncer": {
                "r0": np.array([-0.20, 0.0, 2.25]),
                "v0": np.array([0.10, 41.0, -4.50]),  # ~148 km/h steep angle
                "swing": 0.05,
                "restitution": 0.62,                   # Hard bounce
                "seam_dev": 0.010
            },
            "lbw_candidate": {
                "r0": np.array([-0.25, 0.0, 2.08]),
                "v0": np.array([0.30, 39.2, -2.10]),  # ~141 km/h
                "swing": 0.22,
                "restitution": 0.53,
                "seam_dev": 0.012
            },
            "reverse_swing_yorker": {
                "r0": np.array([-0.35, 0.0, 2.05]),
                "v0": np.array([0.20, 40.5, -1.90]),  # ~146 km/h
                "swing": 0.48,                         # Late sharp reverse swing
                "restitution": 0.54,
                "seam_dev": 0.020
            }
        }

        cfg = presets.get(delivery_type, presets["inswinging_yorker"])
        props = AerodynamicProperties(
            swing_force=cfg["swing"],
            restitution_z=cfg["restitution"],
            seam_deviation_rad=cfg["seam_dev"]
        )
        sim = BallisticTrajectorySimulator(props=props, pitch=self.pitch)
        trajectory = sim.simulate_delivery(cfg["r0"], cfg["v0"], dt=self.dt, max_time=1.4)

        if output_path is None:
            output_dir = "data/samples"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{delivery_type}.mp4")

        # Setup OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = self.resolution
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        base_bg = self._draw_stadium_and_pitch()
        ball_col = (20, 20, 190) if self.ball_type == "red" else (240, 240, 245)

        for pt in trajectory:
            frame = base_bg.copy()
            u, v, valid = self.camera.world_to_pixel(pt.position)

            if valid and (0 <= u < w) and (0 <= v < h):
                # Depth based ball radius
                depth = max(1.5, pt.position[1])
                rad = max(3.0, 11.0 * (4.5 / depth))

                # Cast ground shadow beneath ball
                shadow_3d = pt.position.copy()
                shadow_3d[2] = 0.0
                su, sv, s_valid = self.camera.world_to_pixel(shadow_3d)
                if s_valid:
                    shadow_alpha = max(0.15, 0.55 * (1.0 - pt.position[2] / 2.5))
                    s_overlay = frame.copy()
                    cv2.ellipse(s_overlay, (su, sv), (int(rad * 1.4), int(rad * 0.6)), 0, 0, 360, (50, 60, 70), -1)
                    cv2.addWeighted(s_overlay, shadow_alpha, frame, 1.0 - shadow_alpha, 0, frame)

                # Directional motion blur vector
                vx, vy = pt.velocity[0], pt.velocity[1]
                v_norm = np.hypot(vx, vy)
                if v_norm > 5.0:
                    pt_future = pt.position + (pt.velocity / self.fps) * 0.4
                    fu, fv, _ = self.camera.world_to_pixel(pt_future)
                    blur_vec = (fu - u, fv - v)
                else:
                    blur_vec = (0, 0)

                # Render ball with motion blur steps
                if abs(blur_vec[0]) > 2 or abs(blur_vec[1]) > 2:
                    for step in range(3):
                        step_u = int(round(u - blur_vec[0] * (step / 3.0)))
                        step_v = int(round(v - blur_vec[1] * (step / 3.0)))
                        alpha = 0.35 + 0.2 * (1.0 - step / 3.0)
                        b_overlay = frame.copy()
                        cv2.circle(b_overlay, (step_u, step_v), int(round(rad)), ball_col, -1, cv2.LINE_AA)
                        cv2.addWeighted(b_overlay, alpha, frame, 1.0 - alpha, 0, frame)
                else:
                    cv2.circle(frame, (int(round(u)), int(round(v))), int(round(rad)), ball_col, -1, cv2.LINE_AA)
                    # Specular shine
                    cv2.circle(frame, (int(round(u - rad*0.3)), int(round(v - rad*0.3))), max(1, int(round(rad*0.25))), (255, 255, 255), -1)

            writer.write(frame)

        writer.release()
        return output_path, trajectory
