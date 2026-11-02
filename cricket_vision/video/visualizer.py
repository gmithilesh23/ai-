"""
Broadcast-Quality Visualizer for Cricket Ball Tracking & Hawk-Eye Overlays

Features:
- Glowing multi-phase trajectory trails (Pre-bounce, Post-bounce, Projected)
- Dynamic 3D pitch wireframe & crease alignment
- Pitch bounce spot projection with expanding concentric rings
- Broadcast telemetry HUD (Speed, Phase, Pitch length, Wicket line)
- Official Hawk-Eye DRS Review Decision banner overlay
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from cricket_vision.physics.homography import CameraModel
from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.hawk_eye import DRSReviewResult
from cricket_vision.tracking.tracker import TrackedBallState


class CricketVisualizer:
    """
    Renders broadcast graphics, tracking ribbons, and telemetry onto video frames.
    """

    def __init__(
        self,
        camera: Optional[CameraModel] = None,
        pitch: Optional[CricketPitch] = None
    ):
        self.camera = camera or CameraModel()
        self.pitch = pitch or CricketPitch()

        # Modern Broadcast Palette (BGR)
        self.COLOR_PRE_BOUNCE = (240, 180, 0)     # Neon Cyan / Sky Blue
        self.COLOR_POST_BOUNCE = (0, 120, 255)    # Fiery Orange
        self.COLOR_PROJECTED = (50, 255, 120)     # Vivid Lime Green
        self.COLOR_BOUNCE_RING = (0, 255, 255)    # Bright Yellow
        self.COLOR_STUMPS = (220, 220, 225)       # Off-white / Silver
        self.COLOR_HUD_BG = (20, 24, 30)          # Dark Slate
        self.COLOR_TEXT = (255, 255, 255)

    def draw_pitch_wireframe(self, frame: np.ndarray) -> np.ndarray:
        """Draws subtle 3D pitch boundary lines, creases, and stumps."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw Creases
        crease_dict = self.pitch.get_crease_lines()
        for name, line3d in crease_dict.items():
            u1, v1, val1 = self.camera.world_to_pixel(line3d[0])
            u2, v2, val2 = self.camera.world_to_pixel(line3d[1])
            if val1 and val2 and (0 <= u1 < w) and (0 <= u2 < w):
                cv2.line(out, (u1, v1), (u2, v2), (240, 240, 240), 2, cv2.LINE_AA)

        # Draw Batsman Stumps (Y=20.12m)
        wbox = self.pitch.get_wicket_bounding_box(is_batsman_end=True)
        stump_xs = [wbox["off_stump_x"], wbox["middle_stump_x"], wbox["leg_stump_x"]]
        for sx in stump_xs:
            b_u, b_v, b_val = self.camera.world_to_pixel(np.array([sx, wbox["y"], 0.0]))
            t_u, t_v, t_val = self.camera.world_to_pixel(np.array([sx, wbox["y"], self.pitch.dims.stump_height]))
            if b_val and t_val:
                cv2.line(out, (b_u, b_v), (t_u, t_v), self.COLOR_STUMPS, 3, cv2.LINE_AA)

        # Stumps Bails
        b1_u, b1_v, _ = self.camera.world_to_pixel(np.array([wbox["off_stump_x"], wbox["y"], self.pitch.dims.total_wicket_height]))
        b2_u, b2_v, _ = self.camera.world_to_pixel(np.array([wbox["leg_stump_x"], wbox["y"], self.pitch.dims.total_wicket_height]))
        if b1_u != -1 and b2_u != -1:
            cv2.line(out, (b1_u, b1_v), (b2_u, b2_v), (255, 255, 255), 2, cv2.LINE_AA)

        return out

    def draw_trajectory_ribbon(
        self,
        frame: np.ndarray,
        trajectory: List[TrackedBallState],
        projected_pts: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Renders a glowing 3D trajectory trail with thickness tapering and phase coloring.
        """
        out = frame.copy()
        if len(trajectory) < 2:
            return out

        # Extract 2D projected points and phases
        pixels = []
        phases = []
        for state in trajectory:
            u, v = state.pixel_coords
            if u != -1 and v != -1:
                pixels.append((u, v))
                phases.append("post" if state.is_bounced else "pre")

        # Draw glowing segments
        overlay = out.copy()
        for i in range(1, len(pixels)):
            p1 = pixels[i - 1]
            p2 = pixels[i]
            phase = phases[i]

            base_color = self.COLOR_POST_BOUNCE if phase == "post" else self.COLOR_PRE_BOUNCE

            # Progressively scale line thickness towards latest position
            progress = i / float(len(pixels))
            thickness = max(2, int(round(2 + 4 * progress)))

            # Outer glow
            cv2.line(overlay, p1, p2, base_color, thickness + 4, cv2.LINE_AA)
            # Inner bright core
            cv2.line(out, p1, p2, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

        # Draw projected future path if supplied
        if projected_pts and len(projected_pts) > 1:
            proj_pixels = []
            for pt in projected_pts:
                u, v, valid = self.camera.world_to_pixel(pt)
                if valid:
                    proj_pixels.append((u, v))

            proj_overlay = out.copy()
            for j in range(1, len(proj_pixels)):
                p1 = proj_pixels[j - 1]
                p2 = proj_pixels[j]
                # Dashed glow effect
                if j % 2 == 0:
                    cv2.line(proj_overlay, p1, p2, self.COLOR_PROJECTED, 4, cv2.LINE_AA)
                    cv2.line(out, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.addWeighted(proj_overlay, 0.5, out, 0.5, 0, out)

        return out

    def draw_bounce_spot(
        self,
        frame: np.ndarray,
        bounce_coords_3d: np.ndarray
    ) -> np.ndarray:
        """Renders an elliptical pitch impact spot on the turf."""
        out = frame.copy()
        u, v, valid = self.camera.world_to_pixel(bounce_coords_3d)
        if not valid:
            return out

        # Ground ellipse perspective
        cv2.ellipse(out, (u, v), (14, 6), 0, 0, 360, self.COLOR_BOUNCE_RING, 2, cv2.LINE_AA)
        cv2.circle(out, (u, v), 3, (255, 255, 255), -1, cv2.LINE_AA)

        # Text tag
        cv2.putText(out, "PITCH BOUNCE", (u + 16, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def draw_telemetry_hud(
        self,
        frame: np.ndarray,
        curr_state: Optional[TrackedBallState],
        drs_result: Optional[DRSReviewResult] = None
    ) -> np.ndarray:
        """
        Overlays broadcast telemetry HUD with speed, pitch length, and review verdict.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Top-Left Telemetry Card
        hud_w, hud_h = 280, 110
        card = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
        card[:] = self.COLOR_HUD_BG
        cv2.rectangle(card, (0, 0), (hud_w - 1, hud_h - 1), (60, 70, 85), 1)

        speed_kmh = curr_state.speed_kmh if curr_state else 0.0
        speed_mph = speed_kmh * 0.621371

        cv2.putText(card, "SPEED RADAR", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1, cv2.LINE_AA)
        cv2.putText(card, f"{speed_kmh:.1f}", (12, 60), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(card, "KM/H", (135, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(card, f"({speed_mph:.1f} MPH)", (190, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 185), 1, cv2.LINE_AA)

        phase_text = "PHASE: "
        if curr_state:
            phase_text += "POST-BOUNCE" if curr_state.is_bounced else "IN FLIGHT"
        else:
            phase_text += "SEARCHING"

        cv2.putText(card, phase_text, (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 220, 150), 1, cv2.LINE_AA)

        # Alpha blend HUD card onto frame
        x1, y1 = 20, 20
        out[y1:y1+hud_h, x1:x1+hud_w] = cv2.addWeighted(
            out[y1:y1+hud_h, x1:x1+hud_w], 0.2, card, 0.8, 0
        )

        # DRS Decision Card (Top Right if available)
        if drs_result is not None:
            drs_w, drs_h = 320, 140
            drs_card = np.zeros((drs_h, drs_w, 3), dtype=np.uint8)
            drs_card[:] = (15, 18, 24)
            cv2.rectangle(drs_card, (0, 0), (drs_w - 1, drs_h - 1), (80, 90, 110), 1)

            cv2.putText(drs_card, "HAWK-EYE DRS REVIEW", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 215, 0), 2, cv2.LINE_AA)

            # Pitching
            p_color = (0, 255, 0) if drs_result.pitching_verdict == "IN LINE" else (0, 165, 255)
            cv2.putText(drs_card, f"PITCHING: {drs_result.pitching_verdict}", (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, p_color, 1, cv2.LINE_AA)

            # Impact
            i_color = (0, 255, 0) if drs_result.impact_verdict == "IN LINE" else (0, 215, 255)
            cv2.putText(drs_card, f"IMPACT:   {drs_result.impact_verdict}", (14, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, i_color, 1, cv2.LINE_AA)

            # Wickets
            w_color = (0, 0, 255) if drs_result.wickets_verdict == "HITTING" else ((0, 255, 0) if drs_result.wickets_verdict == "MISSING" else (0, 215, 255))
            cv2.putText(drs_card, f"WICKETS:  {drs_result.wickets_verdict} ({drs_result.stump_targeted})", (14, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, w_color, 1, cv2.LINE_AA)

            # Overall Decision
            dec_color = (0, 0, 255) if "OUT" in drs_result.overall_lbw_decision and "NOT" not in drs_result.overall_lbw_decision else (0, 255, 0)
            cv2.putText(drs_card, f"DECISION: {drs_result.overall_lbw_decision}", (14, 126), cv2.FONT_HERSHEY_DUPLEX, 0.48, dec_color, 1, cv2.LINE_AA)

            dx1 = w - drs_w - 20
            dy1 = 20
            out[dy1:dy1+drs_h, dx1:dx1+drs_w] = cv2.addWeighted(
                out[dy1:dy1+drs_h, dx1:dx1+drs_w], 0.15, drs_card, 0.85, 0
            )

        return out
