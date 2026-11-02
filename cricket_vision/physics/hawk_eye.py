"""
Hawk-Eye DRS (Decision Review System) & LBW Adjudication Engine

Implements the official ICC Decision Review System logic:
1. Pitching zone evaluation (In-Line, Outside Off, Outside Leg)
2. Impact zone evaluation (In-Line, Outside, Umpire's Call)
3. Wicket hitting projection (Hitting, Missing, Umpire's Call)
4. Comprehensive delivery telemetry (Release speed, bounce speed, deviation, swing)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from cricket_vision.physics.pitch_geometry import CricketPitch, PitchDimensions
from cricket_vision.physics.trajectory import TrajectoryPoint, BallisticTrajectorySimulator


@dataclass
class DRSReviewResult:
    """Official ICC DRS Review Outcome & Telemetry."""
    # Core DRS Verdicts
    pitching_verdict: str         # "IN LINE", "OUTSIDE OFF", "OUTSIDE LEG"
    impact_verdict: str           # "IN LINE", "OUTSIDE OFF", "UMPIRE'S CALL"
    wickets_verdict: str          # "HITTING", "MISSING", "UMPIRE'S CALL"
    overall_lbw_decision: str     # "OUT", "NOT OUT", "UMPIRE'S CALL"

    # Specific details
    stump_targeted: str           # e.g. "Middle Stump", "Off Stump", "Top of Off", "Missing Over"
    pitching_coords: Tuple[float, float]     # (X, Y) of bounce in meters
    impact_coords: Tuple[float, float, float] # (X, Y, Z) of pad impact in meters
    stumps_arrival_coords: Tuple[float, float, float] # (X, Y, Z) at stumps plane Y=20.12m

    # Velocity & Movement Telemetry
    release_speed_kmh: float
    bounce_speed_kmh: float
    stumps_speed_kmh: float
    speed_drop_pct: float
    flight_time_s: float
    length_category: str          # "Yorker", "Full", "Good Length", "Short", "Bouncer"
    line_category: str            # "In Line", "Outside Off", etc.
    pre_bounce_swing_deg: float
    pre_bounce_swing_cm: float
    post_bounce_seam_deg: float
    post_bounce_seam_cm: float
    impact_height_m: float
    stumps_height_m: float


class HawkEyeDRSSystem:
    """
    Simulates the broadcast Hawk-Eye DRS technology for cricket.
    """

    def __init__(self, pitch: Optional[CricketPitch] = None):
        self.pitch = pitch or CricketPitch()
        self.dims = self.pitch.dims
        self.half_wicket_w = self.dims.wicket_width / 2.0  # ~0.1143 m
        self.ball_radius = self.dims.ball_radius           # 0.036 m
        self.wicket_height = self.dims.total_wicket_height # 0.7239 m

    def evaluate_delivery(
        self,
        full_trajectory: List[TrajectoryPoint],
        impact_index: Optional[int] = None
    ) -> DRSReviewResult:
        """
        Analyzes a completed delivery trajectory and evaluates the DRS review.
        """
        if len(full_trajectory) < 5:
            raise ValueError("Trajectory too short for DRS evaluation")

        # 1. Identify Key Landmarks: Release, Bounce, Impact, Stumps
        release_pt = full_trajectory[0]

        # Find bounce point (minimum Z or explicit 'bounce' phase)
        bounce_candidates = [pt for pt in full_trajectory if pt.phase == "bounce"]
        if bounce_candidates:
            bounce_pt = bounce_candidates[0]
        else:
            # Find point of minimum elevation
            z_vals = [pt.position[2] for pt in full_trajectory]
            min_idx = int(np.argmin(z_vals))
            bounce_pt = full_trajectory[min_idx]

        # Determine Impact Point (e.g. pad contact near batsman popping crease)
        if impact_index is not None and impact_index < len(full_trajectory):
            impact_pt = full_trajectory[impact_index]
        else:
            # Default to trajectory point nearest batsman popping crease (Y = 18.9m)
            popping_y = self.pitch.batsman_popping_crease_y
            y_diffs = [abs(pt.position[1] - popping_y) for pt in full_trajectory]
            best_impact_idx = int(np.argmin(y_diffs))
            impact_pt = full_trajectory[best_impact_idx]

        # Stumps Arrival Point (at Y = 20.1168 m)
        stumps_y = self.pitch.batsman_stumps_y
        y_stump_diffs = [abs(pt.position[1] - stumps_y) for pt in full_trajectory]
        stumps_idx = int(np.argmin(y_stump_diffs))
        stumps_pt = full_trajectory[stumps_idx]

        # 2. Evaluate DRS Criteria: PITCHING
        # Pitching zone: between outer stump lines (-half_w to +half_w)
        bx, by = bounce_pt.position[0], bounce_pt.position[1]
        if bx < -self.half_wicket_w:
            pitching_verdict = "OUTSIDE OFF"
        elif bx > self.half_wicket_w:
            pitching_verdict = "OUTSIDE LEG"
        else:
            pitching_verdict = "IN LINE"

        # 3. Evaluate DRS Criteria: IMPACT
        # Impact zone: evaluated at impact_pt [X, Y, Z]
        ix, iy, iz = impact_pt.position[0], impact_pt.position[1], impact_pt.position[2]
        dist_from_stump_line = abs(ix) - self.half_wicket_w

        if dist_from_stump_line <= 0:
            impact_verdict = "IN LINE"
        elif dist_from_stump_line <= self.ball_radius:
            # Within 50% ball margin -> Umpire's Call
            impact_verdict = "UMPIRE'S CALL"
        else:
            impact_verdict = "OUTSIDE OFF" if ix < 0 else "OUTSIDE LEG"

        # 4. Evaluate DRS Criteria: WICKETS
        sx, sy, sz = stumps_pt.position[0], stumps_pt.position[1], stumps_pt.position[2]

        # Check horizontal wicket bounds
        horiz_inside = abs(sx) <= self.half_wicket_w
        horiz_margin = abs(sx) <= (self.half_wicket_w + self.ball_radius)

        # Check vertical wicket bounds
        vert_inside = (0.0 <= sz <= self.wicket_height)
        vert_margin = (0.0 <= sz <= self.wicket_height + self.ball_radius)

        if horiz_inside and vert_inside:
            wickets_verdict = "HITTING"
        elif horiz_margin and vert_margin:
            wickets_verdict = "UMPIRE'S CALL"
        else:
            wickets_verdict = "MISSING"

        # Identify which specific stump was struck or why it missed
        if wickets_verdict in ["HITTING", "UMPIRE'S CALL"]:
            # Stumps width split into thirds
            one_third = self.dims.wicket_width / 3.0
            if sx < -one_third / 2.0:
                stump_targeted = "Off Stump" if sz < self.dims.stump_height else "Top of Off"
            elif sx > one_third / 2.0:
                stump_targeted = "Leg Stump" if sz < self.dims.stump_height else "Top of Leg"
            else:
                stump_targeted = "Middle Stump" if sz < self.dims.stump_height else "Bails"
        else:
            if sz > self.wicket_height + self.ball_radius:
                stump_targeted = "Missing (Over Stumps)"
            elif sx < -self.half_wicket_w:
                stump_targeted = "Missing (Off Side)"
            else:
                stump_targeted = "Missing (Leg Side)"

        # 5. Overall LBW Decision (ICC Law 36)
        # If pitching outside leg -> Not Out automatically
        if pitching_verdict == "OUTSIDE LEG":
            overall_decision = "NOT OUT (Pitched Outside Leg)"
        elif impact_verdict in ["OUTSIDE OFF", "OUTSIDE LEG"]:
            overall_decision = "NOT OUT (Impact Outside Off)"
        elif wickets_verdict == "MISSING":
            overall_decision = "NOT OUT (Wickets Missing)"
        elif wickets_verdict == "HITTING" and impact_verdict == "IN LINE":
            overall_decision = "OUT"
        else:
            overall_decision = "UMPIRE'S CALL"

        # 6. Movement & Telemetry Analytics
        rel_speed = release_pt.speed_kmh
        bnc_speed = bounce_pt.speed_kmh
        stm_speed = stumps_pt.speed_kmh
        speed_drop = ((rel_speed - bnc_speed) / max(rel_speed, 1.0)) * 100.0
        total_time = stumps_pt.time - release_pt.time

        length_cat = self.pitch.classify_length(bounce_pt.position[1])
        line_cat = self.pitch.classify_line(impact_pt.position[0])

        # Lateral swing measurement (pre-bounce)
        # Difference between straight line from release to bounce and actual curved path
        p_rel = release_pt.position
        p_bnc = bounce_pt.position
        straight_x_mid = 0.5 * (p_rel[0] + p_bnc[0])
        # Find actual ball position halfway between release and bounce
        pre_bounce_pts = [pt for pt in full_trajectory if pt.position[1] < p_bnc[1]]
        if len(pre_bounce_pts) > 2:
            mid_pt = pre_bounce_pts[len(pre_bounce_pts) // 2]
            swing_disp = (mid_pt.position[0] - straight_x_mid)
        else:
            swing_disp = 0.0

        swing_deg = np.degrees(np.arctan2(abs(swing_disp), max(p_bnc[1] - p_rel[1], 1.0)))
        swing_cm = abs(swing_disp) * 100.0

        # Post-bounce seam / cut deviation measurement
        # Angle change between incoming velocity vector and post-bounce velocity vector
        post_pts = [pt for pt in full_trajectory if pt.position[1] >= p_bnc[1]]
        if len(post_pts) >= 2:
            v_in = bounce_pt.velocity
            v_out = post_pts[1].velocity
            angle_in = np.arctan2(v_in[0], v_in[1])
            angle_out = np.arctan2(v_out[0], v_out[1])
            seam_deg = abs(np.degrees(angle_out - angle_in))
            seam_cm = abs(np.tan(angle_out - angle_in) * (stumps_pt.position[1] - p_bnc[1])) * 100.0
        else:
            seam_deg = 0.0
            seam_cm = 0.0

        return DRSReviewResult(
            pitching_verdict=pitching_verdict,
            impact_verdict=impact_verdict,
            wickets_verdict=wickets_verdict,
            overall_lbw_decision=overall_decision,
            stump_targeted=stump_targeted,
            pitching_coords=(round(float(bx), 4), round(float(by), 4)),
            impact_coords=(round(float(ix), 4), round(float(iy), 4), round(float(iz), 4)),
            stumps_arrival_coords=(round(float(sx), 4), round(float(sy), 4), round(float(sz), 4)),
            release_speed_kmh=round(rel_speed, 1),
            bounce_speed_kmh=round(bnc_speed, 1),
            stumps_speed_kmh=round(stm_speed, 1),
            speed_drop_pct=round(speed_drop, 1),
            flight_time_s=round(total_time, 3),
            length_category=length_cat,
            line_category=line_cat,
            pre_bounce_swing_deg=round(swing_deg, 2),
            pre_bounce_swing_cm=round(swing_cm, 1),
            post_bounce_seam_deg=round(seam_deg, 2),
            post_bounce_seam_cm=round(seam_cm, 1),
            impact_height_m=round(float(iz), 3),
            stumps_height_m=round(float(sz), 3)
        )
