"""
ICC Cricket Pitch Specifications & Coordinate System

Coordinate Frame:
  Origin (0,0,0): Ground level directly beneath bowler's middle stump.
  +Y axis: Points down pitch towards batsman's middle stump (0.0 to 20.1168m).
  +X axis: Points horizontally towards batsman's leg side (for Right Hand Batsman).
  -X axis: Points towards batsman's off side.
  +Z axis: Points vertically upwards normal to pitch turf.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class PitchDimensions:
    """Standard ICC Pitch Dimensions in SI meters."""
    length_between_stumps: float = 20.1168  # 22 yards
    pitch_width: float = 3.048              # 10 feet
    popping_crease_offset: float = 1.2192   # 4 feet in front of bowling crease
    bowling_crease_width: float = 2.6416    # 8 feet 8 inches
    return_crease_length: float = 2.4384    # 8 feet
    stump_height: float = 0.7112            # 28 inches
    bails_height: float = 0.0127            # 0.5 inches
    total_wicket_height: float = 0.7239     # 28.5 inches
    wicket_width: float = 0.2286            # 9 inches
    ball_radius: float = 0.036              # 3.6 cm


class CricketPitch:
    """
    Cricket pitch spatial reference system and zone analysis.
    Provides canonical 3D geometry of creases, wickets, and landing zones.
    """

    def __init__(self, dims: Optional[PitchDimensions] = None):
        self.dims = dims or PitchDimensions()
        self.batsman_stumps_y = self.dims.length_between_stumps
        self.batsman_popping_crease_y = self.batsman_stumps_y - self.dims.popping_crease_offset
        self.bowler_popping_crease_y = self.dims.popping_crease_offset

    def get_wicket_bounding_box(self, is_batsman_end: bool = True) -> Dict[str, float]:
        """
        Returns spatial bounding box of the 3 stumps and bails.
        """
        y_pos = self.batsman_stumps_y if is_batsman_end else 0.0
        half_w = self.dims.wicket_width / 2.0
        return {
            "x_min": -half_w,
            "x_max": half_w,
            "y": y_pos,
            "z_min": 0.0,
            "z_max": self.dims.total_wicket_height,
            "off_stump_x": -half_w + (self.dims.wicket_width / 4.0),
            "middle_stump_x": 0.0,
            "leg_stump_x": half_w - (self.dims.wicket_width / 4.0),
        }

    def get_pitch_boundary_polygon(self) -> np.ndarray:
        """Returns 3D vertices of the rectangular pitch playing surface."""
        hw = self.dims.pitch_width / 2.0
        return np.array([
            [-hw, -1.0, 0.0],
            [hw, -1.0, 0.0],
            [hw, self.batsman_stumps_y + 1.0, 0.0],
            [-hw, self.batsman_stumps_y + 1.0, 0.0]
        ], dtype=np.float64)

    def get_crease_lines(self) -> Dict[str, np.ndarray]:
        """
        Returns pairs of 3D points representing ICC crease markings:
        - Bowling crease (bowler & batsman)
        - Popping crease (bowler & batsman)
        - Return creases
        """
        hw_bowl = self.dims.bowling_crease_width / 2.0
        lines = {
            "bowler_bowling_crease": np.array([[-hw_bowl, 0.0, 0.0], [hw_bowl, 0.0, 0.0]]),
            "bowler_popping_crease": np.array([[-1.83, self.bowler_popping_crease_y, 0.0],
                                               [1.83, self.bowler_popping_crease_y, 0.0]]),
            "batsman_bowling_crease": np.array([[-hw_bowl, self.batsman_stumps_y, 0.0],
                                                [hw_bowl, self.batsman_stumps_y, 0.0]]),
            "batsman_popping_crease": np.array([[-1.83, self.batsman_popping_crease_y, 0.0],
                                                [1.83, self.batsman_popping_crease_y, 0.0]]),
        }
        return lines

    def classify_length(self, bounce_y: float) -> str:
        """
        Classifies delivery length based on bounce distance from batsman's popping crease.
        """
        dist_from_crease = self.batsman_popping_crease_y - bounce_y

        if dist_from_crease <= 1.5:
            return "Yorker"
        elif dist_from_crease <= 3.5:
            return "Full"
        elif dist_from_crease <= 6.0:
            return "Good Length"
        elif dist_from_crease <= 8.5:
            return "Short of Length"
        else:
            return "Bouncer"

    def classify_line(self, x_coord: float) -> str:
        """
        Classifies delivery line relative to RHB stumps.
        - Negative X: Off Side
        - Around 0.0: Stumps Line
        - Positive X: Leg Side
        """
        half_w = self.dims.wicket_width / 2.0
        if x_coord < -half_w - 0.45:
            return "Wide Outside Off"
        elif x_coord < -half_w:
            return "Outside Off"
        elif x_coord <= half_w:
            return "In Line"
        elif x_coord <= half_w + 0.35:
            return "Down Leg"
        else:
            return "Wide Down Leg"

    def is_within_pitch(self, x: float, y: float) -> bool:
        """Checks if a ground impact point lies within pitch boundaries."""
        hw = self.dims.pitch_width / 2.0
        return (-hw <= x <= hw) and (-1.0 <= y <= self.batsman_stumps_y + 1.0)
