"""Mathematical & Physical Helper Functions for Cricket Tracking"""

from typing import Tuple
import numpy as np


def mps_to_kmh(speed_mps: float) -> float:
    """Converts meters per second to kilometers per hour."""
    return speed_mps * 3.6


def kmh_to_mps(speed_kmh: float) -> float:
    """Converts kilometers per hour to meters per second."""
    return speed_kmh / 3.6


def mps_to_mph(speed_mps: float) -> float:
    """Converts meters per second to miles per hour."""
    return speed_mps * 2.23694


def meters_to_inches(dist_m: float) -> float:
    """Converts meters to inches."""
    return dist_m * 39.3701


def normalize_angle_rad(angle: float) -> float:
    """Wraps angle into [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_aerodynamic_drag_force(
    velocity: np.ndarray,
    drag_coeff: float = 0.44,
    air_density: float = 1.225,
    ball_radius: float = 0.036
) -> np.ndarray:
    """Computes instantaneous 3D aerodynamic drag force vector."""
    speed = np.linalg.norm(velocity)
    area = np.pi * (ball_radius ** 2)
    return -0.5 * air_density * drag_coeff * area * speed * velocity
