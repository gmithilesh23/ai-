"""Unit Tests for Cricket Physics & Aerodynamic Equations"""

import pytest
import numpy as np

from cricket_vision.physics.pitch_geometry import CricketPitch, PitchDimensions
from cricket_vision.physics.trajectory import (
    BallisticTrajectorySimulator,
    AerodynamicProperties,
    TrajectoryPoint
)


def test_pitch_dimensions():
    pitch = CricketPitch()
    assert abs(pitch.dims.length_between_stumps - 20.1168) < 1e-4
    assert abs(pitch.dims.stump_height - 0.7112) < 1e-4
    assert abs(pitch.dims.total_wicket_height - 0.7239) < 1e-4
    assert abs(pitch.dims.ball_radius - 0.036) < 1e-4


def test_pitch_length_classification():
    pitch = CricketPitch()
    pop_y = pitch.batsman_popping_crease_y  # ~18.8976m

    assert pitch.classify_length(pop_y - 1.0) == "Yorker"
    assert pitch.classify_length(pop_y - 2.5) == "Full"
    assert pitch.classify_length(pop_y - 4.5) == "Good Length"
    assert pitch.classify_length(pop_y - 7.0) == "Short of Length"
    assert pitch.classify_length(pop_y - 10.0) == "Bouncer"


def test_pitch_line_classification():
    pitch = CricketPitch()
    assert pitch.classify_line(0.0) == "In Line"
    assert pitch.classify_line(-0.35) == "Outside Off"
    assert pitch.classify_line(-0.80) == "Wide Outside Off"
    assert pitch.classify_line(0.25) == "Down Leg"


def test_rk4_simulation_and_bounce():
    sim = BallisticTrajectorySimulator()
    r0 = np.array([0.0, 0.0, 2.1])
    v0 = np.array([0.0, 38.0, -2.0])  # ~136 km/h

    traj = sim.simulate_delivery(r0, v0, dt=0.002, max_time=1.5)

    assert len(traj) > 20
    # First point is release
    assert traj[0].position[1] == 0.0
    # Flight progresses down pitch (increasing Y)
    assert traj[-1].position[1] > traj[0].position[1]
    # Check that bounce event occurred
    bounce_pts = [p for p in traj if p.phase == "bounce"]
    assert len(bounce_pts) == 1
    # Bounce elevation should be ball radius
    assert abs(bounce_pts[0].position[2] - sim.props.radius) < 1e-3


def test_drag_reduces_speed():
    sim = BallisticTrajectorySimulator()
    r0 = np.array([0.0, 0.0, 2.0])
    v0 = np.array([0.0, 40.0, 0.0])

    traj = sim.simulate_delivery(r0, v0, dt=0.005, max_time=0.4, stop_at_stumps=False)
    # Speed should strictly decrease due to aerodynamic drag
    speed_init = traj[0].speed_mps
    speed_later = traj[10].speed_mps
    assert speed_later < speed_init

