"""Unit Tests for Kalman Filtering, Gating & Bounce Detection"""

import numpy as np
from cricket_vision.tracking.kalman_filter import BallisticKalmanFilter
from cricket_vision.tracking.bounce_detector import BounceDetector


def test_kalman_filter_prediction_and_update():
    kf = BallisticKalmanFilter(dt=1.0/60.0)
    init_pos = np.array([0.0, 1.0, 2.0])
    init_vel = np.array([0.0, 35.0, -1.0])
    kf.initialize(init_pos, init_vel)

    pred_pos = kf.predict()
    # Y should advance forward
    assert pred_pos[1] > init_pos[1]
    # Z should drop under gravity
    assert pred_pos[2] < init_pos[2]

    # Clean measurement update
    meas = pred_pos + np.array([0.01, -0.01, 0.005])
    accepted, d_m = kf.update(meas)
    assert accepted is True
    assert d_m < 5.0


def test_kalman_mahalanobis_gate_rejects_spurious_outlier():
    kf = BallisticKalmanFilter(dt=1.0/60.0)
    kf.initialize(np.array([0.0, 5.0, 1.8]), np.array([0.0, 30.0, -2.0]))
    kf.predict()

    # Outlier 10 meters away
    wild_outlier = np.array([10.0, 15.0, 8.0])
    accepted, d_m = kf.update(wild_outlier)
    assert accepted is False
    assert d_m > 16.0


def test_bounce_detector_identifies_reversal():
    detector = BounceDetector()
    pos_hist = [
        np.array([0.0, 14.0, 0.35]),
        np.array([0.0, 14.5, 0.08]),
        np.array([0.0, 15.0, 0.15])
    ]
    vel_hist = [
        np.array([0.0, 30.0, -3.2]),
        np.array([0.0, 25.0, -1.5]),
        np.array([0.0, 24.0, 1.8])  # Reversal vz < 0 to vz > 0
    ]
    accel_hist = [
        np.array([0.0, 0.0, -9.8]),
        np.array([0.0, 0.0, 65.0]), # Impact spike
        np.array([0.0, 0.0, -9.8])
    ]

    event = detector.check_bounce(12, 0.20, pos_hist, vel_hist, accel_hist)
    assert event is not None
    assert event.restitution_ratio > 0.35
    assert event.position[2] == 0.036

