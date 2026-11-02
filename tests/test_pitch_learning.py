"""Unit Tests for Match-Adaptive Pitch Learning Model (Research Innovation)"""

import numpy as np
from cricket_vision.physics.pitch_learning import MatchAdaptivePitchModel


def test_prior_prediction_without_data():
    model = MatchAdaptivePitchModel(base_restitution=0.54)
    v_in = np.array([0.0, 35.0, -4.0])

    pred = model.predict_bounce(0.0, 14.0, v_in)
    # Without data, expected restitution is base_restitution
    assert abs(pred.predicted_restitution - 0.54) < 1e-4
    assert abs(pred.v_out[2] - (-0.54 * v_in[2])) < 1e-3
    assert pred.is_learned_from_history is False


def test_online_adaptation_to_hard_pitch_patch():
    # Simulate a hard patch at X=0.0, Y=14.0 where pitch has higher restitution e=0.68
    model = MatchAdaptivePitchModel(base_restitution=0.54)
    v_in = np.array([0.0, 36.0, -4.0])

    # Delivery #1 lands on hard patch with true restitution e=0.68
    v_out_1 = np.array([0.0, 28.0, 0.68 * 4.0])  # vz_out = 2.72 m/s
    obs1 = model.observe_delivery(
        delivery_idx=1,
        timestamp=0.3,
        bounce_x=0.0,
        bounce_y=14.0,
        v_in=v_in,
        v_out_actual=v_out_1
    )
    # First delivery has higher error because prior was 0.54
    assert obs1.prediction_error_norm > 0.4

    # Delivery #2 lands near the same spot
    v_out_2 = np.array([0.0, 28.0, 0.68 * 4.0])
    obs2 = model.observe_delivery(
        delivery_idx=2,
        timestamp=0.6,
        bounce_x=0.05,
        bounce_y=14.1,
        v_in=v_in,
        v_out_actual=v_out_2
    )

    # Now predict for Delivery #3 in that same area
    pred3 = model.predict_bounce(0.02, 14.05, v_in)

    # Model should have learned higher restitution for this zone!
    assert pred3.predicted_restitution > 0.58
    assert pred3.is_learned_from_history is True

    # Check error convergence
    conv = model.get_convergence_metrics()
    assert len(conv["delivery"]) == 2

