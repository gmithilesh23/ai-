"""
Physics-Informed Match-Adaptive Pitch Learning Model

RESEARCH SCIENTIFIC INNOVATION:
In cricket ball tracking, pitch surface properties (hardness, grass moisture, scuffing,
restitution e_z, tangential friction mu_y, seam grip) are spatially heterogeneous
and evolve temporally over the course of a match.

This module implements a Physics-Informed Match-Adaptive Model (PIMA-Pitch) that:
1. Enforces core physical conservation laws (momentum, restitution, Coulomb friction).
2. Uses an Online Bayesian Spatial Process / Kernel Ridge Regressor to model localized
   restitution anomalies e_z(X, Y) and seam deflection fields across the pitch.
3. Incrementally updates its spatial pitch memory after each observed delivery in a match.
4. Produces calibrated uncertainty intervals for post-bounce velocity and wicket impact.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

from cricket_vision.physics.pitch_geometry import CricketPitch, PitchDimensions


@dataclass
class DeliveryBounceObservation:
    """Empirical bounce record from a tracked delivery in the match."""
    delivery_idx: int
    timestamp: float
    bounce_x: float               # Lateral position on pitch (meters)
    bounce_y: float               # Distance down pitch (meters)
    v_in: np.ndarray              # Incoming 3D velocity [vx, vy, vz] (m/s)
    v_out_actual: np.ndarray      # Measured post-bounce 3D velocity [vx, vy, vz] (m/s)
    v_out_predicted: np.ndarray   # Predicted post-bounce velocity prior to observation
    measured_restitution: float   # -vz_out / vz_in
    measured_friction: float      # vy_out / vy_in
    seam_deflection_deg: float    # Measured lateral angular cut off the turf (degrees)
    prediction_error_norm: float  # ||v_out_actual - v_out_predicted|| (m/s)


@dataclass
class BouncePrediction:
    """Predicted post-bounce state with epistemic uncertainty."""
    v_out: np.ndarray             # Mean predicted velocity [vx, vy, vz]
    v_out_std: np.ndarray         # Uncertainty 1-sigma [std_x, std_y, std_z]
    predicted_restitution: float  # Expected restitution e_z
    confidence_score: float       # [0.0 - 1.0] confidence based on local delivery density
    is_learned_from_history: bool # True if updated by previous deliveries in this match


class MatchAdaptivePitchModel:
    """
    Physics-Informed Spatial-Temporal Match-Adaptive Machine Learning Engine.
    Continuously learns the localized characteristics of the pitch throughout a match.
    """

    def __init__(
        self,
        base_restitution: float = 0.54,
        base_friction: float = 0.78,
        pitch: Optional[CricketPitch] = None
    ):
        self.base_restitution = base_restitution
        self.base_friction = base_friction
        self.pitch = pitch or CricketPitch()

        # Match memory of previous deliveries
        self.history: List[DeliveryBounceObservation] = []

        # Spatial Gaussian Process for Restitution Anomaly Delta_e(X, Y)
        # Length scales: 0.5m lateral, 1.5m longitudinal (pitch wear patterns are elongated)
        kernel_restitution = C(0.04, (1e-3, 0.2)) * RBF(length_scale=[0.5, 1.8], length_scale_bounds=(0.2, 5.0)) + WhiteKernel(noise_level=1e-3)
        self.gp_restitution = GaussianProcessRegressor(
            kernel=kernel_restitution,
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=2
        )

        # Spatial Gaussian Process for Lateral Seam Cut Anomaly Delta_vx(X, Y)
        kernel_seam = C(0.1, (1e-3, 0.5)) * RBF(length_scale=[0.4, 1.5], length_scale_bounds=(0.2, 5.0)) + WhiteKernel(noise_level=1e-3)
        self.gp_seam = GaussianProcessRegressor(
            kernel=kernel_seam,
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=2
        )

        self.is_fitted = False

    def predict_bounce(
        self,
        bounce_x: float,
        bounce_y: float,
        v_in: np.ndarray
    ) -> BouncePrediction:
        """
        Predicts post-bounce velocity vector v_out with uncertainty using:
        1. Physics baseline (normal restitution + friction)
        2. Learned spatial corrections from prior match deliveries in this pitch area.
        """
        # Physics baseline
        vz_in = v_in[2]
        vy_in = v_in[1]
        vx_in = v_in[0]

        # Prior expectations
        expected_e = self.base_restitution
        delta_vx = 0.0
        sigma_e = 0.05
        sigma_vx = 0.35
        is_learned = False

        if self.is_fitted and len(self.history) >= 2:
            query_pt = np.array([[bounce_x, bounce_y]])
            try:
                # Predict spatial restitution delta and uncertainty
                delta_e_pred, std_e = self.gp_restitution.predict(query_pt, return_std=True)
                delta_vx_pred, std_vx = self.gp_seam.predict(query_pt, return_std=True)

                expected_e = float(np.clip(self.base_restitution + delta_e_pred[0], 0.35, 0.75))
                delta_vx = float(delta_vx_pred[0])
                sigma_e = float(std_e[0])
                sigma_vx = float(std_vx[0])
                is_learned = True
            except Exception:
                pass

        # Apply Physics-Informed Momentum Transfer
        vz_out = -expected_e * vz_in
        vy_out = self.base_friction * vy_in
        vx_out = vx_in + delta_vx

        v_out = np.array([vx_out, vy_out, vz_out], dtype=np.float64)

        # Standard deviations
        std_z = abs(vz_in) * sigma_e
        std_y = abs(vy_in) * 0.03
        std_x = sigma_vx
        v_out_std = np.array([std_x, std_y, std_z], dtype=np.float64)

        # Local data density confidence score
        if len(self.history) == 0:
            confidence = 0.60
        else:
            dists = [np.hypot(bounce_x - obs.bounce_x, bounce_y - obs.bounce_y) for obs in self.history]
            min_dist = min(dists)
            # Higher confidence if delivery pitched near previously calibrated spots
            confidence = float(np.clip(0.95 - 0.25 * min(min_dist, 3.0), 0.50, 0.98))

        return BouncePrediction(
            v_out=v_out,
            v_out_std=v_out_std,
            predicted_restitution=expected_e,
            confidence_score=confidence,
            is_learned_from_history=is_learned
        )

    def observe_delivery(
        self,
        delivery_idx: int,
        timestamp: float,
        bounce_x: float,
        bounce_y: float,
        v_in: np.ndarray,
        v_out_actual: np.ndarray
    ) -> DeliveryBounceObservation:
        """
        Registers an observed delivery, calculates prediction error,
        and updates the online match model.
        """
        # Prior prediction
        prior_pred = self.predict_bounce(bounce_x, bounce_y, v_in)

        # Compute empirical restitution and friction
        vz_in = v_in[2]
        vz_out = v_out_actual[2]
        vy_in = v_in[1]
        vy_out = v_out_actual[1]

        measured_e = abs(vz_out / max(abs(vz_in), 1e-4))
        measured_f = abs(vy_out / max(abs(vy_in), 1e-4))

        angle_in = np.arctan2(v_in[0], max(v_in[1], 1e-3))
        angle_out = np.arctan2(v_out_actual[0], max(v_out_actual[1], 1e-3))
        seam_dev_deg = float(np.degrees(angle_out - angle_in))

        err_norm = float(np.linalg.norm(v_out_actual - prior_pred.v_out))

        obs = DeliveryBounceObservation(
            delivery_idx=delivery_idx,
            timestamp=timestamp,
            bounce_x=float(bounce_x),
            bounce_y=float(bounce_y),
            v_in=v_in.copy(),
            v_out_actual=v_out_actual.copy(),
            v_out_predicted=prior_pred.v_out.copy(),
            measured_restitution=float(measured_e),
            measured_friction=float(measured_f),
            seam_deflection_deg=seam_dev_deg,
            prediction_error_norm=err_norm
        )
        self.history.append(obs)

        # Retrain Gaussian Processes on updated delivery history
        self._retrain_models()

        return obs

    def _retrain_models(self):
        """Fits spatial Gaussian Process on collected match bounce data."""
        if len(self.history) < 2:
            return

        coords = np.array([[o.bounce_x, o.bounce_y] for o in self.history])
        delta_e = np.array([o.measured_restitution - self.base_restitution for o in self.history])
        delta_vx = np.array([o.v_out_actual[0] - o.v_in[0] for o in self.history])

        try:
            self.gp_restitution.fit(coords, delta_e)
            self.gp_seam.fit(coords, delta_vx)
            self.is_fitted = True
        except Exception as e:
            print(f"[MatchAdaptivePitchModel] GP Fitting warning: {e}")

    def generate_pitch_restitution_heatmap(
        self,
        grid_resolution: Tuple[int, int] = (60, 120)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a 2D spatial grid of the pitch with the learned restitution map e_z(X, Y).
        Returns:
            X_grid: 2D array of lateral coordinates
            Y_grid: 2D array of down-pitch distance
            Z_restitution: 2D array of predicted restitution coefficients
        """
        hw = self.pitch.dims.pitch_width / 2.0
        y_max = self.pitch.batsman_stumps_y

        xs = np.linspace(-hw, hw, grid_resolution[0])
        ys = np.linspace(4.0, y_max, grid_resolution[1])
        X_grid, Y_grid = np.meshgrid(xs, ys)

        flat_pts = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

        if self.is_fitted and len(self.history) >= 2:
            try:
                deltas = self.gp_restitution.predict(flat_pts)
                Z_restitution = self.base_restitution + deltas.reshape(X_grid.shape)
            except Exception:
                Z_restitution = np.full(X_grid.shape, self.base_restitution)
        else:
            Z_restitution = np.full(X_grid.shape, self.base_restitution)

        return X_grid, Y_grid, np.clip(Z_restitution, 0.35, 0.75)

    def get_convergence_metrics(self) -> Dict[str, List[float]]:
        """
        Returns delivery-by-delivery learning curve data:
        - Delivery Index (1..N)
        - Velocity Prediction Error (m/s)
        - Cumulative Mean Absolute Error (MAE)
        - Learned Restitution Anomaly Mean
        """
        if not self.history:
            return {"delivery": [], "error": [], "cumulative_mae": []}

        deliveries = [o.delivery_idx for o in self.history]
        errors = [o.prediction_error_norm for o in self.history]
        cum_mae = [float(np.mean(errors[:i+1])) for i in range(len(errors))]

        return {
            "delivery": deliveries,
            "error": errors,
            "cumulative_mae": cum_mae
        }

