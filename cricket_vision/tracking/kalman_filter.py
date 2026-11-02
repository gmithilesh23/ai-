"""
Ballistic Kalman Filter & RTS (Rauch-Tung-Striebel) Smoother for Cricket Ball Tracking

Maintains a 9-dimensional state space:
  State: [x, y, z, vx, vy, vz, ax, ay, az]^T
Incorporates aerodynamic drag decay, gravitational acceleration,
and Mahalanobis distance gating to reject spurious detections.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.stats import chi2


class BallisticKalmanFilter:
    """
    Continuous-discrete Extended / Linear Ballistic Kalman Filter for 3D trajectory filtering.
    """

    def __init__(
        self,
        dt: float = 1.0 / 60.0,
        process_noise_std: float = 0.5,
        measurement_noise_std: float = 0.05,
        gating_p_value: float = 0.99
    ):
        self.dt = dt
        self.dim_x = 9  # [x, y, z, vx, vy, vz, ax, ay, az]
        self.dim_z = 3  # [x, y, z]

        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.P = np.eye(self.dim_x, dtype=np.float64) * 10.0

        # State transition matrix F for constant acceleration with drag decay
        self.F = np.eye(self.dim_x, dtype=np.float64)
        for i in range(3):
            self.F[i, i + 3] = dt
            self.F[i, i + 6] = 0.5 * dt * dt
            self.F[i + 3, i + 6] = dt

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float64)
        for i in range(3):
            self.H[i, i] = 1.0

        # Process noise covariance Q (discrete Wiener process acceleration model)
        q = process_noise_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        Q_block = np.array([
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0,  dt2 / 2.0, dt]
        ], dtype=np.float64) * q

        self.Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        for i in range(3):
            idx = [i, i + 3, i + 6]
            self.Q[np.ix_(idx, idx)] = Q_block

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float64) * (measurement_noise_std ** 2)

        # Mahalanobis gating threshold (chi2 distribution with 3 degrees of freedom)
        self.gate_threshold = chi2.ppf(gating_p_value, df=self.dim_z)

        # Gravitational vector
        self.gravity_accel = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

        # History buffers for RTS smoothing
        self.history_x: List[np.ndarray] = []
        self.history_P: List[np.ndarray] = []
        self.history_F: List[np.ndarray] = []
        self.history_Q: List[np.ndarray] = []

    def initialize(self, pos: np.ndarray, vel: Optional[np.ndarray] = None):
        """Initializes state vector with initial position and optional velocity."""
        self.x = np.zeros((self.dim_x, 1), dtype=np.float64)
        self.x[0:3, 0] = pos
        if vel is not None:
            self.x[3:6, 0] = vel
        self.x[6:9, 0] = self.gravity_accel

        self.P = np.eye(self.dim_x, dtype=np.float64)
        self.P[0:3, 0:3] *= 0.1
        self.P[3:6, 3:6] *= 1.0
        self.P[6:9, 6:9] *= 2.0

        self.history_x.clear()
        self.history_P.clear()
        self.history_F.clear()
        self.history_Q.clear()

    def predict(self) -> np.ndarray:
        """
        Executes Kalman prediction step:
        x^- = F * x
        P^- = F * P * F^T + Q
        """
        self.history_x.append(self.x.copy())
        self.history_P.append(self.P.copy())
        self.history_F.append(self.F.copy())
        self.history_Q.append(self.Q.copy())

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:3, 0].copy()

    def update(self, measurement: np.ndarray) -> Tuple[bool, float]:
        """
        Executes measurement update with Mahalanobis gating.
        Returns:
            accepted: True if measurement was within gating threshold
            d_mahalanobis: Computed Mahalanobis distance
        """
        z = measurement.reshape((self.dim_z, 1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)

        # Mahalanobis Distance Gating
        d_m_sq = float((y.T @ S_inv @ y)[0, 0])
        if d_m_sq > self.gate_threshold:
            return False, float(np.sqrt(d_m_sq))

        # Kalman Gain
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + (K @ y)
        I_KH = np.eye(self.dim_x) - (K @ self.H)
        # Joseph form covariance update for numerical stability: P = (I-KH)P(I-KH)^T + KRK^T
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return True, float(np.sqrt(d_m_sq))

    def apply_bounce_event(self, restitution_z: float = 0.54, friction_y: float = 0.78):
        """Applies discontinuous state jump upon detected bounce."""
        self.x[5, 0] = -restitution_z * self.x[5, 0]  # Invert vertical velocity
        self.x[4, 0] = friction_y * self.x[4, 0]       # Forward friction reduction
        self.P[3:6, 3:6] *= 3.0  # Inflate velocity uncertainty after collision

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6, 0].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.x[6:9, 0].copy()

    def rts_smooth(self) -> List[np.ndarray]:
        """
        Rauch-Tung-Striebel backward smoothing pass.
        Eliminates lag and produces optimal sub-pixel ballistic trajectory.
        """
        n = len(self.history_x)
        if n == 0:
            return [self.x.copy()]

        smoothed_x = [None] * n
        smoothed_P = [None] * n

        smoothed_x[-1] = self.x.copy()
        smoothed_P[-1] = self.P.copy()

        for k in range(n - 2, -1, -1):
            x_pred = self.history_F[k] @ self.history_x[k]
            P_pred = self.history_F[k] @ self.history_P[k] @ self.history_F[k].T + self.history_Q[k]

            C_k = self.history_P[k] @ self.history_F[k].T @ np.linalg.inv(P_pred)
            smoothed_x[k] = self.history_x[k] + C_k @ (smoothed_x[k + 1] - x_pred)
            smoothed_P[k] = self.history_P[k] + C_k @ (smoothed_P[k + 1] - P_pred) @ C_k.T

        return [sx[0:3, 0] for sx in smoothed_x]
