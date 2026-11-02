"""
Aerodynamic & Ballistic Trajectory Simulation Engine for Cricket Deliveries

Implements:
- 3D flight dynamics with atmospheric drag, gravity, and Magnus/seam forces
- Runge-Kutta 4th-order (RK4) numerical ODE solver
- Ground bounce mechanics (restitution, friction, seam cut & spin turn)
- Trajectory forward projection for Hawk-Eye DRS analysis
- Non-linear least-squares trajectory curve fitting to tracklet observations
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import minimize

from cricket_vision.physics.pitch_geometry import CricketPitch, PitchDimensions


@dataclass
class AerodynamicProperties:
    """Aerodynamic parameters governing ball flight."""
    mass: float = 0.160                 # kg
    radius: float = 0.036               # m
    air_density: float = 1.225          # kg/m^3
    drag_coeff: float = 0.44            # Cd
    magnus_coeff: float = 0.22          # Cl
    seam_angle_deg: float = 0.0         # Seam tilt (-15 to +15 deg)
    swing_force: float = 0.0            # Lateral aerodynamic swing force (N)
    restitution_z: float = 0.54         # Vertical restitution coefficient
    friction_y: float = 0.78            # Tangential friction coefficient
    seam_deviation_rad: float = 0.0     # Angular deviation at bounce (cut/spin)

    @property
    def cross_sectional_area(self) -> float:
        return np.pi * (self.radius ** 2)


@dataclass
class TrajectoryPoint:
    """Spatial and kinematic state at a discrete simulation epoch."""
    time: float
    position: np.ndarray      # [x, y, z] in meters
    velocity: np.ndarray      # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s^2
    phase: str                # 'flight_pre_bounce', 'bounce', 'flight_post_bounce', 'projected'

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6

    @property
    def speed_mph(self) -> float:
        return self.speed_mps * 2.23694


class BallisticTrajectorySimulator:
    """
    High-fidelity physics simulator for cricket ball flight and pitch interaction.
    """

    def __init__(
        self,
        props: Optional[AerodynamicProperties] = None,
        pitch: Optional[CricketPitch] = None,
        pitch_model: Optional[object] = None
    ):
        self.props = props or AerodynamicProperties()
        self.pitch = pitch or CricketPitch()
        self.pitch_model = pitch_model
        self.gravity = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

    def compute_acceleration(self, velocity: np.ndarray, spin_vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes net instantaneous acceleration including gravity, drag, and swing/Magnus force.
        a = g - (1/2 * rho * Cd * A / m) * |v| * v + F_swing / m
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-4:
            return self.gravity.copy()

        # Aerodynamic Drag
        drag_const = 0.5 * self.props.air_density * self.props.drag_coeff * self.props.cross_sectional_area
        f_drag = -drag_const * speed * velocity
        a_drag = f_drag / self.props.mass

        # Lateral Aerodynamic Swing (perpendicular to flight path in horizontal plane)
        # Swing force acts laterally on the X-axis proportional to forward speed squared
        v_horiz = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if v_horiz > 1.0:
            lateral_unit = np.array([-velocity[1] / v_horiz, velocity[0] / v_horiz, 0.0], dtype=np.float64)
            a_swing = (self.props.swing_force * (speed / 35.0)**2 * lateral_unit) / self.props.mass
        else:
            a_swing = np.zeros(3, dtype=np.float64)

        # Magnus force from spin vector omega (rad/s) if supplied
        if spin_vector is not None and np.linalg.norm(spin_vector) > 1e-2:
            f_magnus = self.props.magnus_coeff * self.props.cross_sectional_area * self.props.air_density * np.cross(spin_vector, velocity)
            a_magnus = f_magnus / self.props.mass
        else:
            a_magnus = np.zeros(3, dtype=np.float64)

        return self.gravity + a_drag + a_swing + a_magnus

    def simulate_delivery(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        dt: float = 0.002,
        max_time: float = 1.8,
        spin_vector: Optional[np.ndarray] = None,
        stop_at_stumps: bool = True
    ) -> List[TrajectoryPoint]:
        """
        Executes forward integration using RK4 from release to stumps.
        Accurately captures the pitch bounce discontinuity.
        """
        r = np.array(r0, dtype=np.float64)
        v = np.array(v0, dtype=np.float64)
        t = 0.0

        trajectory: List[TrajectoryPoint] = []
        bounced = False
        target_y = self.pitch.batsman_stumps_y + 0.3 if stop_at_stumps else 25.0

        while t <= max_time and r[1] <= target_y:
            a = self.compute_acceleration(v, spin_vector)
            phase = "flight_post_bounce" if bounced else "flight_pre_bounce"

            trajectory.append(TrajectoryPoint(
                time=round(t, 5),
                position=r.copy(),
                velocity=v.copy(),
                acceleration=a.copy(),
                phase=phase
            ))

            # RK4 Integration step
            # k1
            k1_r = v
            k1_v = a

            # k2
            v_mid1 = v + 0.5 * dt * k1_v
            a_mid1 = self.compute_acceleration(v_mid1, spin_vector)
            k2_r = v_mid1
            k2_v = a_mid1

            # k3
            v_mid2 = v + 0.5 * dt * k2_v
            a_mid2 = self.compute_acceleration(v_mid2, spin_vector)
            k3_r = v_mid2
            k3_v = a_mid2

            # k4
            v_end = v + dt * k3_v
            a_end = self.compute_acceleration(v_end, spin_vector)
            k4_r = v_end
            k4_v = a_end

            r_next = r + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            v_next = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

            # Ground impact / Bounce detection
            if not bounced and r_next[2] <= self.props.radius and v_next[2] < 0:
                # Interpolate exact bounce contact epoch
                fraction = (self.props.radius - r[2]) / (r_next[2] - r[2] + 1e-9)
                fraction = np.clip(fraction, 0.0, 1.0)
                t_bounce = t + fraction * dt
                r_bounce = r + fraction * (r_next - r)
                r_bounce[2] = self.props.radius

                # Velocity at impact
                v_impact = v + fraction * (v_next - v)

                # Record bounce point
                trajectory.append(TrajectoryPoint(
                    time=round(t_bounce, 5),
                    position=r_bounce.copy(),
                    velocity=v_impact.copy(),
                    acceleration=np.array([0.0, 0.0, 120.0]), # Impact impulse spike
                    phase="bounce"
                ))

                # Apply pitch collision dynamics (using Match-Adaptive Model if available)
                if self.pitch_model is not None:
                    bounce_pred = self.pitch_model.predict_bounce(r_bounce[0], r_bounce[1], v_impact)
                    v_post = bounce_pred.v_out
                else:
                    v_post = v_impact.copy()
                    v_post[2] = -self.props.restitution_z * v_impact[2]  # Normal rebound
                    v_post[1] = self.props.friction_y * v_impact[1]      # Pitch forward drag

                    # Seam cut or spin deflection
                    if abs(self.props.seam_deviation_rad) > 1e-4:
                        # Deflect horizontal angle
                        v_post[0] = v_post[0] + np.tan(self.props.seam_deviation_rad) * v_post[1]

                r = r_bounce
                v = v_post
                t = t_bounce
                bounced = True
                continue

            r = r_next
            v = v_next
            t += dt

        return trajectory

    def project_virtual_path(
        self,
        impact_point: TrajectoryPoint,
        dt: float = 0.002,
        spin_vector: Optional[np.ndarray] = None
    ) -> List[TrajectoryPoint]:
        """
        Projects future ball path from pad/bat impact point through to the stumps plane.
        Used directly by Hawk-Eye LBW adjudication.
        """
        r = impact_point.position.copy()
        v = impact_point.velocity.copy()
        t = impact_point.time

        projected: List[TrajectoryPoint] = []
        target_y = self.pitch.batsman_stumps_y + 0.15

        while r[1] <= target_y and t < impact_point.time + 0.8:
            a = self.compute_acceleration(v, spin_vector)
            projected.append(TrajectoryPoint(
                time=round(t, 5),
                position=r.copy(),
                velocity=v.copy(),
                acceleration=a.copy(),
                phase="projected"
            ))

            # RK4 step
            k1_r = v
            k1_v = a
            v_mid1 = v + 0.5 * dt * k1_v
            a_mid1 = self.compute_acceleration(v_mid1, spin_vector)
            k2_r = v_mid1
            k2_v = a_mid1
            v_mid2 = v + 0.5 * dt * k2_v
            a_mid2 = self.compute_acceleration(v_mid2, spin_vector)
            k3_r = v_mid2
            k3_v = a_mid2
            v_end = v + dt * k3_v
            a_end = self.compute_acceleration(v_end, spin_vector)
            k4_r = v_end
            k4_v = a_end

            r += (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            v += (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            t += dt

            # If ball bounces before stumps during projection
            if r[2] <= self.props.radius and v[2] < 0:
                r[2] = self.props.radius
                if self.pitch_model is not None:
                    v = self.pitch_model.predict_bounce(r[0], r[1], v).v_out
                else:
                    v[2] = -self.props.restitution_z * v[2]
                    v[1] = self.props.friction_y * v[1]

        return projected

    def fit_trajectory(
        self,
        observations: List[Tuple[float, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fits continuous aerodynamic trajectory to noisy 3D observations.
        Returns:
            optimal_r0: Fitted release position [x0, y0, z0]
            optimal_v0: Fitted release velocity [vx0, vy0, vz0]
            fit_residual_rmse: Root Mean Square Error of the fit (meters)
        """
        if len(observations) < 4:
            raise ValueError(f"Need at least 4 observations to fit trajectory, got {len(observations)}")

        times = np.array([o[0] for o in observations])
        obs_pts = np.array([o[1] for o in observations])

        # Initial linear estimate
        dt_total = times[-1] - times[0]
        v_est = (obs_pts[-1] - obs_pts[0]) / max(dt_total, 1e-4)
        v_est[2] += 0.5 * 9.8 * dt_total # Rough gravity correction
        r_est = obs_pts[0] - v_est * times[0]

        param_init = np.concatenate([r_est, v_est, [self.props.swing_force]])

        def loss(p):
            r0 = p[0:3]
            v0 = p[3:6]
            self.props.swing_force = float(p[6])
            sim = self.simulate_delivery(r0, v0, dt=0.005, max_time=float(times[-1]) + 0.05, stop_at_stumps=False)
            sim_times = np.array([pt.time for pt in sim])
            sim_pos = np.array([pt.position for pt in sim])

            err = 0.0
            for t_obs, p_obs in zip(times, obs_pts):
                idx = np.argmin(np.abs(sim_times - t_obs))
                err += np.sum((sim_pos[idx] - p_obs) ** 2)
            return err

        res = minimize(loss, param_init, method="Nelder-Mead", options={"maxiter": 150})
        best_r0 = res.x[0:3]
        best_v0 = res.x[3:6]
        self.props.swing_force = float(res.x[6])

        rmse = np.sqrt(res.fun / len(observations))
        return best_r0, best_v0, rmse
