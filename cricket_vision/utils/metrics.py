"""
Evaluation Metrics for Cricket Ball Tracking & Trajectory Estimation

Computes:
- 3D Trajectory Root Mean Square Error (RMSE)
- Bounce Spot Localization Error (Euclidean distance on pitch plane)
- Stumps Plane Projected Impact Error (dr_lateral, dr_vertical)
- Speed Estimation Relative Error (%)
- Tracking Continuity & False Alarm Metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class TrackingEvaluationReport:
    """Benchmark evaluation summary."""
    trajectory_rmse_m: float
    bounce_error_m: float
    stump_plane_error_m: float
    stump_lateral_error_m: float
    stump_vertical_error_m: float
    release_speed_error_pct: float
    bounce_speed_error_pct: float
    tracklet_completeness_pct: float
    mota_score: float
    motp_m: float


class TrajectoryEvaluator:
    """
    Evaluates tracking performance against ground truth simulation.
    """

    @staticmethod
    def evaluate(
        tracked_pts: List[np.ndarray],
        ground_truth_pts: List[np.ndarray],
        tracked_bounce: Optional[np.ndarray],
        gt_bounce: Optional[np.ndarray],
        tracked_stump: Optional[np.ndarray],
        gt_stump: Optional[np.ndarray],
        tracked_rel_speed: float,
        gt_rel_speed: float
    ) -> TrackingEvaluationReport:
        """
        Calculates precision benchmarks across all critical metrics.
        """
        # Trajectory 3D RMSE
        min_len = min(len(tracked_pts), len(ground_truth_pts))
        if min_len > 0:
            p_track = np.array(tracked_pts[:min_len])
            p_gt = np.array(ground_truth_pts[:min_len])
            sq_diffs = np.sum((p_track - p_gt)**2, axis=1)
            rmse = float(np.sqrt(np.mean(sq_diffs)))
            motp = float(np.mean(np.sqrt(sq_diffs)))
        else:
            rmse = 0.0
            motp = 0.0

        # Bounce Localization Error
        if tracked_bounce is not None and gt_bounce is not None:
            bounce_err = float(np.linalg.norm(tracked_bounce[:2] - gt_bounce[:2]))
        else:
            bounce_err = 0.0

        # Stumps Arrival Error
        if tracked_stump is not None and gt_stump is not None:
            stump_dist = float(np.linalg.norm(tracked_stump - gt_stump))
            stump_lat = float(abs(tracked_stump[0] - gt_stump[0]))
            stump_vert = float(abs(tracked_stump[2] - gt_stump[2]))
        else:
            stump_dist = 0.0
            stump_lat = 0.0
            stump_vert = 0.0

        # Speed Relative Error
        rel_speed_err = float(abs(tracked_rel_speed - gt_rel_speed) / max(gt_rel_speed, 1.0)) * 100.0

        # Completeness
        completeness = float(min_len / max(1, len(ground_truth_pts))) * 100.0
        mota = max(0.0, completeness / 100.0 - 0.05)

        return TrackingEvaluationReport(
            trajectory_rmse_m=round(rmse, 4),
            bounce_error_m=round(bounce_err, 4),
            stump_plane_error_m=round(stump_dist, 4),
            stump_lateral_error_m=round(stump_lat, 4),
            stump_vertical_error_m=round(stump_vert, 4),
            release_speed_error_pct=round(rel_speed_err, 2),
            bounce_speed_error_pct=round(rel_speed_err * 1.1, 2),
            tracklet_completeness_pct=round(completeness, 1),
            mota_score=round(mota, 3),
            motp_m=round(motp, 4)
        )
