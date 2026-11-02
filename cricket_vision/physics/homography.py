"""
Camera Geometry, Planar Homography & Monocular 3D Reconstruction Engine

Implements:
- Standard Pinhole Camera Model (K, R, t)
- Planar Homography between video pixel space (u, v) and pitch surface (X, Y, 0)
- Automatic calibration from pitch markings and crease keypoints
- Physics-constrained 3D ray projection for monocular elevation reconstruction
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

from cricket_vision.physics.pitch_geometry import CricketPitch, PitchDimensions


class CameraModel:
    """
    Calibrated pinhole camera representing broadcast / tracking camera viewpoints.
    """

    def __init__(
        self,
        camera_pos: np.ndarray = np.array([0.0, -4.5, 3.8]),
        target_pos: np.ndarray = np.array([0.0, 14.0, 0.5]),
        fov_vertical_deg: float = 32.0,
        resolution: Tuple[int, int] = (1280, 720)
    ):
        self.camera_pos = np.array(camera_pos, dtype=np.float64)
        self.target_pos = np.array(target_pos, dtype=np.float64)
        self.resolution = resolution
        self.width, self.height = resolution
        self.fov_vertical_deg = fov_vertical_deg

        # Compute camera intrinsic matrix K
        fov_rad = np.radians(fov_vertical_deg)
        fy = (self.height / 2.0) / np.tan(fov_rad / 2.0)
        fx = fy  # Square pixels
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Compute look-at rotation matrix R and translation t
        forward = self.target_pos - self.camera_pos
        forward /= np.linalg.norm(forward)

        # World up is [0, 0, 1]
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        # In camera coordinates: X right, Y down, Z forward
        # R rotates world vectors into camera space
        R_world_to_cam = np.array([
            right,
            -up,       # Y down
            forward
        ], dtype=np.float64)

        self.R = R_world_to_cam
        self.t = -self.R @ self.camera_pos
        self.P = self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

        # Compute planar homography for Z=0 (pitch ground surface)
        # H maps [X, Y, 1]^T on pitch ground to [u, v, 1]^T
        H_ground = self.P[:, [0, 1, 3]]
        self.H = H_ground / H_ground[2, 2]
        self.H_inv = np.linalg.inv(self.H)

    def world_to_pixel(self, pt3d: np.ndarray) -> Tuple[int, int, bool]:
        """
        Projects 3D world coordinate [X, Y, Z] to image coordinates [u, v].
        Returns (u, v, is_in_front_of_camera).
        """
        p_homo = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0], dtype=np.float64)
        p_cam = self.R @ pt3d + self.t

        if p_cam[2] <= 0.1:  # Behind camera or at camera plane
            return -1, -1, False

        uv_w = self.P @ p_homo
        u = uv_w[0] / uv_w[2]
        v = uv_w[1] / uv_w[2]

        return int(round(u)), int(round(v)), True

    def pixel_to_ground(self, u: float, v: float) -> np.ndarray:
        """
        Projects image pixel (u, v) onto ground pitch plane (Z=0).
        Returns [X, Y, 0.0].
        """
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        xyw = self.H_inv @ uv1
        x = xyw[0] / xyw[2]
        y = xyw[1] / xyw[2]
        return np.array([x, y, 0.0], dtype=np.float64)

    def get_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the 3D unit optical ray starting at camera position:
        Ray(s) = camera_pos + s * ray_dir
        """
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        K_inv = np.linalg.inv(self.K)
        cam_ray = K_inv @ uv1
        cam_ray /= np.linalg.norm(cam_ray)

        world_ray = self.R.T @ cam_ray
        world_ray /= np.linalg.norm(world_ray)
        return self.camera_pos.copy(), world_ray

    def reconstruct_3d_point(
        self,
        u: float,
        v: float,
        height_prior: Optional[float] = None,
        depth_y_prior: Optional[float] = None
    ) -> np.ndarray:
        """
        Resolves monocular depth ambiguity along the viewing ray using:
        1. Height prior Z (e.g. from aerodynamic flight model or bounce constraint), or
        2. Depth prior Y (e.g. from down-the-pitch forward velocity kinematics).
        """
        origin, ray = self.get_ray(u, v)

        if height_prior is not None:
            # Ray: Z = origin[2] + s * ray[2] => s = (height_prior - origin[2]) / ray[2]
            if abs(ray[2]) > 1e-5:
                s = (height_prior - origin[2]) / ray[2]
                return origin + s * ray

        if depth_y_prior is not None:
            # Ray: Y = origin[1] + s * ray[1] => s = (depth_y_prior - origin[1]) / ray[1]
            if abs(ray[1]) > 1e-5:
                s = (depth_y_prior - origin[1]) / ray[1]
                return origin + s * ray

        # Default fallback to ground plane
        return self.pixel_to_ground(u, v)

    @classmethod
    def from_crease_correspondences(
        cls,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float, float]],
        resolution: Tuple[int, int] = (1280, 720)
    ) -> "CameraModel":
        """
        Calibrates camera perspective matrix directly from 4+ pitch marking correspondences
        using cv2.findHomography / cv2.solvePnP.
        """
        pts_img = np.array(image_points, dtype=np.float32)
        pts_world = np.array(world_points, dtype=np.float32)

        # Approximate initial intrinsics
        w, h = resolution
        fx = fy = max(w, h)
        cx, cy = w / 2.0, h / 2.0
        K_init = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(pts_world, pts_img, K_init, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return cls(resolution=resolution)

        R, _ = cv2.Rodrigues(rvec)
        cam_pos = -R.T @ tvec.flatten()

        model = cls(resolution=resolution)
        model.K = K_init
        model.R = R
        model.t = tvec.flatten()
        model.camera_pos = cam_pos
        model.P = model.K @ np.hstack([model.R, model.t.reshape(3, 1)])
        H_g = model.P[:, [0, 1, 3]]
        model.H = H_g / H_g[2, 2]
        model.H_inv = np.linalg.inv(model.H)
        return model
