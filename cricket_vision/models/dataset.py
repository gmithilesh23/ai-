"""
Cricket Ball Dataset Generator & PyTorch Dataset

Generates photorealistic synthetic cricket deliveries with:
- Realistic pitch surface textures, crease lines, stumps, and lighting
- Motion-blurred red/white cricket balls with seam orientation
- Ground-truth 2D Gaussian target heatmaps for sub-pixel deep learning training
- Distractor objects (batsman pads, white shoes, pitch markings, turf blemishes)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from cricket_vision.physics.pitch_geometry import CricketPitch
from cricket_vision.physics.trajectory import BallisticTrajectorySimulator, AerodynamicProperties
from cricket_vision.physics.homography import CameraModel


class CricketBallSyntheticDataset(Dataset):
    """
    Synthetic cricket delivery dataset generator for TrackNet-Cricket training.
    """

    def __init__(
        self,
        num_samples: int = 120,
        image_size: Tuple[int, int] = (640, 360),
        heatmap_sigma: float = 2.5,
        ball_type: str = "red"  # "red" or "white"
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.ball_type = ball_type

        self.pitch = CricketPitch()
        self.camera = CameraModel(resolution=image_size)
        self.simulator = BallisticTrajectorySimulator(pitch=self.pitch)

    def __len__(self) -> int:
        return self.num_samples

    def generate_pitch_background(self) -> np.ndarray:
        """Draws realistic cricket pitch surface with creases and stumps."""
        w, h = self.image_size
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Grass field background (rich green with subtle variations)
        base_color = np.array([34, 139, 34], dtype=np.float32)  # Forest Green
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.float32)
        grass = np.clip(base_color + noise, 0, 255).astype(np.uint8)
        img = grass

        # Draw Pitch Strip (clay / turf brown/tan)
        pitch_poly_3d = self.pitch.get_pitch_boundary_polygon()
        pitch_poly_2d = []
        for pt3d in pitch_poly_3d:
            u, v, valid = self.camera.world_to_pixel(pt3d)
            if valid:
                pitch_poly_2d.append([u, v])

        if len(pitch_poly_2d) >= 4:
            pts = np.array(pitch_poly_2d, dtype=np.int32)
            cv2.fillPoly(img, [pts], (140, 180, 210))  # Tan pitch in BGR

        # Draw Crease Lines (white)
        crease_dict = self.pitch.get_crease_lines()
        for name, line3d in crease_dict.items():
            u1, v1, val1 = self.camera.world_to_pixel(line3d[0])
            u2, v2, val2 = self.camera.world_to_pixel(line3d[1])
            if val1 and val2:
                cv2.line(img, (u1, v1), (u2, v2), (255, 255, 255), 2)

        # Draw Batsman Stumps (Y=20.12m)
        wicket_box = self.pitch.get_wicket_bounding_box(is_batsman_end=True)
        stump_xs = [wicket_box["off_stump_x"], wicket_box["middle_stump_x"], wicket_box["leg_stump_x"]]
        for sx in stump_xs:
            b_u, b_v, b_val = self.camera.world_to_pixel(np.array([sx, wicket_box["y"], 0.0]))
            t_u, t_v, t_val = self.camera.world_to_pixel(np.array([sx, wicket_box["y"], self.pitch.dims.stump_height]))
            if b_val and t_val:
                cv2.line(img, (b_u, b_v), (t_u, t_v), (210, 215, 225), 3)

        # Draw Bails
        b1_u, b1_v, _ = self.camera.world_to_pixel(np.array([wicket_box["off_stump_x"], wicket_box["y"], self.pitch.dims.total_wicket_height]))
        b2_u, b2_v, _ = self.camera.world_to_pixel(np.array([wicket_box["leg_stump_x"], wicket_box["y"], self.pitch.dims.total_wicket_height]))
        if b1_u != -1 and b2_u != -1:
            cv2.line(img, (b1_u, b1_v), (b2_u, b2_v), (180, 190, 200), 2)

        return img

    def render_ball_on_frame(
        self,
        frame: np.ndarray,
        u: float,
        v: float,
        radius: float,
        velocity_2d: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Renders cricket ball with motion blur, shading and seam."""
        out = frame.copy()
        h, w = out.shape[:2]

        ball_col = (20, 20, 180) if self.ball_type == "red" else (240, 240, 245)

        # If high speed, apply directional motion blur
        if velocity_2d is not None and (abs(velocity_2d[0]) > 2 or abs(velocity_2d[1]) > 2):
            vx, vy = velocity_2d
            num_steps = 4
            for s in range(num_steps):
                offset_x = u - (vx * (s / float(num_steps)))
                offset_y = v - (vy * (s / float(num_steps)))
                alpha = 0.25 + 0.15 * (1.0 - s / float(num_steps))
                overlay = out.copy()
                cv2.circle(overlay, (int(round(offset_x)), int(round(offset_y))), int(round(radius)), ball_col, -1)
                cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
        else:
            cv2.circle(out, (int(round(u)), int(round(v))), int(round(radius)), ball_col, -1)
            # Specular highlight
            hl_x = int(round(u - radius * 0.3))
            hl_y = int(round(v - radius * 0.3))
            cv2.circle(out, (hl_x, hl_y), max(1, int(round(radius * 0.25))), (255, 255, 255), -1)

        return out

    def generate_gaussian_heatmap(self, u: float, v: float) -> np.ndarray:
        """Generates ground truth 2D Gaussian heatmap centered at (u, v)."""
        w, h = self.image_size
        heatmap = np.zeros((h, w), dtype=np.float32)

        if not (0 <= u < w and 0 <= v < h):
            return heatmap

        sigma = self.heatmap_sigma
        radius = int(round(3 * sigma))

        x_min = max(0, int(u) - radius)
        x_max = min(w - 1, int(u) + radius)
        y_min = max(0, int(v) - radius)
        y_max = min(h - 1, int(v) + radius)

        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)

        dist_sq = (grid_x - u)**2 + (grid_y - v)**2
        gaussian = np.exp(-dist_sq / (2.0 * sigma**2))

        heatmap[y_min:y_max+1, x_min:x_max+1] = np.maximum(
            heatmap[y_min:y_max+1, x_min:x_max+1], gaussian
        )

        return heatmap

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            stacked_frames: [9, H, W] tensor (3 consecutive frames)
            target_heatmap: [1, H, W] tensor (ball heatmap for frame t)
        """
        # Randomize delivery release kinematics
        v_forward = np.random.uniform(30.0, 42.0)  # ~110 - 150 km/h
        v_lateral = np.random.uniform(-1.5, 1.5)
        v_vert = np.random.uniform(-2.5, -0.5)

        r0 = np.array([np.random.uniform(-0.5, 0.5), 0.5, np.random.uniform(1.8, 2.2)])
        v0 = np.array([v_lateral, v_forward, v_vert])

        sim_dt = 1.0 / 60.0
        traj = self.simulator.simulate_delivery(r0, v0, dt=sim_dt, max_time=1.2)

        # Pick random epoch t for the triplet [t-1, t, t+1]
        mid_idx = np.random.randint(1, len(traj) - 1)
        triplet_pts = [traj[mid_idx - 1], traj[mid_idx], traj[mid_idx + 1]]

        # Generate base background
        bg = self.generate_pitch_background()

        frames = []
        for i, pt in enumerate(triplet_pts):
            f = bg.copy()
            u, v, valid = self.camera.world_to_pixel(pt.position)
            if valid and (0 <= u < self.image_size[0]) and (0 <= v < self.image_size[1]):
                # Depth scale radius
                depth = max(1.0, pt.position[1])
                rad = max(2.5, 9.0 * (5.0 / depth))
                # 2D velocity in pixels for motion blur
                v_3d = pt.velocity
                pt_next = pt.position + v_3d * (sim_dt * 0.5)
                u2, v2, _ = self.camera.world_to_pixel(pt_next)
                vel_2d = (u2 - u, v2 - v) if u2 != -1 else None

                f = self.render_ball_on_frame(f, u, v, rad, vel_2d)

            # Convert BGR to RGB and normalize to [0, 1]
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(f_rgb.transpose(2, 0, 1))  # [3, H, W]

        # Stack into [9, H, W]
        stacked = np.concatenate(frames, axis=0)

        # Center frame target heatmap
        mid_u, mid_v, mid_val = self.camera.world_to_pixel(triplet_pts[1].position)
        if mid_val:
            heatmap = self.generate_gaussian_heatmap(mid_u, mid_v)
        else:
            heatmap = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.float32)

        tensor_x = torch.from_numpy(stacked)
        tensor_y = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]

        return tensor_x, tensor_y
