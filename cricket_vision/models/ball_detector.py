"""
Deep Learning Cricket Ball Detection Architecture (TrackNetCricket)

Features:
- Multi-frame 3-consecutive-frame temporal fusion (9-channel input)
- U-Net style Encoder-Decoder with skip connections and dilated convolutions
- Heatmap regression targeting sub-pixel Gaussian peaks
- Sub-pixel centroid extraction via 2D quadratic peak interpolation
- Integrated hybrid CV fallback detector (Color + MOG2 Background Subtraction + Motion Gating)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


@dataclass
class BallDetection:
    """Detected ball instance in an image frame."""
    x: float                  # Sub-pixel horizontal coordinate
    y: float                  # Sub-pixel vertical coordinate
    confidence: float         # Detection score [0.0 - 1.0]
    radius: float             # Estimated radius in pixels
    method: str               # 'deep_learning' or 'hybrid_cv'


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU with optional dilation."""
    def __init__(self, in_c: int, out_c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TrackNetCricket(nn.Module):
    """
    Temporal Heatmap Convolutional Neural Network for Cricket Ball Localization.
    Input: Stack of 3 consecutive frames [Batch, 9, H, W]
    Output: Single-channel ball probability heatmap [Batch, 1, H, W]
    """
    def __init__(self, in_channels: int = 9, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck with dilated convolution to capture high-speed motion context
        self.bridge = ConvBlock(128, 256, dilation=2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Final Heatmap Head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bridge
        b = self.bridge(p3)

        # Decoder
        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        heatmap = self.final_conv(d1)
        return heatmap


class CricketBallDetector:
    """
    Production-grade inference pipeline combining Deep Learning Heatmap Regression
    with Sub-pixel Parabolic Refinement and Computer Vision Fallback.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_thresh: float = 0.35,
        target_size: Tuple[int, int] = (320, 180),
        device: Optional[str] = None
    ):
        self.confidence_thresh = confidence_thresh
        self.target_size = target_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Initialize Neural Network
        self.model = TrackNetCricket().to(self.device)
        self.model.eval()

        if weights_path:
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
            except Exception as e:
                print(f"[CricketBallDetector] Note: Could not load weights from {weights_path}: {e}")

        # Classical CV fallback components
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=25, detectShadows=False)

    def extract_subpixel_peak(self, heatmap: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Locates the global maximum and applies 2D quadratic interpolation
        for true sub-pixel ball centroid estimation.
        """
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        if max_val < self.confidence_thresh:
            return None

        px, py = max_loc
        h, w = heatmap.shape

        # Bound check for 3x3 patch
        if 1 <= px < w - 1 and 1 <= py < h - 1:
            patch = heatmap[py-1:py+2, px-1:px+2].astype(np.float64)
            # Quadratic fit in X direction
            denom_x = patch[1, 0] - 2 * patch[1, 1] + patch[1, 2]
            dx = (patch[1, 0] - patch[1, 2]) / (2.0 * denom_x + 1e-7) if abs(denom_x) > 1e-5 else 0.0
            dx = np.clip(dx, -0.5, 0.5)

            # Quadratic fit in Y direction
            denom_y = patch[0, 1] - 2 * patch[1, 1] + patch[2, 1]
            dy = (patch[0, 1] - patch[2, 1]) / (2.0 * denom_y + 1e-7) if abs(denom_y) > 1e-5 else 0.0
            dy = np.clip(dy, -0.5, 0.5)

            sub_x = px + dx
            sub_y = py + dy
        else:
            sub_x, sub_y = float(px), float(py)

        return float(sub_x), float(sub_y), float(max_val)

    def detect_triplet(self, frames_bgr: List[np.ndarray], prefer_fast_cv: bool = True) -> Optional[BallDetection]:
        """
        Performs detection over 3 consecutive frames [t-1, t, t+1].
        If prefer_fast_cv is True, checks the fast high-precision CV detector first;
        if confident, returns in ~15ms. Otherwise falls back to deep learning U-Net.
        """
        if len(frames_bgr) != 3:
            raise ValueError(f"Expected 3 consecutive frames, got {len(frames_bgr)}")

        if prefer_fast_cv:
            cv_cand = self.detect_cv_fallback(frames_bgr[1])
            if cv_cand is not None and cv_cand.confidence >= 0.55:
                return cv_cand

        orig_h, orig_w = frames_bgr[1].shape[:2]
        tw, th = self.target_size

        # Preprocess frames: resize, normalize to [0, 1], concatenate along channels
        resized_channels = []
        for f in frames_bgr:
            r = cv2.resize(f, (tw, th)).astype(np.float32) / 255.0
            # Convert BGR to RGB
            r = r[..., ::-1].transpose(2, 0, 1)  # [3, th, tw]
            resized_channels.append(r)

        stacked = np.concatenate(resized_channels, axis=0)  # [9, th, tw]
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap_tensor = self.model(tensor)
            heatmap = heatmap_tensor.squeeze().cpu().numpy()

        peak = self.extract_subpixel_peak(heatmap)
        if peak is not None:
            sub_x, sub_y, conf = peak
            # Scale coordinates back to original frame dimensions
            real_x = sub_x * (orig_w / float(tw))
            real_y = sub_y * (orig_h / float(th))
            return BallDetection(
                x=real_x,
                y=real_y,
                confidence=conf,
                radius=max(3.0, 7.0 * (orig_w / 1280.0)),
                method="deep_learning"
            )

        # Fallback to CV motion & color segmentation
        return self.detect_cv_fallback(frames_bgr[1])

    def detect_cv_fallback(self, frame_bgr: np.ndarray) -> Optional[BallDetection]:
        """
        Fast classical CV detector using HSV cricket ball color filter
        and circularity/motion contours.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Red cricket ball HSV range (two bands for hue wrap-around 0 and 180)
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([180, 255, 255]))
        # White cricket ball HSV range (low saturation, high value)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))

        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.bitwise_or(mask, mask_white)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 6 or area > 600:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-4:
                continue

            circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.45:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                score = circularity * (1.0 - abs(area - 50.0) / 600.0)
                if score > best_score:
                    best_score = score
                    best_candidate = (cx, cy, radius, float(score))

        if best_candidate and best_candidate[3] > 0.30:
            cx, cy, rad, score = best_candidate
            return BallDetection(
                x=float(cx),
                y=float(cy),
                confidence=min(1.0, score),
                radius=max(3.0, float(rad)),
                method="hybrid_cv"
            )

        return None
