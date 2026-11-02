"""
Robust OpenCV Video Loader and Frame Processing Pipeline.
Designed for cricket match delivery ingestion, metadata extraction,
frame streaming, batching, and temporal windowing for ball detection.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    """Detailed metadata container for an ingested video clip."""
    file_path: str
    file_name: str
    file_size_mb: float
    width: int
    height: int
    aspect_ratio: str
    fps: float
    total_frames: int
    duration_seconds: float
    fourcc: str
    is_valid: bool


class VideoLoader:
    """
    Production-grade OpenCV Video Ingestion & Processing Pipeline.
    
    Supports MP4, AVI, MOV containers with memory-efficient frame generation,
    timestamp seeking, sub-sampling, frame extraction, and temporal triplet batching.
    """
    
    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self._validate_file()
        self._metadata: Optional[VideoMetadata] = None

    def _validate_file(self) -> None:
        """Ensure file exists and has a supported video extension."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")
        
        ext = self.video_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported video format '{ext}'. Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

    def get_metadata(self) -> VideoMetadata:
        """
        Inspect video stream and extract comprehensive spatial and temporal metadata.
        """
        if self._metadata is not None:
            return self._metadata

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {self.video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
            
            # Safe fallbacks if metadata tags are corrupted/missing
            if fps <= 0:
                fps = 30.0
            duration = total_frames / fps if fps > 0 else 0.0

            file_size_mb = self.video_path.stat().st_size / (1024 * 1024)

            # Aspect ratio calculation
            gcd_val = np.gcd(width, height) if width > 0 and height > 0 else 1
            aspect_ratio = f"{width // gcd_val}:{height // gcd_val}"

            self._metadata = VideoMetadata(
                file_path=str(self.video_path.resolve()),
                file_name=self.video_path.name,
                file_size_mb=round(file_size_mb, 2),
                width=width,
                height=height,
                aspect_ratio=aspect_ratio,
                fps=round(fps, 2),
                total_frames=total_frames,
                duration_seconds=round(duration, 3),
                fourcc=fourcc.strip(),
                is_valid=(width > 0 and height > 0 and total_frames > 0)
            )
            return self._metadata
        finally:
            cap.release()

    def stream_frames(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        color_space: str = "RGB",
        stride: int = 1,
        start_frame: int = 0,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Stream frames sequentially as a memory-efficient generator.
        
        Args:
            target_size: (width, height) resize tuple if downsampling requested.
            color_space: Output color mode ('RGB', 'BGR', 'GRAY').
            stride: Step size (1 = every frame, 2 = every second frame).
            start_frame: Initial frame offset to seek to.
            max_frames: Maximum number of frames to yield.
            
        Yields:
            Tuple of (frame_index, timestamp_seconds, frame_array)
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        meta = self.get_metadata()
        fps = meta.fps if meta.fps > 0 else 30.0

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        curr_frame_idx = start_frame
        yielded_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if (curr_frame_idx - start_frame) % stride == 0:
                    # Apply color conversion
                    if color_space.upper() == "RGB":
                        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif color_space.upper() == "GRAY":
                        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        processed_frame = frame

                    # Apply resize if specified
                    if target_size is not None:
                        processed_frame = cv2.resize(
                            processed_frame, target_size, interpolation=cv2.INTER_LINEAR
                        )

                    timestamp = curr_frame_idx / fps
                    yield curr_frame_idx, timestamp, processed_frame
                    yielded_count += 1

                    if max_frames is not None and yielded_count >= max_frames:
                        break

                curr_frame_idx += 1
        finally:
            cap.release()

    def stream_triplets(
        self,
        target_size: Tuple[int, int] = (640, 360),
        color_space: str = "RGB"
    ) -> Generator[Tuple[int, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Yields temporal 3-frame triplets (t-1, t, t+1) for motion-based ball detection.
        Replicates border frames at t=0 and t=T-1 to preserve frame sequence length.
        """
        frame_buffer: List[Tuple[int, np.ndarray]] = []
        
        for idx, _, frame in self.stream_frames(target_size=target_size, color_space=color_space, stride=1):
            frame_buffer.append((idx, frame))
            if len(frame_buffer) == 3:
                yield (
                    frame_buffer[1][0],  # Middle frame index
                    frame_buffer[0][1],  # t-1
                    frame_buffer[1][1],  # t
                    frame_buffer[2][1]   # t+1
                )
                frame_buffer.pop(0)

    def extract_frames(
        self,
        output_dir: Union[str, Path],
        sample_stride: int = 1,
        max_frames: Optional[int] = None,
        image_format: str = "png"
    ) -> List[str]:
        """
        Extract video frames and save them to disk for dataset annotation or visual inspection.
        
        Returns:
            List of saved image absolute file paths.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        fmt = image_format.lower().lstrip(".")

        for idx, timestamp, frame in self.stream_frames(
            color_space="BGR",  # cv2.imwrite expects BGR
            stride=sample_stride,
            max_frames=max_frames
        ):
            file_name = f"frame_{idx:06d}.{fmt}"
            save_file = out_path / file_name
            success = cv2.imwrite(str(save_file), frame)
            if success:
                saved_files.append(str(save_file.resolve()))

        return saved_files

