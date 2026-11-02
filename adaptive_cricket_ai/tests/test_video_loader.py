"""
Unit and Integration Tests for OpenCV Video Loader and Frame Ingestion.
Tests metadata extraction, streaming, color conversions, resizing,
frame extraction, and temporal triplet generation.
"""

from pathlib import Path
import pytest
import numpy as np

from adaptive_cricket_ai.src.data_processing.video_loader import VideoLoader, VideoMetadata

TEST_VIDEO_PATH = Path("adaptive_cricket_ai/datasets/sample_videos/inswinging_yorker.mp4")


@pytest.fixture
def sample_video():
    if not TEST_VIDEO_PATH.exists():
        pytest.skip(f"Test video not found at: {TEST_VIDEO_PATH}")
    return TEST_VIDEO_PATH


def test_video_metadata_extraction(sample_video):
    """Verify video properties are accurately extracted from the container."""
    loader = VideoLoader(sample_video)
    meta = loader.get_metadata()

    assert isinstance(meta, VideoMetadata)
    assert meta.is_valid is True
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.aspect_ratio == "16:9"
    assert meta.fps == 60.0
    assert meta.total_frames == 35
    assert meta.duration_seconds > 0.5
    assert meta.file_size_mb > 0.0


def test_invalid_file_handling():
    """Verify non-existent files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        VideoLoader("non_existent_cricket_video_12345.mp4")


def test_unsupported_extension_handling(tmp_path):
    """Verify non-video file formats raise ValueError."""
    fake_txt = tmp_path / "notes.txt"
    fake_txt.write_text("not a video")
    with pytest.raises(ValueError, match="Unsupported video format"):
        VideoLoader(fake_txt)


def test_frame_streaming_rgb(sample_video):
    """Verify RGB frame streaming with target size and stride."""
    loader = VideoLoader(sample_video)
    frames = list(loader.stream_frames(target_size=(640, 360), color_space="RGB", stride=2))

    assert len(frames) > 0
    first_idx, first_t, first_frame = frames[0]
    assert first_idx == 0
    assert first_t == 0.0
    assert first_frame.shape == (360, 640, 3)
    assert first_frame.dtype == np.uint8


def test_frame_streaming_gray(sample_video):
    """Verify Grayscale frame conversion and shape."""
    loader = VideoLoader(sample_video)
    frames = list(loader.stream_frames(target_size=(320, 180), color_space="GRAY", max_frames=5))

    assert len(frames) == 5
    for idx, t, frame in frames:
        assert frame.shape == (180, 320)
        assert len(frame.shape) == 2


def test_temporal_triplet_generation(sample_video):
    """Verify temporal 3-frame triplets for motion-aware deep learning."""
    loader = VideoLoader(sample_video)
    triplets = list(loader.stream_triplets(target_size=(640, 360)))

    assert len(triplets) == 33  # 35 frames yields 33 triplets
    middle_idx, prev_frame, curr_frame, next_frame = triplets[0]
    assert middle_idx == 1
    assert prev_frame.shape == (360, 640, 3)
    assert curr_frame.shape == (360, 640, 3)
    assert next_frame.shape == (360, 640, 3)


def test_extract_frames_to_disk(sample_video, tmp_path):
    """Verify disk extraction produces valid readable image files."""
    loader = VideoLoader(sample_video)
    out_dir = tmp_path / "extracted"
    extracted_paths = loader.extract_frames(out_dir, sample_stride=5, image_format="jpg")

    assert len(extracted_paths) == 7
    for p in extracted_paths:
        path_obj = Path(p)
        assert path_obj.exists()
        assert path_obj.stat().st_size > 1000  # Non-empty JPEG

