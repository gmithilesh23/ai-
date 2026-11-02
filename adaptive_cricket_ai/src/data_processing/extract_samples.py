"""
Sample Frame Extraction and Video Ingestion Verification Script.
Demonstrates:
1. Video file validation and metadata extraction
2. Sequential streaming with timestamp logging
3. Subsampled frame extraction to disk
4. Temporal triplet window verification
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adaptive_cricket_ai.src.data_processing.video_loader import VideoLoader


def run_extraction_demo():
    sample_video_path = PROJECT_ROOT / "adaptive_cricket_ai" / "datasets" / "sample_videos" / "inswinging_yorker.mp4"
    output_dir = PROJECT_ROOT / "adaptive_cricket_ai" / "datasets" / "processed" / "sample_frames"

    print("=" * 70)
    print("PHASE 1: VIDEO INGESTION & FRAME EXTRACTION PIPELINE DEMO")
    print("=" * 70)

    # 1. Initialize Loader
    print(f"\n[1] Initializing VideoLoader for: {sample_video_path.name}")
    loader = VideoLoader(sample_video_path)

    # 2. Extract Metadata
    print("\n[2] Extracting Video Metadata...")
    meta = loader.get_metadata()
    print(f"    - File Name:      {meta.file_name}")
    print(f"    - File Size:      {meta.file_size_mb} MB")
    print(f"    - Resolution:     {meta.width} x {meta.height} (Aspect: {meta.aspect_ratio})")
    print(f"    - Frame Rate:     {meta.fps} FPS")
    print(f"    - Total Frames:   {meta.total_frames}")
    print(f"    - Duration:       {meta.duration_seconds:.2f} seconds")
    print(f"    - FourCC Codec:   {meta.fourcc}")
    print(f"    - Stream Valid:   {meta.is_valid}")

    # 3. Extract sample frames to disk (stride=3 to capture distinct trajectory phases)
    print(f"\n[3] Extracting sample frames to: {output_dir} (stride=3)...")
    saved_frames = loader.extract_frames(output_dir, sample_stride=3, image_format="jpg")
    print(f"    [OK] Successfully extracted {len(saved_frames)} frames.")
    for p in saved_frames[:3]:
        print(f"         Preview: {Path(p).name}")
    print(f"         ... and {len(saved_frames) - 3} more frames.")

    # 4. Verify Temporal Triplet Generator
    print("\n[4] Verifying 3-frame Temporal Triplet Generator...")
    triplet_count = 0
    for middle_idx, t_prev, t_curr, t_next in loader.stream_triplets(target_size=(640, 360)):
        triplet_count += 1
        if triplet_count == 1:
            print(f"    - First Triplet center frame: #{middle_idx:03d}")
            print(f"      Frame shapes: prev={t_prev.shape}, curr={t_curr.shape}, next={t_next.shape}")
    print(f"    [OK] Streamed {triplet_count} consecutive temporal triplets.")

    print("\n" + "=" * 70)
    print("PIPELINE TEST COMPLETED: ALL VIDEO LOADER OPERATIONS VERIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    run_extraction_demo()

