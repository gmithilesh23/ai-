"""
Adaptive Cricket AI - Main Entry Point & Command Line Interface.
Provides CLI commands for:
  - environment-check: Inspect OS, Python, CPU/GPU, and core packages.
  - ingest-video: Inspect metadata and extract sample frames from cricket videos.
  - run-tests: Execute unit and integration tests across the project.
"""

import argparse
import os
import platform
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_environment():
    """Print comprehensive hardware and software diagnostics."""
    print("=" * 70)
    print("SYSTEM & COMPUTATION ENVIRONMENT DIAGNOSTICS")
    print("=" * 70)
    print(f"OS Platform:          {platform.platform()}")
    print(f"Processor:            {platform.processor()}")
    print(f"Logical CPU Cores:    {os.cpu_count()}")
    print(f"Python Executable:    {sys.executable}")
    print(f"Python Version:       {sys.version.split()[0]}")

    # PyTorch & Hardware Acceleration Check
    try:
        import torch
        print(f"PyTorch Version:      {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA/GPU Available:   {cuda_avail}")
        if cuda_avail:
            print(f"GPU Device Count:     {torch.cuda.device_count()}")
            print(f"Active GPU Name:      {torch.cuda.get_device_name(0)}")
        else:
            print("Compute Device:       Intel/AMD Multi-Core CPU (Optimized)")
    except ImportError:
        print("PyTorch:              NOT INSTALLED")

    # OpenCV Check
    try:
        import cv2
        print(f"OpenCV Version:       {cv2.__version__}")
    except ImportError:
        print("OpenCV:               NOT INSTALLED")

    # Scikit-Learn Check
    try:
        import sklearn
        print(f"Scikit-Learn Version: {sklearn.__version__}")
    except ImportError:
        print("Scikit-Learn:         NOT INSTALLED")

    # FilterPy Check
    try:
        import filterpy
        print(f"FilterPy Version:     {filterpy.__version__}")
    except ImportError:
        print("FilterPy:             NOT INSTALLED")

    print("=" * 70)


def ingest_video(video_path_str: str, extract_frames: bool = False, stride: int = 2):
    """Inspect video file and optionally extract frames."""
    from adaptive_cricket_ai.src.data_processing.video_loader import VideoLoader

    video_path = Path(video_path_str)
    if not video_path.is_absolute():
        video_path = WORKSPACE_ROOT / video_path

    print(f"\n[+] Loading video: {video_path}")
    loader = VideoLoader(video_path)
    meta = loader.get_metadata()

    print("\n--- Video Metadata ---")
    print(f"File Name:        {meta.file_name}")
    print(f"File Size:        {meta.file_size_mb} MB")
    print(f"Resolution:       {meta.width} x {meta.height} (Aspect: {meta.aspect_ratio})")
    print(f"Framerate:        {meta.fps} FPS")
    print(f"Total Frames:     {meta.total_frames}")
    print(f"Duration:         {meta.duration_seconds:.2f} s")
    print(f"FourCC Codec:     {meta.fourcc}")

    if extract_frames:
        out_dir = PROJECT_ROOT / "datasets" / "processed" / "frames" / video_path.stem
        print(f"\n[+] Extracting frames to: {out_dir} (stride={stride})...")
        saved = loader.extract_frames(out_dir, sample_stride=stride, image_format="jpg")
        print(f"[OK] Successfully extracted {len(saved)} frames.")


def run_tests():
    """Run pytest suite on the project tests."""
    import subprocess
    cmd = [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v"]
    print(f"[+] Running test suite: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Cricket AI - Next-Gen DRS System")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: env
    subparsers.add_parser("env", help="Inspect runtime environment & compute capabilities")

    # Subcommand: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest video & extract metadata/frames")
    ingest_parser.add_argument("video", type=str, help="Path to video file (.mp4, .avi, .mov)")
    ingest_parser.add_argument("--extract", action="store_true", help="Extract frames to dataset directory")
    ingest_parser.add_argument("--stride", type=int, default=2, help="Frame sampling stride (default: 2)")

    # Subcommand: test
    subparsers.add_parser("test", help="Run project test suite")

    args = parser.parse_args()

    if args.command == "env" or args.command is None:
        check_environment()
    elif args.command == "ingest":
        ingest_video(args.video, extract_frames=args.extract, stride=args.stride)
    elif args.command == "test":
        run_tests()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

