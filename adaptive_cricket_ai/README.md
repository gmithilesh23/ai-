# Match-Adaptive AI Cricket Ball Tracking & DRS System

An enterprise-grade, physics-informed computer vision and machine learning platform for high-speed cricket ball detection, 3D trajectory reconstruction, and match-adaptive pitch bounce prediction for Next-Generation Decision Review Systems (DRS).

---

## Phase 1: Environment & Video Ingestion Baseline

### 1. Project Directory Architecture
```text
adaptive_cricket_ai/
├── datasets/
│   ├── raw/                 # Raw video footage & external annotations
│   ├── processed/           # Extracted frames, parquets, normalized labels
│   ├── sample_videos/       # Curated delivery clips (yorker, bouncer, outswinger, etc.)
│   ├── annotations/         # Ground-truth point & bounding box labels
│   └── synthetic/           # Aerodynamic RK4 simulated benchmark deliveries
├── models/
│   ├── checkpoints/         # Trained model checkpoints
│   └── pretrained/          # Pretrained vision weights
├── src/
│   ├── data_processing/     # OpenCV video loader, frame streaming, batching
│   ├── detection/           # Ball detection models (YOLO & temporal CNNs)
│   ├── tracking/            # 9D Ballistic Kalman filter & RTS smoothing
│   ├── reconstruction/      # Camera calibration, homography, 3D ray back-projection
│   ├── physics/             # Aerodynamic RK4 simulator (gravity, drag, Magnus force)
│   ├── prediction/          # Physics residual bounce prediction models
│   ├── adaptive/            # PIMA-Pitch Online Bayesian spatial learning engine
│   └── evaluation/          # Tracking, detection, and DRS adjudication metrics
├── dashboard/               # Streamlit interactive analytics & 3D visualization
├── api/                     # FastAPI backend endpoints
├── notebooks/               # Research experiments & exploratory analysis
├── tests/                   # Pytest automated test suite
├── configs/                 # YAML geometry, physics, and tracker parameters
├── reports/                 # Research guides & experimental benchmark reports
├── main.py                  # Unified CLI entrypoint
└── requirements.txt         # Pinned production dependencies
```

---

## 2. Ingestion & Video Loader Capabilities

The `adaptive_cricket_ai.src.data_processing.video_loader.VideoLoader` module provides:
- **Comprehensive Metadata Extraction**: Resolution, aspect ratio, frame rate (FPS), total frame count, duration, FourCC codec, and file size.
- **Memory-Efficient Frame Streaming**: Generator-based sequential access without loading massive videos into memory.
- **Color Space Transformation**: Supports BGR, RGB, and Grayscale normalization.
- **Spatial Resizing**: Bilinear downsampling to arbitrary target sizes (e.g., 640x640 or 640x360).
- **Temporal Triplet Windowing**: Generates $(t-1, t, t+1)$ frame stacks for motion-aware ball detection networks.
- **Frame Disk Extraction**: Extracts and writes high-quality sampled image frames with zero-padded filenames.

---

## 3. Quick Start & CLI Usage

### System & Environment Diagnostics
```bash
python main.py env
```

### Video Ingestion & Metadata Inspection
```bash
python main.py ingest adaptive_cricket_ai/datasets/sample_videos/inswinging_yorker.mp4
```

### Frame Extraction for Dataset Annotation
```bash
python main.py ingest adaptive_cricket_ai/datasets/sample_videos/inswinging_yorker.mp4 --extract --stride 3
```

### Execute Test Suite
```bash
pytest adaptive_cricket_ai/tests/test_video_loader.py -v
```

