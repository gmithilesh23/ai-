# Cricket Ball Tracking & DRS Dataset Preparation & Annotation Guide

This guide establishes the data engineering standards, dataset acquisition workflows, annotation protocols, and delivery-level schemas for the **Match-Adaptive AI Cricket Ball Tracking and DRS Trajectory Prediction System**.

---

## 1. Data Architecture Overview

The system operates across two interconnected data representations:

1. **Per-Frame Computer Vision Data (Low-Level)**:
   - High-framerate (50/60/120+ fps) or broadcast (25/30 fps) video streams.
   - Bounding boxes ($[x_c, y_c, w, h]$ normalized to $[0, 1]$) or sub-pixel centroid coordinates $(u, v)$ with visibility flags.
   - Temporal triplet frames $(t-1, t, t+1)$ for motion-aware ball detection networks.

2. **Per-Delivery Kinematic & Pitch Data (High-Level)**:
   - Trajectory sequences in 3D pitch coordinates $(X, Y, Z)$ where $Y \in [0, 20.12]\,\text{m}$ (stump to stump).
   - Pre-impact velocity vector $\mathbf{v}_{\text{in}} = (v_{x,\text{in}}, v_{y,\text{in}}, v_{z,\text{in}})$.
   - Pitch impact location $(X_p, Y_p)$ and post-bounce velocity $\mathbf{v}_{\text{out}}$.
   - Match adaptation state: pitch restitution coefficient $e_z(X, Y)$, seam deflection angle, and observation uncertainty.

---

## 2. Public Datasets & Data Acquisition

### 2.1 Recommended Open-Source Datasets
| Dataset Name | Source | Description | Format |
| :--- | :--- | :--- | :--- |
| **TrackNet Cricket Ball Dataset** | Academic Sports Vision | Video frames of cricket bowling deliveries with sub-pixel ball center annotations. | CSV $(x, y, \text{visibility})$ |
| **Roboflow Cricket Ball Detection** | Roboflow Universe | Multi-match broadcast crops annotated with cricket ball bounding boxes. | YOLOv8 / Darknet |
| **CricData Trajectory Benchmark** | Research Repositories | Calibrated multi-camera tracking sequences with ground-truth 3D positions. | JSON / HDF5 |
| **PIMA Synthetic Physics Benchmark** | Built-in Generator | Deterministic 4th-order Runge-Kutta aerodynamic simulations across 6 delivery archetypes. | CSV / MP4 |

### 2.2 Directory Ingestion Layout
Place downloaded and extracted datasets in the following structure:
```text
adaptive_cricket_ai/datasets/
├── raw/
│   ├── videos/              # Raw MP4/AVI/MOV match clips
│   └── annotations/         # Raw annotations from CVAT / Roboflow
├── processed/
│   ├── frames/              # Extracted image frames (PNG/JPG)
│   ├── labels/              # Normalized YOLO .txt label files
│   └── deliveries.parquet   # Delivery-level tabular kinematics
├── sample_videos/           # Curated test clips for verification
└── synthetic/               # Physics-accurate generated validation deliveries
```

---

## 3. Computer Vision Annotation Protocols

### 3.1 Bounding Box Format (YOLO Standard)
For single-frame ball detection (YOLO):
Each frame `frame_000123.jpg` must have an associated `frame_000123.txt`:
```text
<class_id> <x_center> <y_center> <width> <height>
```
- `<class_id>`: Always `0` (Cricket Ball).
- `<x_center>`, `<y_center>`: Bounding box center coordinates normalized by image width and height $[0.0, 1.0]$.
- `<width>`, `<height>`: Box dimensions normalized by image width and height $[0.0, 1.0]$.

*Example*:
```text
0 0.524100 0.618400 0.012500 0.021000
```

### 3.2 Key Challenges & Annotation Rules
1. **Sub-Pixel Scale & Resolution**:
   - In 720p broadcast footage, a cricket ball ($72\,\text{mm}$ diameter) typically measures only $6\times 6$ to $16\times 16$ pixels.
   - Annotators must zoom in to $400\%$ to locate the exact center of mass.
2. **Motion Blur & Elongation**:
   - At $140\,\text{km/h}$ ($38.9\,\text{m/s}$), the ball moves $\sim 0.78\,\text{m}$ during a standard $1/50\,\text{s}$ shutter opening, appearing as a blurred teardrop or streak.
   - **Rule**: Annotate the leading dense core of the streak where the physical mass resides, enclosing the streak symmetrically if illumination is uniform.
3. **Temporary Occlusion & Partial Visibility**:
   - If the ball passes behind the batter, stumps, or pitch umpire, annotate with visibility status:
     - `visible`: Ball clearly identifiable.
     - `occluded`: Ball partially obscured ($<50\%$ visible); box placed on estimated visible edge.
     - `invisible`: Ball completely hidden; do not guess bounding box in YOLO datasets. The Kalman filter handles coasting across invisible frames.
4. **False Positive Suppression**:
   - Do NOT label red shoes, pitch marks, boundary cones, bowler release hands, or helmet badges.
   - When training YOLO, include negative background frames (frames without any ball) to suppress false detections.

---

## 4. Delivery-Level Tabular Schema

For trajectory reconstruction, bounce dynamics, and the **PIMA-Pitch match-adaptive learning engine**, data must adhere to the 15-field schema below:

| Field Name | Type | Unit | Description | Nullable |
| :--- | :--- | :--- | :--- | :--- |
| `match_id` | String | - | Unique match identifier (e.g., `M2026_IND_ENG_01`) | No |
| `delivery_id` | String | - | Over and ball index (e.g., `O14.3`) | No |
| `frame_id` | Integer | - | Monotonic frame sequence index in delivery video | No |
| `timestamp` | Float | s | Delivery elapsed time from bowler release | No |
| `ball_x` | Float | m | Lateral distance from pitch centerline (+X = leg side) | No |
| `ball_y` | Float | m | Longitudinal distance down pitch ($0.0 = \text{bowler crease}, 20.12 = \text{stumps}$) | No |
| `ball_speed` | Float | m/s | 3D instantaneous velocity magnitude $\|\mathbf{v}\|$ | No |
| `spin_rate` | Float | rpm | Ball angular velocity (revolutions per minute) | Yes |
| `seam_angle` | Float | deg | Seam orientation relative to pitch axis ($[-90^\circ, +90^\circ]$) | Yes |
| `pitch_x` | Float | m | Lateral impact coordinate on pitch surface | Yes (pre-bounce) |
| `pitch_y` | Float | m | Longitudinal impact coordinate down pitch | Yes (pre-bounce) |
| `outgoing_velocity` | Float | m/s | Post-bounce velocity vector magnitude | Yes (pre-bounce) |
| `bounce_height` | Float | m | Maximum trajectory apex or post-pitch height at batter crease | Yes (pre-bounce) |
| `pitch_region` | String | - | Pitch zone classification (`Yorker`, `Full`, `Good`, `Short`, `Bouncer`) | Yes (pre-bounce) |
| `observation_quality`| Float | $[0, 1]$ | Tracking confidence score based on detection SNR & Mahalanobis distance | No |

---

## 5. Scientific Integrity & Data Policy

> [!IMPORTANT]
> **Zero Fabrication Policy**: Never invent or fabricate physical measurements.
> 1. If broadcast footage does not provide high-speed multi-angle tracking to measure spin or seam angle, record `spin_rate = NULL` and `seam_angle = NULL`.
> 2. Pixel-space velocity ($\text{px}/\text{frame}$) must **never** be labeled as $\text{km/h}$ unless camera calibration matrix $\mathbf{K}$ and extrinsic homography $\mathbf{H}$ are experimentally established.
> 3. Synthetic physics trajectories (generated via RK4 integration) must be tagged with `data_source = "synthetic_rk4"` to ensure complete auditability against real-match data.

