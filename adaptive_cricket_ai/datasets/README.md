# Adaptive Cricket AI Datasets

Refer to the complete dataset specification and annotation standards in:
[Dataset Download & Annotation Guide](../reports/dataset_guide.md)

### Directory Structure
- `raw/`: Unprocessed match video files (.mp4, .avi, .mov) and raw CVAT / Roboflow annotations.
- `processed/`: Extracted frames, normalized YOLO label text files, and tabular trajectory parquets.
- `sample_videos/`: Curated short delivery clips for integration testing and calibration.
- `synthetic/`: Synthesized aerodynamic test deliveries with exact physical ground truth.
- `annotations/`: Point and bounding box ground truth annotations.

### Tabular Delivery Schema
Deliveries saved in `processed/` follow the 15-field standard schema:
`match_id, delivery_id, frame_id, timestamp, ball_x, ball_y, ball_speed, spin_rate, seam_angle, pitch_x, pitch_y, outgoing_velocity, bounce_height, pitch_region, observation_quality`.

