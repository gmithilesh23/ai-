"""Unit Tests for Camera Homography & Perspective Projections"""

import numpy as np
from cricket_vision.physics.homography import CameraModel
from cricket_vision.physics.pitch_geometry import CricketPitch


def test_camera_projection_in_front_of_camera():
    cam = CameraModel(resolution=(1280, 720))
    # Stumps are in front of camera
    stump_pos = np.array([0.0, 20.12, 0.5])
    u, v, valid = cam.world_to_pixel(stump_pos)

    assert valid is True
    assert 0 <= u <= 1280
    assert 0 <= v <= 720


def test_pixel_to_ground_cycle():
    cam = CameraModel(resolution=(1280, 720))
    # Test point on ground surface
    target_ground = np.array([0.2, 14.5, 0.0])
    u, v, valid = cam.world_to_pixel(target_ground)
    assert valid is True

    # Invert via homography
    reconstructed_ground = cam.pixel_to_ground(u, v)

    # Reconstructed X, Y should match target within 5mm
    assert abs(reconstructed_ground[0] - target_ground[0]) < 0.02
    assert abs(reconstructed_ground[1] - target_ground[1]) < 0.02
    assert reconstructed_ground[2] == 0.0


def test_ray_reconstruction_with_height_prior():
    cam = CameraModel(resolution=(1280, 720))
    original_pt = np.array([-0.3, 10.0, 1.85])
    u, v, valid = cam.world_to_pixel(original_pt)
    assert valid is True

    reconstructed_pt = cam.reconstruct_3d_point(u, v, height_prior=1.85)
    assert np.linalg.norm(reconstructed_pt - original_pt) < 0.03

