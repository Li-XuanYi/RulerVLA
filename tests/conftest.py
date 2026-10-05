from pathlib import Path

import numpy as np
import pytest

from rulervla.adapters.libero import LiberoAdapter
from rulervla.config import load_config
from rulervla.geometry.action_coord import canonical_geometry
from rulervla.geometry.visual_anchor import SimulatorGTAnchorProvider


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "configs/v05_default.yaml")


@pytest.fixture
def sample():
    yy, xx = np.indices((256, 256))
    rgb = np.stack((xx, yy, (xx * 3 + yy) % 256), axis=-1).astype(np.uint8)
    ee = np.array([0, 0, 1], dtype=np.float32)
    return dict(agentview_rgb=rgb, wrist_rgb=rgb[::-1].copy(),
                agentview_depth_m=np.full((256, 256), 2, dtype=np.float32),
                intrinsic=np.array([[200., 0, 128.], [0, 200., 128.], [0, 0, 1.]]),
                camera_to_world=np.eye(4), ee_world_m=ee, state8=np.r_[ee, np.zeros(5)].astype(np.float32),
                eef_quaternion_xyzw=np.array([0., 0, 0, 1.]), task="pick up the black bowl",
                actions=np.zeros((10, 7), dtype=np.float32))


@pytest.fixture
def landmarks(config, sample):
    camera = LiberoAdapter(config["geometry"]).camera(sample)
    uv = np.array([[80, 90], [110, 90], [140, 90], [170, 90],
                   [80, 140], [110, 140], [140, 140], [170, 140]], dtype=float)
    camera_points = np.c_[uv, np.ones(8)] @ np.linalg.inv(camera.intrinsic).T * 2
    points = camera_points - sample["ee_world_m"]
    depth = canonical_geometry(sample, config["geometry"])["depth"]
    provider = SimulatorGTAnchorProvider(points, sample["ee_world_m"], np.eye(3), camera, depth)
    return points, provider
