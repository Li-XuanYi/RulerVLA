"""Controlled geometry intervention for tests; labels never enter reliability."""
import numpy as np

from rulervla.data.schema import Camera, finite


def inject_shared_translation(camera: Camera, bias_xyz_m):
    t = camera.camera_to_action.copy()
    t[:3, 3] += finite(bias_xyz_m, (3,), "injected translation")
    return Camera(camera.intrinsic, t, camera.resolution)


def translate_valid_map(values, valid, bias_xyz_m):
    values = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape + (3,):
        raise ValueError("Map and mask shapes differ")
    values[mask] += finite(bias_xyz_m, (3,), "injected translation")
    values[~mask] = 0
    return values
