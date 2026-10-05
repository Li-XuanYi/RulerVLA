import numpy as np


def geometry_error(reference_m, prediction_m, valid):
    reference, prediction = np.asarray(reference_m), np.asarray(prediction_m)
    mask = np.asarray(valid, dtype=bool)
    if reference.shape != prediction.shape or reference.shape != mask.shape + (3,):
        raise ValueError("Expected equal metric XYZ maps and shared validity")
    if not mask.any():
        raise ValueError("No valid geometry for error evaluation")
    distances = np.linalg.norm(prediction[mask] - reference[mask], axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite geometry error")
    return {"mean_m": float(distances.mean()), "median_m": float(np.median(distances)),
            "max_m": float(distances.max()), "valid_pixels": int(mask.sum())}
