"""Uncertainty readouts for flow-matching policies.

The :mod:`openpi.uncertainty.accel` module contains a numpy-only implementation
of the denoising-acceleration proxy from ``fm-geometry``.  It intentionally
does not import JAX, Flax, Torch, or a robot environment, so a recorded FM
trajectory can be scored offline on a CPU-only analysis machine.  The same holds
for :mod:`openpi.uncertainty.cusum`, :mod:`openpi.uncertainty.calibration`, and
:mod:`openpi.uncertainty.accel_runtime`, which is what lets a serving process,
an offline fitting script, and a test all share one implementation.
"""

from openpi.uncertainty.accel import DEFAULT_ACCEL_MODE
from openpi.uncertainty.accel import AccelConfig
from openpi.uncertainty.accel import ReferenceStd
from openpi.uncertainty.accel import action_level_accel
from openpi.uncertainty.accel import load_fixed_std
from openpi.uncertainty.accel import parse_accel_mode
from openpi.uncertainty.accel import prefix_depth
from openpi.uncertainty.accel import prefix_layers
from openpi.uncertainty.accel import resolve_accel_mode
from openpi.uncertainty.accel import save_fixed_std
from openpi.uncertainty.accel import score_chunks
from openpi.uncertainty.accel import whole_chunk_curvature
from openpi.uncertainty.accel import whole_chunk_prefix_accel
from openpi.uncertainty.accel_runtime import AccelRuntime
from openpi.uncertainty.accel_runtime import AccelRuntimeConfig
from openpi.uncertainty.accel_runtime import episodes_from_records
from openpi.uncertainty.accel_runtime import streams_from_records
from openpi.uncertainty.calibration import AccelCalibration
from openpi.uncertainty.calibration import digest_std
from openpi.uncertainty.cusum import AccelMonitor
from openpi.uncertainty.cusum import calibrate
from openpi.uncertainty.cusum import conformal_height
from openpi.uncertainty.cusum import cusum_alarm
from openpi.uncertainty.cusum import cusum_peak
from openpi.uncertainty.cusum import evaluate_streams

__all__ = [
    "DEFAULT_ACCEL_MODE",
    "AccelCalibration",
    "AccelConfig",
    "AccelMonitor",
    "AccelRuntime",
    "AccelRuntimeConfig",
    "ReferenceStd",
    "action_level_accel",
    "calibrate",
    "conformal_height",
    "cusum_alarm",
    "cusum_peak",
    "digest_std",
    "episodes_from_records",
    "evaluate_streams",
    "load_fixed_std",
    "parse_accel_mode",
    "prefix_depth",
    "prefix_layers",
    "resolve_accel_mode",
    "save_fixed_std",
    "score_chunks",
    "streams_from_records",
    "whole_chunk_curvature",
    "whole_chunk_prefix_accel",
]
