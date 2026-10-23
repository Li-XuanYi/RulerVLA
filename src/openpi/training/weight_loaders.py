import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class SixChannelCheckpointWeightLoader(WeightLoader):
    """Load all original weights, expanding only the SigLIP input kernel."""

    params_path: str
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return merge_six_channel_params(loaded, params, missing_regex=self.missing_regex)


def merge_six_channel_params(loaded_params: at.Params, params: at.Params,
                             *, missing_regex: str = ".*lora.*") -> at.Params:
    """Fail on every unexpected mismatch except a controlled RGB→RGB+Cue stem."""
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    stem_paths = [key for key in flat_ref if key.endswith("PaliGemma/img/embedding/kernel")]
    if len(stem_paths) != 1 or stem_paths[0] not in flat_loaded:
        raise ValueError(f"Expected exactly one loaded SigLIP patch kernel; found {stem_paths}")
    unexpected = set(flat_loaded) - set(flat_ref)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint parameters: {sorted(unexpected)[:5]}")
    allowed_missing = re.compile(missing_regex)
    missing = {key for key in flat_ref if key not in flat_loaded}
    disallowed = {key for key in missing if not allowed_missing.fullmatch(key)}
    if disallowed:
        raise ValueError(f"Missing checkpoint parameters: {sorted(disallowed)[:5]}")
    result = {}
    stem = stem_paths[0]
    for key, reference in flat_ref.items():
        if key in missing:
            result[key] = reference
            continue
        value = flat_loaded[key]
        if value.shape != reference.shape:
            old_shape, new_shape = value.shape, reference.shape
            if (key != stem or len(old_shape) != 4 or len(new_shape) != 4
                    or old_shape[:2] != new_shape[:2] or old_shape[2] != 3
                    or new_shape[2] != 6 or old_shape[3] != new_shape[3]):
                raise ValueError(f"Checkpoint shape mismatch at {key}: {old_shape} vs {new_shape}")
            value = np.concatenate((value, np.zeros(new_shape[:2] + (3, new_shape[3]), dtype=value.dtype)), axis=2)
        result[key] = value.astype(reference.dtype) if value.dtype != reference.dtype else value
    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
