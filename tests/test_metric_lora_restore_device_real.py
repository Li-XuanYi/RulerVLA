import os
from pathlib import Path

import jax
import numpy as np
import pytest

from openpi.models import model as model_lib
from rulervla.training.metric_workflow import require_training_state_device, state_signatures


def test_real_lora_checkpoint_requires_explicit_cpu_device_placement():
    checkpoint = Path(os.environ['METRIC_LORA_RESTORE_CHECKPOINT'])
    cpu = jax.devices('cpu')[0]
    params = model_lib.restore_params(checkpoint / 'params', restore_type=np.ndarray)
    assert sum(value.size for value in jax.tree.leaves(params)) == 53509280
    assert all(isinstance(value, np.ndarray) for value in jax.tree.leaves(params))
    with pytest.raises(ValueError, match='device'):
        require_training_state_device(params, cpu)
    placed = jax.device_put(params, cpu)
    require_training_state_device(placed, cpu)
    assert all(value.devices() == {cpu} for value in jax.tree.leaves(placed))
    assert state_signatures(placed) == state_signatures(params)
