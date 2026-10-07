"""CPU views for a strictly blocking save, never for donated asynchronous use."""
import hashlib

import jax
import numpy as np


def blocking_cpu_snapshot(state):
    """Caller must pause updates until save wait/publish/verification all finish."""
    jax.block_until_ready(state)
    leaves, structure = jax.tree_util.tree_flatten_with_path(state)
    arrays, records = [], []
    for path, value in leaves:
        assert isinstance(value,jax.Array), 'Only CPU train-state array leaves are supported'
        assert all(device.platform=='cpu' for device in value.devices())
        array=np.asarray(jax.device_get(value))
        assert array.dtype==value.dtype and array.shape==value.shape
        arrays.append(array)
        records.append(dict(path=jax.tree_util.keystr(path),shape=list(array.shape),dtype=str(array.dtype),
            bytes=array.nbytes,sha256=hashlib.sha256(memoryview(array.reshape(-1).view(np.uint8))).hexdigest()))
    return jax.tree_util.tree_unflatten(structure,arrays),dict(
        schema='rulervla_blocking_cpu_snapshot_v1',independent_copy=False,
        contract='No update/donation until Orbax wait, publication and verification finish',
        total_bytes=sum(r['bytes'] for r in records),leaves=records)


def verify_blocking_snapshot(snapshot,evidence):
    leaves=jax.tree_util.tree_flatten_with_path(snapshot)[0]
    assert len(leaves)==len(evidence['leaves'])
    for (path,value),record in zip(leaves,evidence['leaves'],strict=True):
        array=np.asarray(value)
        assert jax.tree_util.keystr(path)==record['path']
        assert list(array.shape)==record['shape'] and str(array.dtype)==record['dtype']
        assert hashlib.sha256(memoryview(array.reshape(-1).view(np.uint8))).hexdigest()==record['sha256']
