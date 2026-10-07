"""Only immutable CPU snapshots may be used by the blocking save path."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rulervla.training.blocking_snapshot import blocking_cpu_snapshot, verify_blocking_snapshot


class BlockingSnapshotTests(unittest.TestCase):
    def setUp(self):
        cpu=jax.devices('cpu')[0]
        self.state=jax.device_put(dict(step=jnp.array(2,jnp.int32),params=jnp.arange(6,dtype=jnp.float32).reshape(2,3),
            optimizer=jnp.ones((2,3),jnp.float32),ema=jnp.ones((2,3),jnp.bfloat16)),cpu)

    def test_full_cpu_state_hashes_and_dtypes_survive(self):
        snapshot,evidence=blocking_cpu_snapshot(self.state)
        verify_blocking_snapshot(snapshot,evidence)
        for actual,original in zip(jax.tree.leaves(snapshot),jax.tree.leaves(self.state),strict=True):
            self.assertEqual(actual.dtype,original.dtype)
            np.testing.assert_array_equal(actual,jax.device_get(original))
        self.assertFalse(evidence['independent_copy'])

    def test_mutated_snapshot_cannot_pass(self):
        snapshot,evidence=blocking_cpu_snapshot(self.state)
        snapshot={k:np.array(v,copy=True) for k,v in snapshot.items()}
        snapshot['params'][0,0]=123
        with self.assertRaises(AssertionError):verify_blocking_snapshot(snapshot,evidence)

    def test_missing_leaf_cannot_pass(self):
        snapshot,evidence=blocking_cpu_snapshot(self.state)
        evidence=deepcopy(evidence);evidence['leaves']=evidence['leaves'][:-1]
        with self.assertRaises(AssertionError):verify_blocking_snapshot(snapshot,evidence)


if __name__=='__main__':unittest.main()
