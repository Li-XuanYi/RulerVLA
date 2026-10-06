"""Compare accumulated updates with an independent full-batch optimizer."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip('jax')
nnx = pytest.importorskip('flax.nnx')
optax = pytest.importorskip('optax')
import jax.numpy as jnp
from flax import struct
from rulervla.training.accumulation import accumulated_step, accumulated_gradients, apply_gradients


class Regression(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array([[.2], [-.3]], jnp.float32))

    def compute_loss(self, rng, observations, actions, train):
        del rng, train
        return (observations @ self.weight.value - actions)**2


@struct.dataclass
class State:
    step: object
    params: object
    opt_state: object
    ema_params: object
    model_def: object = struct.field(pytree_node=False)
    tx: object = struct.field(pytree_node=False)
    ema_decay: object = struct.field(pytree_node=False)


@pytest.mark.parametrize('split_compilation', [False, True])
def test_accumulation_matches_full_batch_with_one_schedule_adam_and_ema_update(split_compilation):
    model = Regression()
    graph, params = nnx.split(model)
    # Nonconstant schedule and active clipping expose accidental micro-updates.
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adamw(lambda n: .01/(n+1), weight_decay=.02))
    state = State(jnp.asarray(0), params, tx.init(params), params, graph, tx, .9)
    x = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)/10
    y = jnp.arange(8, dtype=jnp.float32).reshape(4, 2, 1)/5
    reference_weight = model.weight.value
    reference_tx = optax.chain(optax.clip_by_global_norm(.5), optax.adamw(lambda n: .01/(n+1), weight_decay=.02))
    reference_opt = reference_tx.init(reference_weight)
    reference_ema = reference_weight
    config = SimpleNamespace(trainable_filter=nnx.Param)
    for update in range(3):
        loss_fn = lambda weight: jnp.mean((x.reshape(-1, 2) @ weight - y.reshape(-1, 1))**2)
        reference_gradient = jax.grad(loss_fn)(reference_weight)
        reference_updates, reference_opt = reference_tx.update(reference_gradient, reference_opt, reference_weight)
        reference_weight = optax.apply_updates(reference_weight, reference_updates)
        reference_ema = .9*reference_ema + .1*reference_weight
        if split_compilation:
            gradients, info = jax.jit(lambda params, step: accumulated_gradients(config,
                jax.random.key(42), graph, params, step, (x, y)))(state.params, state.step)
            state = jax.jit(lambda current, grad: apply_gradients(config, current, grad))(state, gradients)
        else:
            state, info = accumulated_step(config, jax.random.key(42), state, (x, y))
        assert int(state.step) == update+1
        np.testing.assert_allclose(state.params['weight'].value, reference_weight, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(state.ema_params['weight'].value, reference_ema, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(info['grad_norm'], optax.global_norm(reference_gradient), rtol=1e-6)
