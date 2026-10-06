"""Average microbatch gradients before one AdamW/schedule/EMA update."""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax


def accumulated_gradients(config, rng, graph, full_params, step, batches):
    """Leading batch axes are [accumulation, global_microbatch, ...]."""
    observation, actions = batches
    count = actions.shape[0]
    params = full_params.filter(config.trainable_filter)
    zero = jax.tree.map(jnp.zeros_like, params)
    update_rng = jax.random.fold_in(rng, step)

    def micro(carry, index):
        total, total_loss = carry
        policy = nnx.merge(graph, full_params)
        policy.train()
        obs = jax.tree.map(lambda value: value[index], observation)
        labels = actions[index]

        def loss_fn(model):
            return jnp.mean(model.compute_loss(jax.random.fold_in(update_rng, index), obs, labels, train=True))

        loss, gradients = nnx.value_and_grad(loss_fn,
            argnums=nnx.DiffState(0, config.trainable_filter))(policy)
        total = jax.tree.map(lambda a, b: a + b / count, total, gradients)
        return (total, total_loss + loss / count), None

    (gradients, loss), _ = jax.lax.scan(micro, (zero, jnp.zeros((), jnp.float32)), jnp.arange(count))
    return gradients, {'loss': loss, 'grad_norm': optax.global_norm(gradients)}


def apply_gradients(config, state, gradients):
    """Exactly one clip/AdamW/schedule/EMA update, after all microbatches."""
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(gradients, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    policy = nnx.merge(state.model_def, state.params)
    nnx.update(policy, new_params)
    new_params = nnx.state(policy)
    ema = None if state.ema_decay is None else jax.tree.map(
        lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
        state.ema_params, new_params)
    return dataclasses.replace(state, step=state.step + 1, params=new_params,
        opt_state=opt_state, ema_params=ema)


def accumulated_step(config, rng, state, batches):
    gradients, info = accumulated_gradients(config, rng, state.model_def, state.params, state.step, batches)
    return apply_gradients(config, state, gradients), info
