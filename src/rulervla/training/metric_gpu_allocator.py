import jax
import jax.numpy as jnp


def reserve_lora_gpu_pools(devices, pool_target_mib):
    # 在真实设备上同时持有接近整池的数组，保证BFC实际取得预算后再读取NAS。
    target = (pool_target_mib - 2048) << 20
    tokens = tuple(jax.device_put(0, device) for device in devices)
    jax.block_until_ready(tokens)
    grouped = []
    for device, token in zip(devices, tokens, strict=True):
        remaining = target
        chunks = []
        allocate = jax.jit(lambda value, length: jnp.full((length,), value, dtype=jnp.uint32),
                           static_argnums=(1,), device=device)
        while remaining:
            size = min(2 << 30, remaining)
            elements = size // 4
            chunks.append(allocate(token, elements))
            remaining -= size
        grouped.append(chunks)
    jax.block_until_ready(grouped)
    return dict(initial_reservation_array_bytes_per_device=[
                    sum(array.nbytes for array in chunks) for chunks in grouped],
                initial_bfc_memory_stats=[device.memory_stats() for device in devices])
