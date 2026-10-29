import jax


def reserve_lora_gpu_pools(devices, pool_target_mib):
    # 原生BFC统计确认已经取得的CUDA池；该初始化发生在模型和NAS读取之前。
    minimum = (pool_target_mib - 2048) << 20
    tokens = tuple(jax.device_put(0, device) for device in devices)
    jax.block_until_ready(tokens)
    stats = [device.memory_stats() for device in devices]
    reserved = [value['pool_bytes'] for value in stats]
    if any(value < minimum for value in reserved):
        raise RuntimeError('LoRA CUDA pool initialization did not obtain its reserved budget')
    return dict(initial_reserved_pool_bytes_per_device=reserved, initial_bfc_memory_stats=stats)
