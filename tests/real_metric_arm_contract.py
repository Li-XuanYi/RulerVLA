import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np

from openpi.shared import normalize
from rulervla.adapters.metric_pipeline import MetricPipeline
from rulervla.config import load_metric_method_config
from rulervla.cues.metric_prompt import build_scale_prompt
from rulervla.cues.motion_echo import EpisodeHistoryBuffer, MetricInferenceHistory
from rulervla.training.main_data import SpatialDataset, action_chunk_indices
from rulervla.training.metric_resources import load_run_config
from rulervla.training.metric_workflow import make_training_config, prepare_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--check-policy", action="store_true")
    parser.add_argument("--check-accumulation-shape", action="store_true")
    parser.add_argument("--check-rollout-session", action="store_true")
    parser.add_argument("--check-sharding-shape", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = load_run_config(root / "configs/rulervla_metric_workflow.yaml")
    cache_root = args.cache_root.resolve()
    rgbd_root = cache_root / Path(run["data"]["rgbd_root"]).relative_to("/")
    geometry_root = cache_root / Path(run["data"]["geometry_root"]).relative_to("/")
    dataset = SpatialDataset(
        rgbd_root,
        geometry_root,
        path_mappings={"/home/Dataset2": str(cache_root / "home/Dataset2")},
    )
    geometry, ruler, echo = load_metric_method_config(run["cue_config"])
    statistics = normalize.load(run["norm_stats_dir"])
    sample = dataset[20]
    history = dataset.get_history(20, history_steps=echo.history_steps)
    if not history.valid:
        raise AssertionError("真实样本应有同一 episode 的五步历史")
    if not np.allclose(history.current_m - history.past_m, dataset.cache[1261]["ee_world_m"][20] - dataset.cache[1261]["ee_world_m"][15]):
        raise AssertionError("历史位移与真实本体状态不一致")

    arms = {
        "M0": load_metric_method_config(root / "configs/rulervla_metric_cue_m0.yaml")[1:],
        "M1": load_metric_method_config(root / "configs/rulervla_metric_cue_m1.yaml")[1:],
        "M2": (ruler, echo),
        "M3": load_metric_method_config(root / "configs/rulervla_metric_cue_m3.yaml")[1:],
    }
    results = {}
    for name, (arm_ruler, arm_echo) in arms.items():
        pipeline = MetricPipeline(geometry, arm_ruler, statistics, echo=arm_echo)
        for instruction in {row["task"] for row in dataset.rows}:
            prompt = build_scale_prompt(
                instruction, arm_ruler,
                echo_history_steps=arm_echo.history_steps if arm_echo.enabled else None,
                echo_valid=True if arm_echo.enabled else None,
            )
            pipeline.tokenizer.tokenize(prompt)
        current_history = history if arm_echo.enabled else None
        trained = pipeline.prepare_train(sample, history=current_history)
        inferred = pipeline.prepare_inference(sample, history=current_history)
        actions = trained.pop("actions")
        if actions.shape != (10, 32):
            raise AssertionError(f"{name} 的动作块形状不正确")
        for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
            np.testing.assert_array_equal(trained[key], inferred[key])
        for slot in trained["image"]:
            np.testing.assert_array_equal(trained["image"][slot], inferred["image"][slot])
        cue = inferred["image"]["base_0_rgb"][..., 3:]
        if name == "M0" and np.any(cue):
            raise AssertionError("M0 的空间提示应为空白")
        if name != "M0" and not np.any(cue):
            raise AssertionError(f"{name} 的空间提示没有绘制内容")
        if name in ("M0", "M1") and "cm" in pipeline.last_prompt:
            raise AssertionError(f"{name} 的 Prompt 泄漏公制刻度")
        if name in ("M2", "M3") and "5 cm" not in pipeline.last_prompt:
            raise AssertionError(f"{name} 的 Prompt 缺少公制分度值")
        if name == "M3" and "5 environment control steps" not in pipeline.last_prompt:
            raise AssertionError("M3 的 Prompt 缺少真实历史窗口")
        results[name] = {"cue_pixels": int(np.count_nonzero(cue)), "prompt": pipeline.last_prompt}

    m3_pipeline = MetricPipeline(geometry, arms["M3"][0], statistics, echo=arms["M3"][1])
    batch = prepare_batch(dataset, [20, 21], m3_pipeline)
    if batch.actions.shape != (2, 10, 32):
        raise AssertionError("M3 批次没有连续动作")
    if "5 environment control steps" not in m3_pipeline.last_prompt:
        raise AssertionError("M3 训练批次没有使用真实历史状态")
    protocol = json.loads((root / "experiments/metric_cue_v1/protocol.json").read_text())
    formal = dict(run, training=dict(protocol["training"], seed=42),
                  resources=dict(run["resources"], gpu_indices=[4, 5]))
    formal_config = make_training_config(formal)
    if (formal_config.batch_size != 32 or formal_config.ema_decay != 0.999
            or formal_config.fsdp_devices != 2
            or formal_config.num_train_steps != 30000
            or formal_config.lr_schedule.warmup_steps != 1000):
        raise AssertionError("正式训练配方与协议不一致")
    formal_batch = prepare_batch(dataset, list(range(32)), m3_pipeline)
    if formal_batch.actions.shape != (32, 10, 32):
        raise AssertionError("正式训练批次没有使用 32 条真实动作序列")
    buffer = EpisodeHistoryBuffer(history_steps=echo.history_steps)
    buffer.reset(1261)
    for step in range(15, 21):
        buffer.observe(step, dataset.cache[1261]["ee_world_m"][step])
    np.testing.assert_array_equal(buffer.history().displacement_m, history.displacement_m)
    online = MetricInferenceHistory(arms["M3"][1])
    online.reset_episode(1261)
    for step in range(21):
        online.observe_sample(dataset[step])
    np.testing.assert_array_equal(online.for_sample(dataset[20]).displacement_m, history.displacement_m)
    online.reset_episode(1262)
    online.observe_sample(dataset[110])
    if online.for_sample(dataset[110]).valid:
        raise AssertionError("新 episode 的历史必须清空")
    m3_pipeline.prepare_inference(dataset[110], history=online.for_sample(dataset[110]))
    if "No valid Motion Echo" not in m3_pipeline.last_prompt:
        raise AssertionError("新 episode 的无效历史提示不正确")
    indices, valid = action_chunk_indices(108, 110)
    if indices.tolist() != [108, 109, 109, 109, 109, 109, 109, 109, 109, 109] or valid.sum() != 2:
        raise AssertionError("episode 尾部动作填充不正确")
    if (args.check_policy or args.check_accumulation_shape or args.check_rollout_session
            or args.check_sharding_shape):
        from rulervla.training.metric_workflow import MetricPolicy, MetricTrainer, load_initial_model
        if not hasattr(MetricPolicy, "reset_episode") or not hasattr(MetricPolicy, "observe_control_step"):
            raise AssertionError("在线策略缺少逐控制步历史入口")
        training_config = make_training_config(run)
        model = load_initial_model(training_config)
        if args.check_policy:
            policy = MetricPolicy(model, m3_pipeline)
            policy.reset_episode(1261)
            paired_rng = np.asarray(jax.random.key_data(policy.rng))
            policy.reset_episode(1262)
            if np.array_equal(jax.random.key_data(policy.rng), paired_rng):
                raise AssertionError("不同场景不应复用相同推理随机状态")
            policy.reset_episode(1261)
            np.testing.assert_array_equal(jax.random.key_data(policy.rng), paired_rng)
            for step in range(21):
                policy.observe_control_step(dataset[step])
            prepared = policy.prepare(dataset[20])
            expected = m3_pipeline.prepare_inference(dataset[20], history=history)
            for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
                np.testing.assert_array_equal(prepared[key], expected[key])
            for slot in prepared["image"]:
                np.testing.assert_array_equal(prepared["image"][slot], expected["image"][slot])
        if args.check_accumulation_shape:
            import flax.nnx as nnx
            graph, params = nnx.split(model)
            trainer = MetricTrainer(formal_config, graph)
            observations = jax.tree.map(lambda value: value.reshape((32, 1) + value.shape[1:]),
                                        formal_batch.observation)
            actions = formal_batch.actions.reshape((32, 1, 10, 32))
            _, metric_shapes = jax.eval_shape(trainer._accumulated_backward, params,
                                              observations, actions, np.int32(0))
            if metric_shapes["cue_grad_l2"].shape != ():
                raise AssertionError("梯度累积的 Cue 梯度指标形状不正确")
        if args.check_sharding_shape:
            import flax.nnx as nnx
            from openpi.training import sharding
            from rulervla.training.metric_workflow import STEM, flat_params, rgb_state_from_six
            if len(jax.devices("cpu")) != 2:
                raise AssertionError("分片结构检查需要两台真实 CPU 设备")
            graph, params = nnx.split(model)
            mesh = sharding.make_mesh(2)
            trainer = MetricTrainer(formal_config, graph, gradient_mesh=mesh)
            trainer.configure_sharding(params)
            if "fsdp" not in flat_params(trainer.parameter_sharding)[STEM].spec:
                raise AssertionError("六通道视觉卷积参数没有跨设备分片")
            rgb_config = dataclasses.replace(formal_config.model, image_channels=3)
            rgb_reference = nnx.state(nnx.eval_shape(rgb_config.create, jax.random.key(42)))
            rgb_sharding = sharding.fsdp_sharding(rgb_reference, mesh)
            if not any("fsdp" in item.spec for item in flat_params(rgb_sharding).values()):
                raise AssertionError("三通道视觉模型没有跨设备分片")
            rgb_state = rgb_state_from_six(params)
            if flat_params(rgb_state)[STEM].shape[2] != 3 or flat_params(params)[STEM].shape[2] != 6:
                raise AssertionError("三通道对照状态改变了六通道训练参数")
            rgb_graph = nnx.split(nnx.eval_shape(rgb_config.create, jax.random.key(42)))[0]
            nnx.merge(rgb_graph, rgb_state)
            observations = jax.tree.map(lambda value: value.reshape((32, 1) + value.shape[1:]),
                                        formal_batch.observation)
            actions = formal_batch.actions.reshape((32, 1, 10, 32))
            with sharding.set_mesh(mesh):
                _, metric_shapes = jax.eval_shape(trainer._accumulated_backward, params,
                                                  observations, actions, np.int32(0))
            if metric_shapes["cue_grad_l2"].shape != ():
                raise AssertionError("分片梯度累积的指标形状不正确")
        if args.check_rollout_session:
            from rulervla.evaluation.metric_rollout import MetricRolloutSession
            session = MetricRolloutSession(MetricPolicy(model, m3_pipeline))
            session.infer({"command": "reset", "episode_index": 1261})
            for step in range(21):
                session.infer({"command": "observe", "sample": dataset[step]})
            prediction = session.infer({"command": "act"})
            if prediction["actions"].shape != (10, 7):
                raise AssertionError("真实模型在线动作块形状不正确")
    print(json.dumps({"state": "complete", "arms": results, "real_samples": 2}, ensure_ascii=False))


if __name__ == "__main__":
    main()
