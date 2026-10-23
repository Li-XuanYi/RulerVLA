import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_hf_dataset_cache_prefers_explicit_path(tmp_path, monkeypatch):
    import datasets

    original_cache = datasets.config.HF_DATASETS_CACHE
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "explicit"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", original_cache)

    cache_path = _data_loader._configure_hf_dataset_cache()  # noqa: SLF001

    assert cache_path == str((tmp_path / "explicit").resolve())
    assert (tmp_path / "explicit").is_dir()
    assert cache_path == datasets.config.HF_DATASETS_CACHE


def test_hf_dataset_cache_colocates_with_lerobot_home(tmp_path, monkeypatch):
    import datasets

    original_cache = datasets.config.HF_DATASETS_CACHE
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "huggingface" / "lerobot"))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", original_cache)

    cache_path = _data_loader._configure_hf_dataset_cache()  # noqa: SLF001
    expected = (tmp_path / "huggingface" / "datasets").resolve()

    assert cache_path == str(expected)
    assert expected.is_dir()
    assert str(expected) == datasets.config.HF_DATASETS_CACHE


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
