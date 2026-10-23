from types import SimpleNamespace

from torch import nn

from openpi.models import gemma
from openpi.models_pytorch import gemma_pytorch


def test_vision_projector_width_matches_vlm_width(monkeypatch):
    class FakePaliGemma(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class FakeGemma(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = SimpleNamespace(embed_tokens=object())

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", FakePaliGemma)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", FakeGemma)
    vlm_config = gemma.get_config("dummy")

    model = gemma_pytorch.PaliGemmaWithExpertModel(vlm_config, gemma.get_config("dummy"))

    assert model.paligemma.config.vision_config.projection_dim == vlm_config.width
