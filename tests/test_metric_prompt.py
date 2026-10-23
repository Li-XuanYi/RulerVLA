from dataclasses import replace
from pathlib import Path

import pytest

from rulervla.config import load_metric_cue_config
from rulervla.cues.metric_prompt import StrictPromptTokenizer, build_scale_prompt


def test_ruler_and_prompt_read_one_config_file():
    config = Path(__file__).parents[1] / 'configs/rulervla_metric_cue.yaml'
    _, ruler = load_metric_cue_config(config)
    first = build_scale_prompt('Pick up the black bowl.', ruler)
    second = build_scale_prompt('Pick up the black bowl.', replace(ruler, tick_spacing_m=0.03))
    assert first.startswith('Task: Pick up the black bowl.')
    assert '5 cm' in first and '3 cm' in second
    assert 'Robot world XYZ' in first and 'red for X' in first
    assert 'Solid lines and filled ticks' in first
    assert 'Dashed lines and hollow ticks' in first


class FakeSentencePiece:
    def encode(self, text, add_bos=False):
        return list(range(len(text.split()) + int(add_bos)))


class FakeTokenizer:
    def __init__(self, limit):
        self._max_len = limit
        self._tokenizer = FakeSentencePiece()

    def tokenize(self, prompt, state=None):
        return prompt, state


def test_strict_tokenizer_rejects_truncation():
    prompt = build_scale_prompt('Pick up the black bowl.', load_metric_cue_config_default())
    with pytest.raises(ValueError, match='truncat'):
        StrictPromptTokenizer(FakeTokenizer(10)).tokenize(prompt)
    assert StrictPromptTokenizer(FakeTokenizer(500)).tokenize(prompt)[0] == prompt


def test_echo_prompt_distinguishes_missing_history():
    ruler = load_metric_cue_config_default()
    valid = build_scale_prompt('pick bowl', ruler, echo_history_steps=5, echo_valid=True)
    missing = build_scale_prompt('pick bowl', ruler, echo_history_steps=5, echo_valid=False)
    assert '5 environment control steps ago' in valid
    assert 'No valid Motion Echo' in missing


def load_metric_cue_config_default():
    from rulervla.cues.metric_ruler import MetricCueConfig
    return MetricCueConfig()
