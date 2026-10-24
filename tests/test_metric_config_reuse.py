from pathlib import Path
import sys

import pytest

from rulervla.config import load_metric_cue_config, load_metric_method_config


@pytest.mark.parametrize('loader', [load_metric_cue_config, load_metric_method_config])
def test_each_config_entry_parses_yaml_once(loader):
    path = Path(__file__).resolve().parents[1]/'configs/rulervla_metric_cue.yaml'
    calls = []

    def count_yaml(frame, event, argument):
        if event == 'call' and frame.f_code.co_name == 'safe_load':
            calls.append(frame.f_code.co_filename)

    previous = sys.getprofile()
    try:
        sys.setprofile(count_yaml)
        result = loader(path)
    finally:
        sys.setprofile(previous)
    assert len(calls) == 1
    assert result[0]['canonical_to_policy'] == 'hflip'
    assert result[1].tick_spacing_m == .05
    if loader is load_metric_method_config:
        assert result[2].enabled is False
