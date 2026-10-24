"""A renderer fix must not silently migrate old checkpoint preprocessing."""
import json

import pytest

from rulervla.adapters.metric_pipeline import MetricPipeline
from rulervla.cues.metric_ruler import MetricCueConfig
from rulervla.cues.motion_echo import MotionEchoConfig
from rulervla.training.metric_workflow import CHECKPOINT_SCHEMA, pipeline_from_checkpoint


def test_g2_still_rejects_enabled_echo():
    with pytest.raises(ValueError, match='G2 workflow requires motion_echo.enabled=false'):
        MetricPipeline({}, MetricCueConfig(), {}, echo=MotionEchoConfig(enabled=True))


def test_previous_renderer_checkpoint_requires_its_original_runtime(tmp_path):
    assert MetricPipeline.version == 'metric_cue_observation_v2'
    metadata = dict(schema=CHECKPOINT_SCHEMA, pipeline=dict(version='metric_cue_observation_v1'))
    (tmp_path/'metric_workflow.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='preprocessing version differs'):
        pipeline_from_checkpoint(tmp_path)


def test_experimental_direction_styles_remain_outside_this_fix():
    with pytest.raises(ValueError, match='Unsupported direction style'):
        MetricCueConfig(negative_style='solid_hollow')
