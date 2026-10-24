"""One observation/prompt/normalization implementation for training and inference."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.shared import normalize
from rulervla.adapters.metric_cue import MetricViewAdapter, SixChannelInputs
from rulervla.config import load_metric_method_config
from rulervla.cues.metric_prompt import StrictPromptTokenizer
from rulervla.cues.motion_echo import MotionEchoConfig


def tokenizer_from_asset(path, max_len):
    import sentencepiece
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = max_len
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=Path(path).read_bytes())
    return tokenizer


def metric_model_config():
    """The G1-validated architecture; no alternate input/model recipe."""
    return Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, image_channels=6)


class MetricPipeline:
    # Rendering changes affect Cue pixels even when the geometric units stay fixed.
    version = 'metric_cue_observation_v2'

    def __init__(self, geometry, ruler, norm_stats, *, echo=MotionEchoConfig(), tokenizer=None):
        if echo.enabled:
            raise ValueError('G2 workflow requires motion_echo.enabled=false')
        self.geometry, self.ruler, self.echo = geometry, ruler, echo
        self.model_config = metric_model_config()
        self.norm_stats = norm_stats
        for key, dim in (('state', 8), ('actions', 7)):
            if key not in norm_stats:
                raise ValueError(f'Missing {key} normalization statistics')
            stat = norm_stats[key]
            if (stat.q01 is None or stat.q99 is None or np.shape(stat.q01) != (dim,)
                    or np.shape(stat.q99) != (dim,) or not np.isfinite(stat.q01).all()
                    or not np.isfinite(stat.q99).all() or np.any(stat.q99 < stat.q01)):
                raise ValueError(f'Invalid quantile statistics for {key}')
        self.adapter = MetricViewAdapter(geometry, ruler, echo)
        self.inputs = SixChannelInputs(model_type=self.model_config.model_type)
        self.tokenizer = StrictPromptTokenizer(tokenizer or PaligemmaTokenizer(self.model_config.max_token_len))
        self.tokenizer_bytes = self.tokenizer.base._tokenizer.serialized_model_proto()
        self.tokenizer_sha256 = hashlib.sha256(self.tokenizer_bytes).hexdigest()
        self.last_prompt = None

    @classmethod
    def from_files(cls, cue_config, norm_stats_dir):
        geometry, ruler, echo = load_metric_method_config(cue_config)
        return cls(geometry, ruler, normalize.load(norm_stats_dir), echo=echo)

    def contract(self):
        return dict(version=self.version, geometry=self.geometry, metric_ruler=asdict(self.ruler),
                    motion_echo=asdict(self.echo), model=asdict(self.model_config),
                    tokenizer=dict(source='gs://big_vision/paligemma_tokenizer.model',
                                   sha256=self.tokenizer_sha256, max_len=self.model_config.max_token_len),
                    quantile_norm=True, norm_stats=json.loads(normalize.serialize_json(self.norm_stats)))

    def fingerprint(self):
        value=json.dumps(self.contract(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(value.encode()).hexdigest()

    def _current(self, sample):
        # Future targets and sample identity remain outside the model input path.
        current = {key: value for key, value in sample.items() if key != 'actions'}
        prepared = self.adapter.prepare(current)
        self.last_prompt = prepared['prompt']
        structured = self.inputs(prepared)
        structured = transforms.Normalize({'state': self.norm_stats['state']},
                                          use_quantiles=True, strict=True)(structured)
        structured = transforms.TokenizePrompt(self.tokenizer, discrete_state_input=False)(structured)
        structured = transforms.PadStatesAndActions(32)(structured)
        structured['state'] = np.asarray(structured['state'], np.float32)
        structured['tokenized_prompt'] = np.asarray(structured['tokenized_prompt'], np.int32)
        structured['tokenized_prompt_mask'] = np.asarray(structured['tokenized_prompt_mask'], np.bool_)
        return structured

    def prepare_train(self, sample):
        actions = np.asarray(sample['actions'])
        if actions.shape != (10, 7) or not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
            raise ValueError('Real action chunk must be finite floating (10, 7)')
        result = self._current(sample)
        normalized = transforms.Normalize({'actions': self.norm_stats['actions']},
            use_quantiles=True, strict=True)({'actions': actions.copy()})['actions']
        result['actions'] = np.asarray(transforms.pad_to_dim(normalized, 32), np.float32)
        return result

    def prepare_inference(self, sample):
        return self._current(sample)

    def postprocess_actions(self, actions):
        actions = np.asarray(actions)
        if actions.shape[-2:] != (10, 32) or not np.isfinite(actions).all():
            raise ValueError('Model actions must be finite [...,10,32]')
        raw = transforms.Unnormalize({'actions': self.norm_stats['actions']},
            use_quantiles=True)({'actions': actions})['actions']
        return np.asarray(raw[..., :7], np.float32)
