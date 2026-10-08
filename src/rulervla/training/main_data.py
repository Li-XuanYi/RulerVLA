"""Deterministic frame order and exact Spatial RGB-D/geometry pairing."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from rulervla.training.storage_io import retry_io


def update_indices(frame_count, seed, update, batch_size=32):
    """Uniform shuffled epochs; resume derives the same order from update."""
    absolute = np.arange(update*batch_size, (update+1)*batch_size, dtype=np.int64)
    result = np.empty(batch_size, np.int64)
    for epoch in np.unique(absolute//frame_count):
        mask = absolute//frame_count == epoch
        order = np.random.default_rng(np.random.SeedSequence([seed, int(epoch)])).permutation(frame_count)
        result[mask] = order[absolute[mask] % frame_count]
    return result


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


class SpatialDataset:
    def __init__(self, rgbd_root, geometry_root, max_cache_bytes=None):
        self.rows = sorted(json.loads((Path(rgbd_root)/'regen_manifest.json').read_text())['episodes'],
            key=lambda row: row['episode_index'])
        geometry = json.loads((Path(geometry_root)/'manifest.json').read_text())
        assert geometry['complete'] and geometry['total_frames'] == 52970
        self.geometry = {row['episode_index']: row for row in geometry['episodes']}
        assert [row['episode_index'] for row in self.rows] == list(range(1261,1693))
        assert set(self.geometry) == set(range(1261,1693))
        self.ends = np.cumsum([row['frames'] for row in self.rows])
        assert int(self.ends[-1]) == 52970
        self.cache = OrderedDict()
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        number = int(np.searchsorted(self.ends, index, side='right'))
        row = self.rows[number]
        frame = int(index - (self.ends[number-1] if number else 0))
        ep = row['episode_index']
        if ep not in self.cache:
            loaded = retry_io(lambda: self._load_episode(row, self.geometry[ep]))
            size = sum(value.nbytes for value in loaded.values())
            if self.max_cache_bytes is not None:
                while self.cache and self.cache_bytes + size > self.max_cache_bytes:
                    _, evicted = self.cache.popitem(last=False)
                    self.cache_bytes -= sum(value.nbytes for value in evicted.values())
            self.cache[ep] = loaded
            self.cache_bytes += size
        self.cache.move_to_end(ep)
        arrays = self.cache[ep]
        sample = {key: value[frame].copy() for key,value in arrays.items() if key != 'actions'}
        indices = np.minimum(frame+np.arange(10), row['frames']-1)
        sample.update(actions=arrays['actions'][indices].copy(), task=row['task'])
        return sample

    @staticmethod
    def _load_episode(row, geometry):
        assert file_sha(row['npz']) == row['npz_sha256']
        if geometry['geometry_npz']==row['npz']:
            assert geometry['geometry_sha256']==row['npz_sha256']
        else:
            assert file_sha(geometry['geometry_npz']) == geometry['geometry_sha256']
        with np.load(row['npz'], allow_pickle=False) as z:
            arrays = {key: z[key] for key in ('agentview_rgb','wrist_rgb','agentview_depth_m','ee_world_m','actions')}
        with np.load(geometry['geometry_npz'], allow_pickle=False) as z:
            arrays.update({key: z[key] for key in ('state8','eef_quaternion_xyzw','wrist_depth_m','intrinsic',
                'camera_to_world','wrist_intrinsic','wrist_camera_to_world')})
        assert all(len(value) == row['frames'] for value in arrays.values())
        for value in arrays.values():
            value.setflags(write=False)
        return arrays


class SuiteDataset(SpatialDataset):
    """Same frame/chunk contract for a combined, complete Object/Goal/Long cache."""
    def __init__(self, rgbd_root, suite_definition):
        manifest=json.loads((Path(rgbd_root)/'manifest.json').read_text())
        expected=json.loads(Path(suite_definition).read_text())
        self.rows=sorted(manifest['episodes'],key=lambda row:row['episode_index'])
        assert manifest['complete'] and manifest['suite']==expected['suite']
        assert [row['episode_index'] for row in self.rows]==expected['episode_ids']
        assert len(self.rows)==expected['episodes']
        self.geometry={row['episode_index']:dict(geometry_npz=row['npz'],geometry_sha256=row['npz_sha256']) for row in self.rows}
        self.ends=np.cumsum([row['frames'] for row in self.rows])
        assert int(self.ends[-1])==manifest['total_frames']==expected['frames']
        self.cache=OrderedDict()
        self.max_cache_bytes=None
        self.cache_bytes=0
