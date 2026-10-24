"""Deterministic frame order and exact Spatial RGB-D/geometry pairing."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from rulervla.training.storage_io import retry_io
from rulervla.cues.motion_echo import MotionHistory


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


def action_chunk_indices(frame, frame_count, horizon=10):
    """Actions t..t+h-1; repeat only the same episode's final action."""
    if frame_count < 1 or horizon < 1:
        raise ValueError('Positive episode length and action horizon required')
    if not isinstance(frame, (int, np.integer)) or not 0 <= frame < frame_count:
        raise IndexError(frame)
    unpadded = int(frame) + np.arange(horizon, dtype=np.int64)
    return np.minimum(unpadded, frame_count-1), unpadded < frame_count


class SpatialDataset:
    def __init__(self, rgbd_root, geometry_root, max_cache_bytes=None, *, path_mappings=None):
        self.path_mappings = dict(path_mappings or {})
        self.manifest_hashes = {
            'rgbd': file_sha(Path(rgbd_root)/'regen_manifest.json'),
            'geometry': file_sha(Path(geometry_root)/'manifest.json'),
        }
        self.rows = sorted(json.loads((Path(rgbd_root)/'regen_manifest.json').read_text())['episodes'],
            key=lambda row: row['episode_index'])
        geometry = json.loads((Path(geometry_root)/'manifest.json').read_text())
        assert geometry['complete'] and geometry['total_frames'] == 52970
        self.geometry = {row['episode_index']: row for row in geometry['episodes']}
        assert [row['episode_index'] for row in self.rows] == list(range(1261,1693))
        assert set(self.geometry) == set(range(1261,1693))
        for row in self.rows:
            paired = self.geometry[row['episode_index']]
            if paired.get('frames', row['frames']) != row['frames']:
                raise ValueError('RGB-D and geometry episode lengths differ')
            if paired.get('rgbd_sha256', row['npz_sha256']) != row['npz_sha256']:
                raise ValueError('Geometry was generated from a different RGB-D source')
        self.ends = np.cumsum([row['frames'] for row in self.rows])
        assert int(self.ends[-1]) == 52970
        self.cache = OrderedDict()
        self.max_cache_bytes = max_cache_bytes
        self.cache_bytes = 0

    def resolve_source(self, path):
        """Map a byte-verified local cache without rewriting original manifests."""
        for source, destination in sorted(getattr(self, 'path_mappings', {}).items(),
                                           key=lambda item: len(item[0]), reverse=True):
            try:
                relative = Path(path).relative_to(source)
            except ValueError:
                continue
            return str(Path(destination)/relative)
        return str(path)

    def sample_identity(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        number = int(np.searchsorted(self.ends, index, side='right'))
        row = self.rows[number]
        frame = int(index-(self.ends[number-1] if number else 0))
        indices, valid = action_chunk_indices(frame, row['frames'])
        geo = self.geometry[row['episode_index']]
        return dict(global_index=int(index), episode_index=row['episode_index'], frame_index=frame,
                    episode_frames=row['frames'], task=row['task'],
                    action_frame_indices=indices.tolist(), action_valid_mask=valid.tolist(),
                    rgbd_source=row.get('npz'), rgbd_sha256=row.get('npz_sha256'),
                    geometry_source=geo.get('geometry_npz'), geometry_sha256=geo.get('geometry_sha256'),
                    manifest_sha256=getattr(self, 'manifest_hashes', {}))

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
            io_row, io_geometry = dict(row), dict(self.geometry[ep])
            if 'npz' in io_row:
                io_row['npz'] = self.resolve_source(io_row['npz'])
            if 'geometry_npz' in io_geometry:
                io_geometry['geometry_npz'] = self.resolve_source(io_geometry['geometry_npz'])
            loaded = retry_io(lambda: self._load_episode(io_row, io_geometry))
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
        indices, _ = action_chunk_indices(frame, row['frames'])
        sample.update(actions=arrays['actions'][indices].copy(), task=row['task'])
        return sample

    def get_history(self, index, *, history_steps=5):
        """Frame offsets are recorded environment control steps within one episode."""
        if not isinstance(history_steps, int) or history_steps < 1:
            raise ValueError("history_steps must be positive")
        if not 0 <= index < len(self):
            raise IndexError(index)
        number = int(np.searchsorted(self.ends, index, side='right'))
        row = self.rows[number]
        frame = int(index - (self.ends[number-1] if number else 0))
        episode = row['episode_index']
        self[index]  # Load and cache the verified episode without crossing its boundary.
        current = self.cache[episode]['ee_world_m'][frame].copy()
        past_frame = frame - history_steps
        if past_frame < 0:
            return MotionHistory(False, episode, frame, past_frame, current, None,
                                 'insufficient_history')
        past = self.cache[episode]['ee_world_m'][past_frame].copy()
        return MotionHistory(True, episode, frame, past_frame, current, past)

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
        if arrays['actions'].shape != (row['frames'], 7) or not np.isfinite(arrays['actions']).all():
            raise ValueError('LIBERO episode actions must be finite [frames,7]')
        if arrays['state8'].shape != (row['frames'], 8) or not np.isfinite(arrays['state8']).all():
            raise ValueError('Episode state8 must be finite [frames,8]')
        if not np.allclose(arrays['state8'][:, :3], arrays['ee_world_m'], atol=2e-6, rtol=0):
            raise ValueError('RGB-D EE and geometry state8 are not synchronized')
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
