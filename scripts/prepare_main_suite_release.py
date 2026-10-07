"""CPU-only shared training normalization and reused cue audit for remaining suites."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from main_gpu_smoke import write_json


def main():
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root/'src'),str(root/'outputs/main_comparison/runtime_v3/src')]
    from openpi.shared import normalize
    from rulervla.training.main_data import file_sha
    from rulervla.training.storage_io import retry_io
    out=root/'outputs/main_comparison/suite_release'
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',pid=os.getpid(),started_at_unix=time.time(),completed_suites=[],
        current_suite=None,formal_training_started=False,policy_rollouts=0)
    status_path=out/'status.json';write_json(status_path,status)
    try:
        for suite in ('libero_object','libero_goal','libero_10'):
            status.update(current_suite=suite,stage='training_only_normalization');write_json(status_path,status)
            data=Path('/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/suite_rgbd')/suite
            manifest_path=data/'manifest.json';manifest=json.loads(manifest_path.read_text())
            expected=json.loads((root/'experiments/main_comparison_v1/suite_manifests'/(suite+'.json')).read_text())
            rows=sorted(manifest['episodes'],key=lambda row:row['episode_index'])
            assert manifest['complete'] and manifest['total_frames']==expected['frames']
            assert [row['episode_index'] for row in rows]==expected['episode_ids']
            destination=out/suite;destination.mkdir(exist_ok=True)
            stats_dir=destination/'norm';source_path=stats_dir/'source.json'
            provenance=dict(suite=suite,episodes=expected['episodes'],frames=expected['frames'],
                manifest_sha256=file_sha(manifest_path),suite_definition_sha256=file_sha(root/'experiments/main_comparison_v1/suite_manifests'/(suite+'.json')),
                policy='all training frames; 10-step action horizon, repeat-last within each episode; shared by all four methods',
                state_components=8,action_components=7,action_horizon=10,evaluation_frames=0)
            if (stats_dir/'norm_stats.json').exists():
                assert json.loads(source_path.read_text())==provenance
            else:
                state_stats,action_stats=normalize.RunningStats(),normalize.RunningStats()
                total=0
                for row in rows:
                    def read():
                        with np.load(row['npz'],allow_pickle=False) as z:
                            return z['state8'],z['actions']
                    state,actions=retry_io(read)
                    assert state.shape==(row['frames'],8) and actions.shape==(row['frames'],7)
                    assert np.isfinite(state).all() and np.isfinite(actions).all()
                    state_stats.update(state)
                    indices=np.minimum(np.arange(len(actions))[:,None]+np.arange(10),len(actions)-1)
                    action_stats.update(actions[indices]);total+=len(actions)
                    status['processed_frames']=total;write_json(status_path,status)
                assert total==expected['frames']
                normalize.save(stats_dir,{'state':state_stats.get_statistics(),'actions':action_stats.get_statistics()})
                write_json(source_path,provenance)
            status['stage']='four_method_cue_audit';write_json(status_path,status)
            command=[sys.executable,str(root/'scripts/audit_main_cues.py'),'--suite-root',str(data),
                '--aimbot-source','/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
                '--axisguide-source','/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py',
                '--out',str(destination/'cue_audit')]
            audit_path=destination/'cue_audit/audit.json'
            if not audit_path.exists():
                with (destination/'cue_audit.log').open('a') as stream:
                    subprocess.run(command,cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
            audit=json.loads(audit_path.read_text());assert audit['passed'] and audit['tasks']==10
            write_json(destination/'release.json',dict(**provenance,norm_stats_sha256=file_sha(stats_dir/'norm_stats.json'),
                cue_audit_sha256=file_sha(audit_path),numerical_preflight_passed=True,visual_review_pending=True,
                formal_training_started=False,policy_rollouts=0))
            status['completed_suites'].append(suite);write_json(status_path,status)
        status.update(state='complete',stage='complete',current_suite=None,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(status_path,status)


if __name__=='__main__':main()
