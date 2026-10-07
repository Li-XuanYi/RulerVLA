"""S27 旧案交互: reuse B2/B3/B4/B5, with a strict cross-run pairing gate."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from main_gpu_smoke import write_json
from rulervla.training.main_data import file_sha


def main():
    out=ROOT/'outputs/small_ablation_v1/S27_旧案交互';out.mkdir(parents=True,exist_ok=True)
    status=dict(experiment_id='S27',state='running',pid=os.getpid(),started_at_unix=time.time(),
        training_updates=0,policy_inferences=0,new_rollouts=0)
    path=out/'status.json';write_json(path,status)
    try:
        base=Path('/home/Dataset2/yqt/actioncoord-v02-20260928')
        old=Path('/home/yqt/XuanyiLi/actioncoord/repo/third_party/AimBot/Xuanyi/outputs/v02_actioncoord/evaluation_retry3/pilot')
        raw={'B2':old/'B2','B3':old/'B3','B4':base/'evaluation_B4B5_20261005/pilot/B4',
             'B5':base/'evaluation_B4B5_20261005/pilot/B5'}
        run_names=['pilot_B2_20260928_v1','pilot_B3_20260928_v3','pilot_B4_fixed_20261005','pilot_B5_fixed_20261005']
        meta=[json.loads((base/name/'run_metadata.json').read_text()) for name in run_names]
        config_keys=('num_train_steps','seed','batch_size','ema_decay','lr_schedule','optimizer','model','freeze_filter')
        controls=[{key:d['config'][key] for key in config_keys} for d in meta]
        problems=[]
        if not all(c==controls[0] for c in controls):problems.append('Training controls differ')
        for key in ('cache_sha256','encoding_sha256','base_checkpoint_manifest_sha256'):
            if len({d[key] for d in meta})!=1:problems.append('Training provenance differs: '+key)
        if not all(d['status']=='completed' for d in meta):problems.append('Incomplete pilot training')
        records={};sources={}
        for arm,folder in raw.items():
            values={}
            for p in sorted(folder.glob('scene_*.json')):
                d=json.loads(p.read_text());key=(d['scene_index'],d['shift_cm'])
                assert key not in values
                assert p.with_suffix('.npz').exists()
                if d.get('error') is not None:problems.append(arm+' technical rollout error')
                values[key]=d;sources[str(p)]=file_sha(p)
            assert len(values)==40
            records[arm]=values
        keys=sorted(records['B2'])
        assert all(set(v)==set(keys) for v in records.values())
        pairing_fields=('initial_state_sha256','settled_state_sha256','camera_state_sha256','lighting_state_sha256',
            'scene_plan_sha256','simulator_seed','sampling_seed_protocol','instruction','requested_shift_xy_m',
            'target_initial_xyz_m','ee_initial_xyz_m')
        mismatches=[]
        for key in keys:
            group=[records[arm][key] for arm in raw]
            for field in pairing_fields:
                if any(field not in d for d in group) or not all(d.get(field)==group[0].get(field) for d in group):
                    mismatches.append(dict(scene=key[0],shift_cm=key[1],field=field))
            if group[0].get('first_agentview_sha256')!=group[1].get('first_agentview_sha256'):
                mismatches.append(dict(scene=key[0],shift_cm=key[1],field='B2/B3 first RGB'))
            if group[2].get('first_agentview_sha256')!=group[3].get('first_agentview_sha256'):
                mismatches.append(dict(scene=key[0],shift_cm=key[1],field='B4/B5 first ruler RGB'))
        if mismatches:problems.append('Cross-run pairing failed')
        summary=dict(experiment_id='S27',training_controls=controls[0],source_json_sha256=sources,
            successes={arm:sum(bool(d['success']) for d in v.values()) for arm,v in records.items()},
            conditions_per_arm=40,pairing_passed=not problems,problems=problems,pairing_mismatches=mismatches,
            scope='Legacy single-seed 5k pilot with 10k warmup; not the new protocol or the Ruler x ActionCoord factorial')
        if not problems:
            scenes=sorted({key[0] for key in keys})
            assert len(scenes)==10
            matrix=np.asarray([[np.mean([records[arm][k]['success'] for k in keys if k[0]==scene])
                for arm in ('B2','B3','B4','B5')] for scene in scenes],float)
            # Columns: Camera, Action, Camera+Ruler, Action+Ruler.
            effects=np.column_stack((matrix[:,2]-matrix[:,0],matrix[:,3]-matrix[:,1],
                (matrix[:,3]-matrix[:,1])-(matrix[:,2]-matrix[:,0])))
            rng=np.random.default_rng(42);boot=effects[rng.integers(0,10,size=(10000,10))].mean(axis=1)
            summary['effects']={name:dict(mean_pp=float(effects[:,i].mean()*100),
                scene_cluster_ci95_pp=(np.quantile(boot[:,i],[.025,.975])*100).tolist())
                for i,name in enumerate(('ruler_in_camera','ruler_in_action','ruler_by_representation_interaction'))}
            summary['interpretation']='Descriptive legacy evidence only; scene CI does not cover training-seed uncertainty'
        else:
            summary['interpretation']='No cross-run causal/interaction effect reported; preserve per-arm historical counts only'
        write_json(out/'summary.json',summary)
        status.update(state='complete',pairing_passed=summary['pairing_passed'],finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


if __name__=='__main__':main()
