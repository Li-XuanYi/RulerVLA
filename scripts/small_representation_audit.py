"""S26 表示核验: real training frames, no policy training or rollout."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from main_gpu_smoke import write_json
from rulervla.adapters.main_cues import MainCueAdapter
from rulervla.adapters.small_ablation import METHODS, SmallAblationAdapter
from rulervla.config import load_config
from rulervla.geometry.action_coord import canonical_geometry
from rulervla.training.main_data import SpatialDataset, file_sha


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    out = ROOT / 'outputs/small_ablation_v1/S26_表示核验'
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = out / 'status.json'
    if path.exists() and json.loads(path.read_text())['state'] == 'complete':
        print('Existing completed S26 reused; no rerun', flush=True)
        return
    status = dict(experiment_id='S26',display_name='S26_表示核验',state='running',pid=os.getpid(),
        started_at_unix=time.time(),frames_checked=0,training_updates=0,policy_inferences=0,rollouts=0)
    write_json(path,status)
    try:
        config = load_config(ROOT/'configs/rulervla_main.yaml')
        rgbd=Path('/home/Dataset2/yqt/openpi-aimbot/spatialcue_depth_source/regenerated_rgbd_v02')
        geometry=Path('/home/Dataset2/yqt/rulervla/main_comparison_v1/data_preparation/spatial_geometry')
        dataset=SpatialDataset(rgbd,geometry)
        adapters={name:SmallAblationAdapter(config,name) for name in METHODS}
        references={name:MainCueAdapter(config,name) for name in ('Baseline','RulerVLA-main')}
        first={}
        for number,row in enumerate(dataset.rows):first.setdefault(row['task'],number)
        assert len(first)==10
        records=[];preview=None
        for task,number in sorted(first.items()):
            row=dataset.rows[number]
            start=int(dataset.ends[number-1]) if number else 0
            for frame in np.linspace(0,row['frames']-1,5,dtype=int):
                sample=dataset[start+int(frame)]
                maps=canonical_geometry(sample,config['geometry'])
                transform=np.asarray(sample['camera_to_world'],np.float64)
                rotation,translation=transform[:3,:3],transform[:3,3]
                assert np.max(np.abs(rotation.T@rotation-np.eye(3)))<1e-6
                camera=maps['camera_xyz'].astype(np.float64)
                world=camera@rotation.T+translation
                ee_camera=rotation.T@(sample['ee_world_m']-translation)
                camera_ee=camera-ee_camera
                expected=world-sample['ee_world_m']
                valid=maps['valid']
                assert valid.any()
                err=float(np.max(np.abs((camera_ee@rotation.T-expected)[valid])))
                actual_err=float(np.max(np.abs((maps['actioncoord']-expected)[valid])))
                assert err<1e-6 and actual_err<1e-6
                prepared={name:adapter.prepare(sample) for name,adapter in adapters.items()}
                a,b,c,d=[prepared[name] for name in ('Baseline','Ruler-only','ActionCoord-only','RulerVLA-main')]
                shared=('observation/wrist_image','observation/state','actions','observation/wrist_cue')
                for key in shared:
                    assert all(np.array_equal(a[key],v[key]) for v in prepared.values())
                assert all(v['prompt']==a['prompt'] for v in prepared.values())
                assert np.array_equal(a['observation/image'],c['observation/image'])
                assert np.array_equal(b['observation/image'],d['observation/image'])
                assert np.array_equal(c['observation/spatial_image'],d['observation/spatial_image'])
                assert not a['observation/spatial_image'].any() and not b['observation/spatial_image'].any()
                for name,adapter in references.items():
                    reference=adapter.prepare(sample)
                    for key in reference:
                        assert np.array_equal(reference[key],prepared[name][key]), (name,key)
                values={'camera_xyz':camera,'camera_ee':camera_ee,'action_xyz':world,'actioncoord':expected}
                clip={name:float(np.mean(np.abs(value[valid])>3.25)) for name,value in values.items()}
                record=dict(task=task,episode_index=row['episode_index'],frame=int(frame),
                    basis_equivalence_error_m=err,existing_actioncoord_error_m=actual_err,
                    component_clip_ratio=clip,valid_pixels=int(valid.sum()),
                    input_sha256={name:{key:sha(value) for key,value in data.items() if isinstance(value,np.ndarray)}
                        for name,data in prepared.items()})
                records.append(record)
                if preview is None:preview=prepared
                status['frames_checked']=len(records);write_json(path,status)
            dataset.cache.clear()
        assert len(records)==50
        summary=dict(experiment_id='S26',display_name='S26_表示核验',tasks=10,frames=50,
            input_contract_passed=True,existing_main_endpoints_bitwise_equal=True,
            basis_equivalence_max_error_m=max(r['basis_equivalence_error_m'] for r in records),
            existing_actioncoord_max_error_m=max(r['existing_actioncoord_error_m'] for r in records),
            maximum_component_clip_ratio={name:max(r['component_clip_ratio'][name] for r in records) for name in values},
            encoding_gate_passed=all(max(r['component_clip_ratio'][name] for r in records)<=.001 for name in values),
            rgbd_manifest_sha256=file_sha(rgbd/'regen_manifest.json'),
            geometry_manifest_sha256=file_sha(geometry/'manifest.json'),
            observations='Engineering/geometry evidence only; no learned performance or causal benefit result')
        write_json(out/'records.json',records);write_json(out/'summary.json',summary)
        from PIL import Image,ImageDraw
        names=list(METHODS);canvas=Image.new('RGB',(224*len(names),480),'white');draw=ImageDraw.Draw(canvas)
        for i,name in enumerate(names):
            draw.text((224*i+4,4),name,fill='black')
            canvas.paste(Image.fromarray(preview[name]['observation/image']),(224*i,25))
            spatial=np.clip((preview[name]['observation/spatial_image']+1)*127.5,0,255).astype(np.uint8)
            canvas.paste(Image.fromarray(spatial),(224*i,250))
        canvas.save(out/'contact_sheet.png')
        status.update(state='complete',finished_at_unix=time.time(),input_contract_passed=True,
            encoding_gate_passed=summary['encoding_gate_passed'])
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


if __name__=='__main__':main()
