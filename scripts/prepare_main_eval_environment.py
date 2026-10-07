"""GPU rendering preflight: four suites, 40 tasks, frozen 50-state plans; no policy rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from prepare_main_spatial_geometry import capture_frame, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id',type=int,default=5)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    v03=Path('/home/yqt/XuanyiLi/actioncoord/work/v03_sync_rgbd_20261003T164806')
    sys.path[:0]=[str(v03/'repo/actioncoord_experiments/src'),str(v03/'repo/third_party/libero')]
    from actioncoord.data.state_adapter import state8_from_observation
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out=root/'outputs/main_comparison/eval_environment_preflight';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',pid=os.getpid(),gpu=args.device_id,completed_tasks=0,expected_tasks=40,
        policy_rollouts=0,environment_step_calls=0,started_at_unix=time.time())
    status_path=out/'status.json';write_json(status_path,status)
    plans=[]
    try:
        for suite_name in ('libero_spatial','libero_object','libero_goal','libero_10'):
            suite=benchmark.get_benchmark_dict()[suite_name]();assert suite.n_tasks==10
            for task_index in range(suite.n_tasks):
                task=suite.get_task(task_index);states=np.asarray(suite.get_task_init_states(task_index))
                assert len(states)>=50 and states.ndim==2 and np.isfinite(states[:50]).all()
                task_dir=out/suite_name/('task_%02d'%task_index);task_dir.mkdir(parents=True,exist_ok=True)
                state_path=task_dir/'init_states_50.npy'
                if not state_path.exists():
                    with state_path.open('xb') as stream:np.save(stream,states[:50],allow_pickle=False)
                else:np.testing.assert_array_equal(np.load(state_path),states[:50])
                frozen=[dict(scene_index=i,environment_seed=i,initial_state_sha256=hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest())
                    for i,state in enumerate(states[:50])]
                receipt=task_dir/'geometry.json'
                if not receipt.exists():
                    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
                    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,
                        camera_depths=True,render_gpu_device_id=args.device_id)
                    try:
                        env.seed(0);env.reset();env.set_init_state(states[0])
                        exact=np.asarray(env.sim.get_state().flatten(),np.float64)
                        _,sample=capture_frame(env,exact,state8_from_observation)
                        assert sample['agentview_rgb'].shape==(256,256,3) and sample['wrist_rgb'].shape==(256,256,3)
                        assert sample['state8'].shape==(8,)
                        np.testing.assert_allclose(sample['state8'][:3],sample['ee_world_m'],atol=0,rtol=0)
                        for key in ('camera_to_world','wrist_camera_to_world'):
                            transform=sample[key];np.testing.assert_allclose(transform@np.linalg.inv(transform),np.eye(4),atol=1e-10)
                        from PIL import Image
                        for key in ('agentview_rgb','wrist_rgb'):Image.fromarray(sample[key]).save(task_dir/(key+'.png'))
                        with (task_dir/'observation.npz').open('xb') as stream:np.savez_compressed(stream,**sample)
                        write_json(receipt,dict(task=task.language,suite=suite_name,task_index=task_index,
                            correct_policy_rgb_orientation='raw simulator RGB/depth rot180, using existing capture_frame',
                            agent_depth_range_m=[float(sample['agentview_depth_m'].min()),float(sample['agentview_depth_m'].max())],
                            wrist_depth_range_m=[float(sample['wrist_depth_m'].min()),float(sample['wrist_depth_m'].max())],
                            sim_state_sha256=sample['sim_state_sha256'],geometry_passed=True,policy_rollouts=0))
                    finally:env.close()
                assert json.loads(receipt.read_text())['geometry_passed']
                plans.append(dict(suite=suite_name,task_index=task_index,instruction=task.language,
                    scenes=frozen,init_states_file=str(state_path),init_states_file_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest()))
                status.update(completed_tasks=len(plans),current_suite=suite_name,current_task=task_index);write_json(status_path,status)
        write_json(out/'paired_scene_plan.json',dict(tasks=plans,suites=4,tasks_per_suite=10,rollouts_per_task=50,
            conditions_per_method_seed=2000,training_seeds=[42,43,44],methods=['Baseline','AimBot','AxisGuide-adapted','RulerVLA-main'],
            all_methods_share_initial_states_and_environment_seeds=True,policy_rollouts_executed=0))
        status.update(state='complete',finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(status_path,status)


if __name__=='__main__':main()
