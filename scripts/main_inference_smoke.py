"""Four-slot policy API smoke on official base weights; zero evaluation rollouts."""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from main_gpu_smoke import write_json


def main():
    root=Path(__file__).resolve().parents[1];runtime=root/'outputs/main_comparison/runtime_v3'
    sys.path[:0]=[str(root/'src'),str(runtime/'src'),str(runtime/'packages/openpi-client/src')]
    import jax
    import jax.numpy as jnp
    from openpi.models import model
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.policy import Policy
    from openpi.policies.libero_policy import LiberoOutputs
    from openpi import transforms
    from openpi.shared import normalize
    from rulervla.adapters.main_cues import MainCueAdapter
    from rulervla.adapters.main_comparison import MainComparisonInputs
    from rulervla.config import load_config
    out=root/'outputs/main_comparison/inference_api_smoke';out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.smoke.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(state='running',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],started_at_unix=time.time(),
        checkpoint='/home/Dataset2/yqt/openpi/checkpoints/Official/pi05_base',checkpoint_role='base weights for engineering API smoke only',
        methods_passed=[],formal_evaluation=False,policy_rollouts=0)
    path=out/'status.json';write_json(path,status)
    try:
        assert jax.device_count()==1
        cfg=Pi0Config(pi05=True,action_horizon=10,discrete_state_input=False)
        status['stage']='loading_base_weights';write_json(path,status)
        loaded=cfg.load(model.restore_params(Path(status['checkpoint'])/'params',dtype=jnp.bfloat16))
        norm=normalize.load(root/'outputs/main_comparison/gpu_smoke_20261006/shared_spatial_norm')
        inputs=[MainComparisonInputs(model_type=cfg.model_type),transforms.Normalize(norm,use_quantiles=True,strict=True),
            transforms.TokenizePrompt(PaligemmaTokenizer(cfg.max_token_len),discrete_state_input=False),transforms.PadStatesAndActions(32)]
        policy=Policy(loaded,transforms=inputs,output_transforms=[transforms.Unnormalize(norm,use_quantiles=True),LiberoOutputs()],
            sample_kwargs={'num_steps':10})
        case=root/'outputs/main_comparison/eval_environment_preflight/libero_spatial/task_00'
        with np.load(case/'observation.npz',allow_pickle=False) as z:sample={key:z[key].copy() for key in z.files}
        sample.update(task=json.loads((case/'geometry.json').read_text())['task'],actions=np.zeros((10,7),np.float32))
        for method in ('Baseline','AimBot','AxisGuide-adapted','RulerVLA-main'):
            adapter=MainCueAdapter(load_config(root/'configs/rulervla_main.yaml'),method,
                aimbot_source='/home/yqt/XuanyiLi/openpi0-aimbot/third_party/AimBot/src',
                axisguide_source='/home/yqt/XuanyiLi/AxisGuide-code/src/lerobot/datasets/visual_cue_utils.py')
            status.update(stage='inference',current_method=method);write_json(path,status)
            result=policy.infer(adapter.prepare(sample));action=np.asarray(result['actions'])
            assert action.shape==(10,7) and np.isfinite(action).all()
            np.save(out/(method.replace('-','_')+'_actions.npy'),action,allow_pickle=False)
            status['methods_passed'].append(method);write_json(path,status)
        status.update(state='complete',stage='complete',api_contract_passed=True,finished_at_unix=time.time())
    except BaseException as error:
        status.update(state='failed',error=repr(error),finished_at_unix=time.time());raise
    finally:write_json(path,status)


if __name__=='__main__':main()
