import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from rulervla.training.main_data import SuiteDataset,file_sha


def test_combined_suite_preserves_frame_pairing_episode_boundary_and_input_isolation():
    root=Path(__file__).parents[1]/'outputs'/('suite_data_test_'+uuid4().hex)
    root.mkdir(parents=True)
    npz=root/'episode.npz'
    arrays={key:np.zeros((3,4,4,3),np.uint8) for key in ('agentview_rgb','wrist_rgb')}
    arrays.update({key:np.ones((3,4,4),np.float32) for key in ('agentview_depth_m','wrist_depth_m')})
    arrays.update(state8=np.arange(24,dtype=np.float32).reshape(3,8),actions=np.arange(21,dtype=np.float32).reshape(3,7),
        ee_world_m=np.zeros((3,3)),eef_quaternion_xyzw=np.zeros((3,4)))
    for key in ('intrinsic','wrist_intrinsic'):arrays[key]=np.repeat(np.eye(3)[None],3,axis=0)
    for key in ('camera_to_world','wrist_camera_to_world'):arrays[key]=np.repeat(np.eye(4)[None],3,axis=0)
    np.savez_compressed(npz,**arrays)
    row=dict(episode_index=42,frames=3,npz=str(npz),npz_sha256=file_sha(npz),task='fixed instruction')
    (root/'manifest.json').write_text(json.dumps(dict(complete=True,suite='libero_object',total_frames=3,episodes=[row])))
    definition=root/'definition.json';definition.write_text(json.dumps(dict(suite='libero_object',episode_ids=[42],episodes=1,frames=3)))
    dataset=SuiteDataset(root,definition)
    sample=dataset[1]
    np.testing.assert_array_equal(sample['state8'],arrays['state8'][1])
    np.testing.assert_array_equal(sample['actions'][0],arrays['actions'][1])
    np.testing.assert_array_equal(sample['actions'][1:],np.repeat(arrays['actions'][2:3],9,axis=0))
    sample['agentview_rgb'][0,0,0]=255
    assert dataset[1]['agentview_rgb'][0,0,0]==0
    assert not dataset.cache[42]['agentview_rgb'].flags.writeable
