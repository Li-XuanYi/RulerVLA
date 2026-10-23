import copy

from rulervla.training.metric_resources import preflight


def configuration():
    return dict(data=dict(rgbd_root='/rgbd',geometry_root='/geo',rgbd_manifest_sha256='hash',
                          geometry_manifest_sha256='hash'),norm_stats_dir='/norm',norm_stats_sha256='hash',
                initial_checkpoint='/params',checkpoint_receipt='/receipt',checkpoint_receipt_sha256='hash',
                resources=dict(gpu_indices=[7],min_free_mib=45000,min_cpu_available_gib=0))


def healthy_file(path,parse_json=False):
    value=dict(path=str(path),sha256='hash',bytes=100)
    if parse_json:
        value['json']=dict(episodes=[{}]*432,complete=True,total_frames=52970,
            rgbd_manifest_sha256='hash',geometry_manifest_sha256='hash',
            files=[dict(path='_METADATA',sha256='hash')])
    return value


def test_missing_data_blocks_even_with_enough_gpu_memory():
    def unavailable(path,parse_json=False):
        return dict(path=str(path),error='Host is down') if parse_json else healthy_file(path)
    result=preflight(configuration(),file_probe=unavailable,
        gpu_query=lambda:[dict(index=7,free_mib=48000,total_mib=49140,utilization=0)])
    assert not result['launch_allowed'] and not result['gpu_job_started']
    assert any('Host is down' in message for message in result['blockers'])


def test_inadequate_gpu_memory_blocks_valid_data():
    result=preflight(configuration(),file_probe=healthy_file,
        gpu_query=lambda:[dict(index=7,free_mib=21000,total_mib=49140,utilization=70)])
    assert not result['launch_allowed']
    assert any('45000' in message for message in result['blockers'])


def test_launch_only_after_data_identity_and_memory_pass():
    config=configuration()
    gpus=lambda:[dict(index=7,free_mib=48000,total_mib=49140,utilization=0)]
    assert preflight(config,file_probe=healthy_file,gpu_query=gpus)['launch_allowed']
    config['data']['geometry_manifest_sha256']='different'
    assert not preflight(config,file_probe=healthy_file,gpu_query=gpus)['launch_allowed']
