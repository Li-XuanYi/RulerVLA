"""After successful Camera 10k, preserve interrupted evidence and restart GPU5 small queue."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
from main_gpu_smoke import write_json

root=Path(__file__).resolve().parents[1];base=root/'outputs/small_ablation_v1'
path=base/'gpu5_after_camera_recovery_status.json'
status=dict(state='running',stage='waiting_for_camera_10k',pid=os.getpid(),formal_training_started=False)
write_json(path,status)
try:
    while True:
        current=json.loads((base/'camera_gpu5_now_status.json').read_text())
        if current['state']=='failed':raise RuntimeError('Camera failed; preserve state and stop recovery queue')
        if current['state']=='complete':break
        time.sleep(20)
    checkpoint=Path(current['final_checkpoint']);assert checkpoint.name=='9999' and (checkpoint/'V03_COPY_COMPLETE.json').exists()
    run='S23_actioncoord_only_spatial_seed42'
    record=base/'train_records'/run
    old=json.loads((record/'status.json').read_text());assert old['state']=='interrupted'
    assert not Path('/proc/%d'%old['pid']).exists()
    output=Path(old['checkpoint_root'])
    assert output.resolve()==Path('/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints/S23/actioncoord_only_seed42').resolve()
    assert not any(p.name.isdigit() for p in output.iterdir()), 'Saved checkpoint requires real resume'
    evidence=base/'gpu5_interruption_20261008'
    record.rename(evidence/'actioncoord_step1372_train_records')
    shutil.copy2(base/'logs'/(run+'.log'),evidence/(run+'_interrupted.log'))
    archive=Path('/home/Dataset2/yqt/rulervla/small_ablation_v1/interrupted_attempts/actioncoord_step1372_metadata')
    archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists()
    output.rename(archive)
    old_queue=base/'training_queue_worker1_status.json'
    old_queue.rename(evidence/'gpu5_controller_failed_status.json')
    with (base/'logs/gpu5_after_camera_recovery.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,str(root/'scripts/small_training_queue.py'),
            '--worker-index','1','--include-camera'],cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    status.update(state='released',stage='actioncoord_fresh_base_then_ruler',queue_pid=child.pid,
        camera_final_checkpoint=str(checkpoint),interrupted_updates=old['optimizer_updates'],
        actioncoord_restarts_from_official_base=True,checkpoint_resume_available=False)
except BaseException as error:
    status.update(state='failed',error=repr(error));raise
finally:write_json(path,status)
