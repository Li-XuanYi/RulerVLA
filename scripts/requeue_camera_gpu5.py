"""Move idle Camera queue to GPU5 and adopt its already-running ActionCoord child."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
from main_gpu_smoke import write_json
from small_training_contract import validate_run

root=Path(__file__).resolve().parents[1]
base=root/'outputs/small_ablation_v1'
out=base/'camera_gpu5_requeue_20261008'
out.mkdir(parents=True,exist_ok=False)
receipt=dict(state='waiting_for_input_gate',pid=os.getpid(),started_at_unix=time.time())
write_json(out/'status.json',receipt)
try:
    deadline=time.monotonic()+600
    gate=base/'gpu_batch_gate/status.json'
    while True:
        if gate.exists():
            d=json.loads(gate.read_text())
            if d['state']=='failed':raise RuntimeError('Input gate failed')
            if d['state']=='complete':break
        if time.monotonic()>deadline:raise TimeoutError('Input gate did not finish')
        time.sleep(2)
    matrix=json.loads((root/'experiments/small_ablation_v1/run_matrix.json').read_text())
    for run in matrix['runs']:
        if not run['conditional']:validate_run(root,run)
    workers=[]
    for index in (0,1):
        path=base/('training_queue_worker%d_status.json'%index)
        worker=json.loads(path.read_text());workers.append((path,worker))
        cmd=Path('/proc/%d/cmdline'%worker['pid']).read_bytes().replace(b'\0',b' ').decode()
        assert 'small_training_queue.py' in cmd and ('--worker-index %d'%index) in cmd
    assert workers[0][1]['child_pid'] is None and not workers[0][1]['formal_training_started']
    active=workers[1][1]
    assert active['current_run']=='S23_actioncoord_only_spatial_seed42'
    child=active['child_pid']
    cmd=Path('/proc/%d/cmdline'%child).read_bytes().replace(b'\0',b' ').decode()
    assert 'small_spatial_train.py' in cmd and active['current_run'] in cmd
    for path,worker in workers:
        write_json(out/path.name,worker)
        os.kill(worker['pid'],signal.SIGTERM) # Parent only; keep real training child alive.
    time.sleep(2)
    assert Path('/proc/%d'%child).exists()
    w0=workers[0][1]
    w0.update(state='transferred',stage='camera_queued_on_gpu5',assigned_runs=[],destination_gpu=5,
        transfer_record=str(out/'status.json'),child_pid=None,formal_training_started=False)
    write_json(workers[0][0],w0)
    with (base/'logs/worker1_gpu5_requeue_20261008.log').open('a') as stream:
        controller=subprocess.Popen([sys.executable,str(root/'scripts/small_training_queue.py'),
            '--worker-index','1','--include-camera','--adopt-running'],cwd=root,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(state='released',controller_pid=controller.pid,preserved_training_pid=child,gpu=5,
        methods=['ActionCoord-only','CameraXYZ-only','Ruler-only'],released_at_unix=time.time())
except BaseException as error:
    receipt.update(state='failed',error=repr(error));raise
finally:
    write_json(out/'status.json',receipt)
