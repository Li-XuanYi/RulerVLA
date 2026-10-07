"""Generate the S/L index and non-executable supplemental training specifications."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    entries=[]
    states={'S03':'legacy_diagnostic','S04':'legacy_complete','S08':'legacy_complete','S09':'legacy_complete',
        'S10':'failed','S11':'partial','S22':'blocked_capacity_and_launch_gates','S23':'blocked_capacity_and_launch_gates',
        'S24':'conditional','S25':'blocked_checkpoint','L06':'running','L07':'planned'}
    aliases={'S01':['E1_orientation_debug'],'S02':['E1_checkpoint_audit','E1_smoke'],
        'S03':['libero_plus_camera_ood_20260929'],'S04':['libero_ood_camera_standard_20260929'],
        'S05':['M1'],'S06':['M2'],'S07':['SpatialCue_v02_preflight','depth_replay'],
        'S08':['V02_B0_B1_B2_B3'],'S09':['B4_B5_5k'],'S10':['B4_B5_lora_30k_20261005'],
        'S11':['V03_R2_R3'],'S12':['V03_LOCATION_SPLIT'],'S13':['V05_GT_anchor_smoke'],
        **{'S%02d'%(13+i):['G%d'%i] for i in range(1,8)},'S21':['MAIN_GPU_RELEASE','MAIN_EVAL_PREFLIGHT'],
        'S22':['P0-A'],'S23':['P0-B'],'S24':['P1'],'S25':['MAIN_INFERENCE_INTERVENTION'],
        'S26':['small_representation_audit'],'S27':['legacy_B2_B3_B4_B5_interaction'],
        'L01':['E1_A0_A1_A2_training'],'L02':['E1_formal_567_oriented'],
        'L03':['A3_pixel_ruler_training'],'L04':['A3_standard_ID'],'L05':['A2_A3_depth_formal'],
        'L06':['Main_Comparison_v1'],'L07':['Main_LIBERO_Plus_v1']}
    for line in (ROOT/'docs/EXPERIMENT_CATALOG.md').read_text(encoding='utf-8').splitlines():
        match=re.match(r'\| \*\*([SL]\d{2})_([^*]+)\*\* \|',line)
        if not match:continue
        identifier,label=match.groups()
        assert 1<=len(label)<=5 and all('\u4e00'<=c<='\u9fff' for c in label)
        cells=[c.strip() for c in line.strip('|').split('|')]
        entries.append(dict(experiment_id=identifier,display_name=identifier+'_'+label,
            classification='small' if identifier.startswith('S') else 'large',
            legacy_ids=aliases[identifier],status=states.get(identifier,'completed'),
            purpose=cells[1],details=cells[2:],
            priority='P0' if identifier in ('S22','S23') else 'P1' if identifier=='S24' else None))
    assert len(entries)==34 and len({e['experiment_id'] for e in entries})==34
    write(ROOT/'experiments/experiment_registry.json',dict(schema='rulervla_experiment_registry_v1',
        audited_date='2026-10-07',naming_rule='S/L + two digits + underscore + <=5 Chinese characters',
        historical_paths_preserved=True,entries=entries))
    base=json.loads((ROOT/'experiments/main_comparison_v1/runs/libero_spatial_baseline_seed42.json').read_text())
    shared=deepcopy(base['shared_training'])
    shared.update(micro_batch=1,gradient_accumulation=32,data_parallel_replicas=1,
        memory_configuration_status='common validated single GPU, optimizer/EMA/master CPU',training_seeds=[42])
    plan=[]
    for identifier,method,slug,updates,conditional in [
        ('S22','CameraXYZ-only','camera_xyz_only',10000,False),
        ('S23','Ruler-only','ruler_only',30000,False),
        ('S23','ActionCoord-only','actioncoord_only',30000,False),
        ('S24','CameraEE-only','camera_ee_only',10000,True),
        ('S24','ActionXYZ-only','action_xyz_only',10000,True)]:
        config=deepcopy(shared);config.update(optimizer_updates=updates,
            checkpoint_updates=list(range(5000,updates+1,5000)),formal_checkpoint_index=updates-1)
        plan.append(dict(run_id=identifier+'_'+slug+'_spatial_seed42',experiment_id=identifier,
            suite='libero_spatial',method=method,seed=42,shared_training=config,
            checkpoint_root='/home/Dataset2/yqt/rulervla/small_ablation_v1/checkpoints/'+identifier+'/'+slug+'_seed42',
            execution_ready=False,launch_gate='conditional S22 positive primary result' if conditional else 'capacity + per-arm smoke + evaluation contract',
            trainer_adapter='rulervla.adapters.small_ablation.SmallAblationAdapter',
            trainer_integration_status='pending isolated entry; do not pass directly to main trainer',
            conditional=conditional))
    write(ROOT/'experiments/small_ablation_v1/run_matrix.json',dict(schema='small_ablation_run_specs_v1',
        executable=False,models_added_P0=3,optimizer_updates_added_P0=70000,models_added_conditional_P1=2,runs=plan))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write(ROOT/'experiments/small_ablation_v1/protocol.json',dict(schema='small_ablation_protocol_v1',
        frozen_date='2026-10-07',protocol_document_sha256=sha(ROOT/'docs/SMALL_ABLATION_PROTOCOL.md'),
        run_matrix_sha256=sha(ROOT/'experiments/small_ablation_v1/run_matrix.json'),
        primary_suite='libero_spatial',training_seed=42,effective_batch=32,micro_batch=1,accumulation=32,
        input_slots=4,encoding_radius_m=3.25,warmup_updates=1000,cosine_decay_updates=30000,
        primary_eval=dict(tasks=10,official_initial_states_per_task=20,cases_per_arm=200,
            source='outputs/main_comparison/eval_environment_preflight/paired_scene_plan.json',
            simulator_and_policy_sampling_contract='must exactly match frozen main evaluation before release'),
        inference_intervention=dict(experiment_id='S25',tasks=10,states_per_task=3,conditions_per_state=7,
            inference_calls=210,rollouts=0,required_checkpoint='L06 Spatial RulerVLA-main seed42 step29999'),
        reuse=dict(S22_ActionCoord='S23 ActionCoord-only checkpoint9999; identical first10k optimizer prefix',
            S23_Baseline='L06 libero_spatial_baseline_seed42 checkpoint29999',
            S23_RulerVLA='L06 libero_spatial_rulervla_main_seed42 checkpoint29999',
            S24='S22 matched10k CameraXYZ/ActionCoord snapshots'),
        case_manifests={name:sha(ROOT/'experiments/small_ablation_v1'/name)
            for name in ('primary_cases_200.json','intervention_cases_30.json')},
        new_training_started=False,old_protocol_results_preserved=True))
    print('Registry: 27 S + 7 L; P0: 3 new models; P1: 2 conditional models')


if __name__=='__main__':main()
