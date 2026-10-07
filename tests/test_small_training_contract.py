"""Frozen fairness and resource admission failures must block supplements."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('small_contract',ROOT/'scripts/small_training_contract.py')
contract = importlib.util.module_from_spec(spec); spec.loader.exec_module(contract)


class FrozenContractTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT/'outputs/test_small_contract'/uuid4().hex
        self.root.mkdir(parents=True)
        self.matrix = json.loads((ROOT/'experiments/small_ablation_v1/run_matrix.json').read_text(encoding='utf-8'))
        self.reference = json.loads((ROOT/'experiments/main_comparison_v1/runs/libero_spatial_baseline_seed42.json').read_text(encoding='utf-8'))
        self.write('experiments/small_ablation_v1/run_matrix.json',self.matrix)
        self.write('experiments/main_comparison_v1/runs/libero_spatial_baseline_seed42.json',self.reference)
        self.write('outputs/small_ablation_v1/S26_表示核验/status.json',
            dict(state='complete',input_contract_passed=True,encoding_gate_passed=True))
        self.hash_patch = patch.object(contract,'sha',return_value='b5e3345ee3d2a4698348ea8f8988031f2d003fdcd36c4716a7f7abc2d0834820')
        self.hash_patch.start()

    def tearDown(self):
        self.hash_patch.stop()
        assert self.root.resolve().is_relative_to((ROOT/'outputs/test_small_contract').resolve())
        shutil.rmtree(self.root)

    def write(self,relative,value):
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value),encoding='utf-8')

    def test_all_three_p0_specs_share_main_settings(self):
        for run in self.matrix['runs']:
            if not run['conditional']:contract.validate_run(self.root,run,require_gpu_gate=False)

    def test_run_mutation_rejected_even_if_effective_batch_stays_32(self):
        run=deepcopy(self.matrix['runs'][0]);run['shared_training']['micro_batch']=2
        run['shared_training']['gradient_accumulation']=16
        with self.assertRaises(AssertionError):contract.validate_run(self.root,run,require_gpu_gate=False)

    def test_changed_shared_scheduler_rejected_even_if_frozen_spec_is_rewritten(self):
        matrix=deepcopy(self.matrix);matrix['runs'][0]['shared_training']['scheduler']['decay_updates']=10000
        self.write('experiments/small_ablation_v1/run_matrix.json',matrix)
        with self.assertRaises(AssertionError):contract.validate_run(self.root,matrix['runs'][0],require_gpu_gate=False)

    def test_p1_cannot_be_launched_without_outcome_gate(self):
        run=next(r for r in self.matrix['runs'] if r['conditional'])
        with self.assertRaises(AssertionError):contract.validate_run(self.root,run,require_gpu_gate=False)

    def test_pending_gpu_gate_blocks_training(self):
        self.write('outputs/small_ablation_v1/gpu_batch_gate/status.json',
            dict(state='running',gpu_batch_gate_passed=False))
        with self.assertRaises(AssertionError):contract.validate_run(self.root,self.matrix['runs'][0])


if __name__=='__main__':unittest.main()
