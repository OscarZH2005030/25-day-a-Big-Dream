"""Assemble reviewed code/aggregate results/model weights; exclude raw competition rows."""
from pathlib import Path
import shutil,json,hashlib,sys,platform,importlib.metadata
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/daytime/publish-repo'
DEST=REPO/'experiments/trafficflowbench_local_20261102'
OUT=ROOT/'outputs/daytime'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(value):
 if isinstance(value,dict):return {clean(k):clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,str):
  for prefix,replacement in [(str(ROOT),'<research-workspace>'),(ROOT.as_posix(),'<research-workspace>'),('C:/Users/Lenovo/Downloads/2026-ieee-big-data-traffic-flow-bench/kaggle_public','<release-root>'),('C:\\Users\\Lenovo\\Downloads\\2026-ieee-big-data-traffic-flow-bench\\kaggle_public','<release-root>')]:value=value.replace(prefix,replacement)
 return value

def copy(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True)
 if source.suffix=='.json':dest.write_text(json.dumps(clean(json.loads(source.read_text(encoding='utf-8-sig'))),indent=2,ensure_ascii=False),encoding='utf8')
 else:shutil.copy2(source,dest)

def main():
 for name in ['multimonth','severe_mixture']:
  source=OUT/(name+'_candidate');target=DEST/'task3/packages'/name
  expected=json.loads((source/'manifest.json').read_text())['files']
  for rel,digest in expected.items():assert sha(source/rel)==digest,(name,rel)
  for p in source.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json':copy(p,target/p.relative_to(source))
  files={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file() and p.name!='manifest.json'}
  (target/'manifest.json').write_text(json.dumps(dict(files=files,source_manifest_sha256=sha(source/'manifest.json'),metadata_note='Local absolute source paths redacted; inference source/model bytes unchanged.'),indent=2),encoding='utf8')
 names=['fresh_masks','augmentation','augmentation_original_check','missingness_selection','temporal_regularization','accumulation_fit','refit49','multimonth_fit','severe_masks','severe_augmentation','mild_cached_transfer','severe_seed_ensemble','temporal_gap_gate','task4_information_audit','task4_count_evidence_audit','task4_demand_bounds','task4_metric_invariant','task4_covariance_audit','task4_prior_structure','task4_prior_structure_transfer','task4_span_shrink_stress','task4_observable_gate_stress','task4_candidate_sensitivity','task4_gaussian_posterior_pilot','task4_posterior_longrun']
 names.extend(['task4_smallnet_posterior_stress_retry','task4_background_mean_audit'])
 names.extend(['task4_span_variance_audit','task4_variance_controls_audit','task4_weighted_variance_stress','task4_od_only_direction_stress'])
 names.extend(['task4_crossmonth_pair_audit','task4_pair_marginal_controls'])
 names.extend(['task4_crossmonth_shared_noise_stress_qr'])
 names.extend(['task4_count_residual_alignment','task4_exact_pair_audit'])
 names.extend(['task4_count_source_pair_stress'])
 names.extend(['task4_od_cycle_stress'])
 names.extend(['task4_source_scale_control'])
 names.extend(['task4_cycle_budget_check','task4_budget_dual_bounds'])
 names.extend(['task4_count_bias_gate'])
 names.extend(['task4_bias_cancellation_stress'])
 names.extend(['task4_count_shape_audit','task4_public_range_shape_stress'])
 names.extend(['task4_robust_prior_forecast'])
 names.extend(['task4_robust_prior_scale_audit','task4_sparse_cutoff_stress'])
 names.extend(['task4_pair_penalty_audit','task4_pair_standardization_stress'])
 names.extend(['task4_pair_partial_information','task4_half_pair_stress'])
 names.extend(['task4_local_pair_transfer','task4_half_pair_surrogate_audit'])
 names.extend(['task4_pair_agreement_audit','task4_source_clip_stress'])
 names.extend(['task4_source_clip_extended'])
 names.extend(['task4_source_clip_replaced'])
 names.extend(['task4_count_invariant_drift'])
 names.extend(['task4_extra_measurement_audit','task4_fixed_input_reference_swap'])
 names.extend(['task4_local_span_null_stencil'])
 names.extend(['task4_full_path_incidence'])
 names.extend(['task4_hypothetical_time_bins'])
 names.extend(['task4_hypothetical_time_alias'])
 names.extend(['task4_hypothetical_time_conditioning'])
 names.extend(['task4_hypothetical_time_monotone'])
 names.extend(['task4_released_marginal_information'])
 names.extend(['task4_net_repair'])
 names.extend(['task4_minimum_invisible_repair'])
 names.extend(['task4_minimum_repair_face'])
 names.extend(['task4_l1_repair_stress'])
 names.extend(['task4_invariant_invisible_shrink'])
 names.extend(['task4_invariant_shrink_tiebreak'])
 names.extend(['task4_equal_movement_ties'])
 names.extend(['task4_budgeted_source_direction'])
 names.extend(['task4_budgeted_source_path_cap'])
 names.extend(['task4_path_cap_extended'])
 names.extend(['task4_source_agreement_stress','task4_source_agreement_corruption','task4_source_agreement_common_corruption','task4_source_error_correlation','task4_source_error_sign_symmetry','task4_source_error_symmetric_capacity','task4_source_error_quarter_capacity','task4_source_error_amplitude_panel_check','task4_direct_od_measurement_design','task4_direct_od_measurement_transfer','task4_direct_od_measurement_pooling','task4_direct_od_measurement_counts_only','task4_direct_od_count_geometry','task4_direct_od_off_cell_design','task4_direct_od_hybrid_design','task4_direct_od_heteroscedastic_design','task4_direct_od_sqrt_variance_design','task4_direct_od_variance_transfer','task4_direct_od_variance_interpolation','task4_direct_od_unselected_selection','task4_direct_od_common_support','task4_direct_od_noisy_support','task4_direct_od_fixed_noise','task4_direct_od_noise_adaptation','task4_direct_od_origin_covariance','task4_direct_od_destination_covariance','task4_direct_od_dual_covariance'])
 excluded={'per_link.csv','daily_PM_aggregates.csv'}
 for name in names:
  folder=OUT/name
  for p in folder.iterdir():
   if p.is_file() and p.suffix in ['.json','.csv'] and p.name not in excluded:copy(p,DEST/'results'/name/p.name)
  if (folder/'failure_history').exists():
   for p in (folder/'failure_history').iterdir():
    if p.is_file() and p.suffix in ['.json','.csv']:copy(p,DEST/'results'/name/'failure_history'/p.name)
 for name in ['multimonth','severe_mixture']:copy(OUT/('scored_'+name+'_parts/acceptance.json'),DEST/'results/state_export_checks'/(name+'.json'))
 if (OUT/'runtime_save_repair.json').exists():copy(OUT/'runtime_save_repair.json',DEST/'results/runtime_save_repair.json')
 if (OUT/'od_cycle_numerical_verification.json').exists():copy(OUT/'od_cycle_numerical_verification.json',DEST/'results/od_cycle_numerical_verification.json')
 for p in (ROOT/'work/daytime').glob('*.py'):copy(p,DEST/'archive/daytime'/p.name)
 for n in ['state_experiment.py','context_features.py','residual_state.py']:copy(ROOT/'work/overnight'/n,DEST/'archive/overnight'/n)
 copy(ROOT/'work/review-repo/experiments/task4/robust_odme_20261027/outputs/task4_path_flow.csv',DEST/'task4/frozen_reference_path_flow.csv')
 (DEST/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','pyarrow','lightgbm']},baseline_source_repository='https://github.com/chengweiliu669-pixel/25-day-a-Big-Dream-submission',baseline_source_commit='e254d55acdb94261d03d2ea10a79c983f1e15d55'),indent=2),encoding='utf8')
 files={p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='SHA256SUMS.txt'}
 (DEST/'SHA256SUMS.txt').write_text(''.join(v+'  '+k+'\n' for k,v in sorted(files.items())),encoding='utf8')
 print(json.dumps(dict(files=len(files),bytes=sum((DEST/p).stat().st_size for p in files),models=sum('/models/' in p for p in files)),indent=2))
if __name__=='__main__':main()
