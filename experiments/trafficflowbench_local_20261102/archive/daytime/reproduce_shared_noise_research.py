"""Verify repaired numerical fits and reproduce all fixed shared-noise cases."""
import common as d
import subprocess
import sys
import msvcrt
import traceback
import pandas as pd
import numpy as np


def summarize(folder):
    df=pd.read_csv(folder/'all_cases.csv')
    baseline=df[df.method=='baseline'][['case','solver_success','synthetic_S_ODME','synthetic_S_od']]
    paired=df.merge(baseline,on='case',suffixes=('','_baseline'))
    paired=paired[paired.solver_success & paired.solver_success_baseline].copy()
    paired['score_delta']=paired.synthetic_S_ODME-paired.synthetic_S_ODME_baseline
    paired['OD_delta']=paired.synthetic_S_od-paired.synthetic_S_od_baseline
    paired['score_harmed']=paired.score_delta < -1e-12
    paired['OD_harmed']=paired.OD_delta < -1e-12
    paired['positive_prior_prediction_gain']=paired.pair_prior_prediction_MSE_reduction > 0
    metrics=dict(paired_cases=('case','count'),mean_score_delta=('score_delta','mean'),worst_score_delta=('score_delta','min'),score_harmed_cases=('score_harmed','sum'),mean_OD_delta=('OD_delta','mean'),OD_harmed_cases=('OD_harmed','sum'),mean_prior_prediction_MSE_reduction=('pair_prior_prediction_MSE_reduction','mean'))
    paired.groupby(['noise','drift','method']).agg(**metrics).to_csv(folder/'paired_by_noise_drift.csv')
    paired.groupby(['positive_prior_prediction_gain','method']).agg(**metrics).to_csv(folder/'predictability_vs_OD_risk.csv')
    geometry=paired[paired.method=='geometry005'][['case','synthetic_S_ODME','synthetic_S_od']]
    pair=paired[paired.method=='pair005'].merge(geometry,on='case',suffixes=('','_geometry'))
    pair['score_delta_vs_geometry005']=pair.synthetic_S_ODME-pair.synthetic_S_ODME_geometry
    pair['OD_delta_vs_geometry005']=pair.synthetic_S_od-pair.synthetic_S_od_geometry
    pair.groupby(['noise','drift'])[['score_delta_vs_geometry005','OD_delta_vs_geometry005']].mean().to_csv(folder/'pair_vs_geometry005.csv')
    d.save(folder/'paired_analysis_scope.json',dict(analysis='All same-case eligible comparisons retained. Positive prior-prediction gain is a descriptive slice using supplied-prior outcomes, not a selected deployment gate or fresh heldout claim.',source_sha256=d.sha(__file__),hidden_OD_score=None))


def main():
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        folder=d.OUT/'task4_crossmonth_shared_noise_stress_qr'
        d.state(status='running',stage='shared_noise_independent_SVD_verification',next='Check same-case augmented SVD fits, all five actions and original-prior scoring before ranking repaired results. No Kaggle upload.')
        subprocess.run([sys.executable,'-X','utf8',str(d.WORK/'verify_shared_noise_qr.py')],check=True)
        summarize(folder)
        out=d.WORK/'publication_smoke/shared_noise_stress'
        d.state(status='running',stage='shared_noise_full_portable_reproduction',output=str(out),next='Rerun all 960 synthetic triplets with corrected QR and compare every action/latent/model artifact; no hidden OD validation.')
        source=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/stress_crossmonth_noise.py'
        subprocess.run([sys.executable,'-X','utf8',str(source),'--release-root',str(d.DATA),'--output',str(out)],check=True)
        errors={}
        for name in ['all_cases.csv','by_assumption.csv','paired_risk.csv']:
            a,b=pd.read_csv(folder/name),pd.read_csv(out/'stress'/name)
            assert a.shape==b.shape and a.columns.equals(b.columns),name
            numeric=a.select_dtypes(include='number').columns
            for col in a.columns.difference(numeric):assert a[col].equals(b[col]),(name,col)
            assert np.array_equal(a[numeric].isna(),b[numeric].isna())
            error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
            assert error < 1e-12,(name,error)
            errors[name]=error
        d.save(folder/'portable_reproduction.json',dict(case_count=960,method_rows=4800,all_artifact_hashes_identical=True,all_solver_availability_identical=True,maximum_CSV_metric_differences=errors,runner_sha256=d.sha(__file__),scope='Reproducibility of fixed artificial risks only; no official OD improvement.'))
        d.state(status='stage_complete',stage='shared_noise_full_portable_reproduction_complete',next='Review all shared-noise/drift harms and positive-prior-predictability counterexamples, publish verified results, then continue robust pairing inference research. Official baseline preserved, no Kaggle upload.')
        print('ALL SHARED-NOISE CASES REPRODUCED',errors,flush=True)
    except Exception as error:
        d.state(status='failed',stage='shared_noise_verification_or_reproduction',error=repr(error),next='Keep original and QR attempts, inspect numerical/portable mismatch without changing seed, strengths or cases.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
