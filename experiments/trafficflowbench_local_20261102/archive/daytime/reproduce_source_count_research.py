"""Independent integer count identity and full three-study reproduction."""
import common as d
import task4_span_shrink_stress as base
import subprocess
import sys
import msvcrt
import traceback
import json
import pandas as pd
import numpy as np


def exact_identity():
    results=[]
    for panel in d.night.predictor.load_routes():
        matrices=[]
        for split in ['train','validation','private']:
            paths,A,_,_,_=base.load(panel,split);matrices.append(A)
        origin=paths.origin_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy();dest=paths.destination_zone.str.extract(r'^Z(\d+)$',expand=False).astype(int).to_numpy()
        first,last=int(origin.min()),int(dest.max());first_col={int(dest[i]):i for i in np.flatnonzero(origin==first)};last_col={int(origin[i]):i for i in np.flatnonzero(dest==last)}
        for A in matrices:
            # Pure integer identity: A(o,d)=A(o,last)+A(first,d)-A(first,last).
            rebuilt=A[:,[last_col[int(o)] for o in origin]]+A[:,[first_col[int(v)] for v in dest]]-A[:,first_col[last]][:,None]
            assert np.array_equal(A,rebuilt),panel
        results.append(dict(panel=panel,path_count=len(paths),all_three_input_row_sets_exact_integer_identity=True,maximum_count_incidence_error=0))
    d.save(d.OUT/'task4_exact_pair_audit/independent_integer_identity.json',dict(identity='For every observed count row and OD(o,d): A(o,d)=A(o,last)+A(first,d)-A(first,last). Thus that row is an exact linear combination of origin/destination marginals.',panels=results,source_sha256=d.sha(__file__),scope='Release-specific exact incidence identity, no hidden OD labels or floating rank tolerance required. Counts do not add pairing information conditional on all margins.'))


def main():
    lock=(d.ROOT/'work/overnight/research.lock').open('r+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        d.state(status='running',stage='source_count_independent_verification',next='Check source observations, all six scores and exact integer count/margin identity before full reproduction; no Kaggle upload.')
        subprocess.run([sys.executable,'-X','utf8',str(d.WORK/'verify_count_source_artifacts.py')],check=True)
        exact_identity()
        out=d.WORK/'publication_smoke/source_count_studies';source=d.WORK/'publish-repo/experiments/trafficflowbench_local_20261102/task4/audit_source_counts.py'
        d.state(status='running',stage='source_count_full_portable_reproduction',output=str(out),next='Reproduce 480 same-fold source controls, count-mismatch alignment and exact interaction-prior controls; no hidden OD accuracy claim.')
        subprocess.run([sys.executable,'-X','utf8',str(source),'--release-root',str(d.DATA),'--output',str(out),'--with-stress'],check=True)
        for name in ['task4_count_residual_alignment','task4_exact_pair_audit','task4_count_source_pair_stress']:
            errors={}
            for original in (d.OUT/name).glob('*.csv'):
                a,b=pd.read_csv(original),pd.read_csv(out/name/original.name);assert a.shape==b.shape and a.columns.equals(b.columns),(name,original.name)
                numeric=a.select_dtypes(include='number').columns
                for col in a.columns.difference(numeric):assert a[col].equals(b[col]),(name,original.name,col)
                assert np.array_equal(a[numeric].isna(),b[numeric].isna())
                error=float(np.nanmax(np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())))
                assert error<1e-12,(name,original.name,error)
                errors[original.name]=error
            if name=='task4_exact_pair_audit':
                normalize=lambda values:{key.replace('\\','/').split('task4_exact_pair_models/',1)[1]:value for key,value in values.items()}
                assert normalize(json.loads((d.OUT/name/'model_sha256.json').read_text()))==normalize(json.loads((out/name/'model_sha256.json').read_text()))
            d.save(d.OUT/name/'portable_reproduction.json',dict(maximum_CSV_metric_differences=errors,all_rows_availability_and_artifact_hashes_identical=True,runner_sha256=d.sha(__file__),scope='Reproduction of public development diagnostics/artificial risks, not real hidden OD validation.'))
        d.state(status='stage_complete',stage='source_count_three_studies_reproduced',next='Publish verified conditional-count-rank identity, exact pairing-prior evidence and negative source-count controls with all limitations; then continue OD-only inference tests, no Kaggle upload.')
        print('SOURCE COUNT STUDIES REPRODUCED',flush=True)
    except Exception as error:
        d.state(status='failed',stage='source_count_verification_or_reproduction',error=repr(error),next='Preserve all case results and diagnostics; fix verification without changing folds or methods.');traceback.print_exc();raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':main()
