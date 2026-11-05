# Task4 scaling control

Apply the same source-derived scale either to the updated predictor, the
unchanged baseline, or one of its orthogonal components. The unchanged
baseline with scaling alone changes relative L1 from 0.68934 to 0.68319
and relative squared error from 0.54884 to 0.54086. The full update gives
0.68280 and 0.54019. Scaling the histogram-determining component alone
gives 0.68270 and 0.54184. Results use exposed weak-prior targets and do not
validate gains against hidden OD.

From the experiment directory:

    python task4/audit_pairing_histogram_scaling_control.py --release-root /path/to/kaggle_public --output /path/to/audit-output
