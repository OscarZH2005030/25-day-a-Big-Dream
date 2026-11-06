# Task4 prior-ranked direct OD probes

Select the 256 largest OD entries in the mean normalized train and
validation priors on I10 East. This fixed selection uses no candidate
differences. Keep the same broad feasibility constraints as the adaptive
experiment. Both runs find a pair agreeing numerically at the probes
and differing by 0.740% to 0.742% of demand in full OD L1. The witnesses
can vary, and the test uses no actual OD measurements.

    python task4/audit_pairing_od_probe_prior_top.py --release-root /path/to/kaggle_public --output /path/to/audit-output
