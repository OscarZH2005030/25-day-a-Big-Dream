# Task4 component scaling on two pooled weak priors

Reuse the coefficients fit from train-to-validation on other corridors.
Apply them without refitting to the train-only prediction or the equal
train-validation mean. Both use the same private counts-and-margins
projection and exposed weak-prior target.

On the pooled predictor, separate scaling gives mean relative L1 of 0.61412
and squared error of 0.44582. Shared scaling gives 0.63725 and 0.48281.
The half-average gives 0.68934 and 0.54884. These are weak-prior proxy results.
Hidden OD performance remains unverified.

From the experiment directory:

    python task4/audit_pairing_component_pool_transfer.py --release-root /path/to/kaggle_public --output /path/to/audit-output
