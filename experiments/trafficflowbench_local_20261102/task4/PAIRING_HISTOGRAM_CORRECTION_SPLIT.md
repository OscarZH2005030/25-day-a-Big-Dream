# Task4 split of the transferred histogram-mask correction

Split the fixed mask correction into two orthogonal directions after counts
and margins projection. One direction preserves the span histogram before
norm matching. Whole-prediction norm matching can itself change the histogram.

On the shared exposed weak-prior target, adding the histogram-preserving
direction before normalization improves both losses on 8/10 roads. The
histogram-changing direction improves both on 0/10. The corresponding mean
relative L1 and squared error for the former are 0.68280 and 0.54019, compared
with 0.68934 and 0.54884 for uniform. Hidden OD accuracy is unverified.

From the experiment directory:

    python task4/audit_pairing_histogram_correction_split.py --release-root /path/to/kaggle_public --output /path/to/audit-output
