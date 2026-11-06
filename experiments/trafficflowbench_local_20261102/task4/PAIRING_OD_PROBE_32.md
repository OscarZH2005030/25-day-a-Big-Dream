# Task4 blind spots with 32 direct OD probes

Select the 32 OD cells with largest absolute differences in a minimum-gap
witness pair, then minimize observed differences over the wider feasible
set. Each solution keeps the 1% total movement cap, 125% maximum-cell cap
and near-optimal primary objective. Both share the full span histogram.

Both runs find pairs agreeing at all 32 selected cells. Across runs the
full OD L1 differences are 0.817% to 0.854% on D12 I405 N and 0.834% to
0.839% on D7 I10 E. These are feasible witnesses for the selected designs.

    python task4/audit_pairing_od_probe_32.py --release-root /path/to/kaggle_public --output /path/to/audit-output
