# Long-trip share under a weak-prior budget

Fix baseline count predictions and origin/destination totals. Compare no prior restriction, a convex L1 budget equal to the baseline distance from the weak prior, and the previously tested exact-distance subset with fixed signs. The average interval above the weak-prior median span contracts from 84–85 percentage points to about 21 with the budget. The fixed-sign subset spans about 19.6 points. All 20 intervals satisfy the expected nesting.

The budget permits smaller L1 distance than the baseline. These are feasible ranges under the stated assumptions, without hidden OD validation.

    python task4/audit_pairing_prior_budget_ranges.py --release-root /path/to/release --output /path/to/output
