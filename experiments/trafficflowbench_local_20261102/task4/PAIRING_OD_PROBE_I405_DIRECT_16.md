# I405 North OD designs with 392 hypothetical measurements

Compare 392 conditional direct OD cells against 256 direct cells and 136 nonempty 16-by-16 contiguous origin-destination group totals. Both designs use the same feasible prior views and exact base equalities. In both local and isolated runs, the minimum achievable maximum disagreement is 0.222735 flow units for direct measurements and 0.198708 for the mixed design. Both exact-match endpoint problems are infeasible. Solver primal-status text varies between runs while the model status agrees.

The direct design separates the specified prior views by a larger maximum row discrepancy under these hypothetical measurements. This comparison concerns one road, with equal nominal observation counts and unknown measurement cost and error. It does not establish hidden OD accuracy.

```sh
python task4/audit_pairing_od_probe_i405_direct_16.py --release-root /path/to/kaggle_public --output /path/to/direct
python task4/audit_pairing_od_probe_i405_mixed_16.py --release-root /path/to/kaggle_public --output /path/to/mixed
```
