# Actual-origin augmentation: fixed October candidate, local proxy Overall 0.829591955

This is a retained research point estimate, not a new official score or an automatic production replacement. Primary `dense_100` was ranked first on October before November/February. Secondary `dense_50` is reported without changing that order.

| Fixed candidate | Main 184 ongoing: IoU | FP / FN | Overall with frozen onset | Old 210 ongoing: IoU | Old Overall |
|---|---:|---:|---:|---:|---:|
| Production smooth default | .880264843 | 595 / 470 | .828148579 | .801366893 | .769732821 |
| Primary dense100 | .883151597 | 628 / 428 | **.829591955** | .804370722 | .771234736 |
| Secondary dense50 | .883038656 | 638 / 408 | .829535485 | .807036083 | .772567416 |

Primary reduces missed cells but increases false positives. Its main ongoing gain vs default is .002886754, day-cluster 95% CI [-.001293212,.007278829], crossing zero. Repeated public February development and multiple comparisons limit the claim; the previous descriptive .828896973 candidate and every production artifact remain intact. No February-based candidate/seed/ensemble selection occurred.

## What changed

The same 274 inputs and 100-tree residual capacity were trained on genuine extra origins at -10/-5/+5/+10 minutes, recomputing actual public masked labels, never shifting labels or treating unknowns as negative. Final data contains 32,087 windows / 2,691,898 selected node rows, reusing immutable June–September data and preparing October–November separately. Adjacent origins are correlated; independent date count did not grow. Priors are updated at day end from original roster targets only. Every original X/y/m row was reproduced exactly.

June–September development fit, October selection, and November confirmation preceded June–November final fit. November primary IoU .935912971 vs .928220482, gain .007692489 CI [.004792421,.010720683] on 673 positive windows. R127 and its basis have seen June–November; development is not a completely blind evaluation.

The 144 new native models are in `models/`. The fixed chain is sigmoid(900-tree base + original100-tree specialist + alpha*new100), average the original three seeds, then the original equal log-odds mixture with R127. Alpha1 is primary, alpha.5 secondary. Existing parents are hash-verified dependencies in sibling `best_overall_082639_jaccard_20261031` and `best_overall_082723_horizon_20261031`, with original onset/R127 dependencies unchanged. Do not substitute a 900-only parent.

## Independent acceptance and inference

Fresh public raw reconstruction reproduced all 184+210 ongoing windows, complete X274/R127/y/m arrays, both probabilities within 1e-7, and identical IoU/FP/FN. Sixteen road/population future y/m/speed_y mutations left inputs and predictions unchanged. Scope is 394 ongoing windows; unchanged onset is retained by hash, not claimed newly recomputed here. Reports and full negative/tradeoff evidence are in `reports/` and `acceptance/`.

Keep this directory beside its two sibling parent packages. Use `code/predict_npz.py --root <this-directory> --features <X274,panel,parent NPZ> --candidate dense_100 --output <result.npz>`. `parent` means R127 probability, not labels. `code/hybrid_dense.py` provides a raw-observation Hybrid adapter; preload it before installing the existing R127 archive guard, attach the unchanged parent, then call probability/decode. Both candidates use their independently January-selected smooth horizon thresholds in `profile.json`; changing the candidate or thresholds by February is prohibited.

Raw data, large predictions, caches and credentials are not included. This is public-observation proxy evidence, not official S_queue or competition total. No Kaggle upload/submission; production defaults are unchanged. Original training/audit sources are archived, with local provenance paths retained as historical records rather than claimed portable training commands.
