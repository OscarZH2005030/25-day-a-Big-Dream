Dell holdout prediction supplement (2026-10-26, America/Los_Angeles)

Contents: six directions x four methods = 24 original Parquet prediction files.
Methods: T1, A_dynamic_fixed, Current_fusion, LightGBM.
Holdout: 2031-01-05 through 2031-02-28 (55 days).

Directory mapping to the original Dell project runs:
d7_i405/ -> runs/transformer_training_dev_20261019_183939_881805/
d12_i5/ -> runs/d12_i5_dynamic_dev_20261020_124730_964790/
d7_i210/ -> runs/remaining_dynamic_dev_20261021_141813_814491/batches/01_D7_I210/
Each group has holdout/<panel>/<method>_predictions.parquet.
Each group also includes original COMPLETE.json, artifact_manifest.json, and
holdout_metrics_by_panel_regime.csv. D7_I210 E/W are complete; no rerun is needed.

I405 LightGBM provenance:
The later training run reused LightGBM but did not export it into holdout/.
The two LightGBM files are copied byte-for-byte from its exact protected source:
runs/transformer_priorities_pilot_01/<panel>/LightGBM_predictions.parquet.
Both hashes were checked against the later run's source_manifest.json.
All other 22 prediction files are copied directly from the requested holdout folders.
No model training or new prediction inference was performed for this export.

Schema and units:
panel, timestamp, station_id, link_id, mask_regime, speed_kmh, flow_vph.
flow_vph is TOTAL link flow in vehicles/hour, NOT flow per lane. Divide by the
matching station lane count only when a diagnostic calls for per-lane flow.
Station/link identifiers are strings. Preserve them and join using all five key
columns. Target order is identical across the four methods for each direction.
These files contain the scored missing targets, not complete observed road grids.
Physics diagnostics that need full grids must also load the released masked
observations and matching network/ramp metadata from the official dataset.
No ground-truth label columns are included in these prediction files.

Integrity:
export_manifest.json records original sources, row counts, byte sizes, and SHA256.
The source COMPLETE/artifact manifests describe full original runs, not only this
selected subset; files not requested for this supplement remain outside the bundle.
Existing eight-direction summaries are preserved; this supplement does not rewrite
or relabel their scores. Use the I210 metrics to complete a separate ten-direction
aggregation with the same regime -> direction -> family averaging protocol.
