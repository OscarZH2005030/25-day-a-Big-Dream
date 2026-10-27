# Task4 v10: four components evaluated jointly

All tables report S_od, S_link, S_dev, S_attr and weighted S_ODME together. Synthetic scores are sensitivity experiments, not competition scores. Actual path truth is unavailable; only actual S_link can be calculated. Unknown components remain blank in actual_four_components.csv.

480 synthetic cases: ten panels, 16 error families, three shapes. Nine candidates compare raw/scaled KL blends against official ridge. New errors include origin-destination correlated prior noise, localized 40% undercount, count rounding, and a nearly correct prior. Selection requires each calibration component and total to be nondecreasing, real count CV mean improvement >=0.5%, worst selection-seed ratio <=1.01, and worst calibration-family total loss <=0.005. Calibration conditions apply to means, not every individual case.

Selection count seeds: 20261201/02. Audit seeds: 20261203/04. These reuse the same actual released train counts, not new independent path observations. Heavy and sparse artificial evaluations do not feed back into policy choice. Shape truth is still derived from released weak priors with perturbations, so this does not remove artificial-data bias. No new anonymous real five-minute source is claimed in this round.

Frozen policy:
{
  "D12_I405_N": "official",
  "D12_I405_S": "official",
  "D12_I5_N": "official",
  "D12_I5_S": "official",
  "D7_I10_E": "official",
  "D7_I10_W": "official",
  "D7_I210_E": "official",
  "D7_I210_W": "official",
  "D7_I405_N": "official",
  "D7_I405_S": "scaled_0.1"
}

Four components (synthetic):

        phase       method     S_od   S_link    S_dev   S_attr  synthetic_S_ODME
  calibration joint_policy 0.500402 0.997364 0.654961 0.904863          0.708496
  calibration     official 0.498240 0.997363 0.654881 0.904470          0.707452
  calibration  previous_v9 0.504455 0.997368 0.654187 0.905374          0.710281
   evaluation joint_policy 0.490484 0.997214 0.662038 0.899907          0.704313
   evaluation     official 0.488663 0.997212 0.662204 0.899558          0.703466
   evaluation  previous_v9 0.494196 0.997217 0.662033 0.900376          0.706054
sparse_stress joint_policy 0.437919 0.996581 0.648318 0.885990          0.676355
sparse_stress     official 0.435081 0.996579 0.646841 0.885471          0.674778
sparse_stress  previous_v9 0.440899 0.996587 0.648098 0.886823          0.677789

Real grouped held-out count NMAE (lower better):

    seed       method     NMAE
20261201 joint_policy 0.035012
20261201     official 0.035126
20261202 joint_policy 0.037261
20261202     official 0.037447
20261203 joint_policy 0.038697
20261203     official 0.038845
20261204 joint_policy 0.035828
20261204     official 0.035958

The original submission is preserved. No Kaggle submission. V10 is an experiment; passing calibration guards does not establish an actual hidden S_dev or total improvement. See all_candidates_four_components.csv for all rejected methods and joint_selection_checks.csv for every guard.
