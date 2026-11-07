# Time-LLM traffic forecasting pilot

This exploratory public-data experiment uses the official Time-LLM implementation with frozen pretrained GPT-2 and trained reprogramming/output layers. Full 12-layer GPT-2 improved flow prediction on this small panel. Speed prediction remained worse than persistence.

| Variant | Speed RMSE, km/h | Flow RMSE, vehicles/hour |
| --- | --- | --- |
| Persistence from latest available input | 1.824824 | 385.213542 |
| Same time on previous day | 2.074644 | 606.703871 |
| GPT-2 first 4 layers, LR 0.001, 200 steps | 2.195881 | 491.432454 |
| GPT-2 first 4 layers, LR 0.0001, 1000 steps | 2.313904 | 387.928243 |
| GPT-2 full 12 layers, LR 0.0001, 1000 steps | 2.062214 | 310.708435 |

Full GPT-2 reduced flow RMSE by 19.34% relative to persistence. Speed RMSE increased by 13.01%. These comparisons use the same observed evaluation cells for every method.

## Data and limitations

- One road, D7 I10 East, public synthetic June 2030 release. Four stations selected using training-period availability only.
- June 1–20 training, June 21–24 checkpoint validation, June 25–30 evaluation. The first model exposes this evaluation period for all later variants.
- 164 training windows, 36 validation windows and 50 evaluation windows across station/variable channels. Each window uses 96 history steps and predicts 12 steps, one hour at five-minute resolution.
- Targets must all be finite and score eligible. History must be at least 75% observed. Missing history uses causal forward filling for up to 12 steps, then a training-period channel mean.
- Previous-day availability restricts common metric support to 238 points per variable. Samples are correlated within stations and windows. This is a small exploratory result without confidence intervals.
- Speed is scaled by 100 and flow by 1000 during joint training. A single checkpoint is selected by validation scaled MSE every 50 steps.
- This forecasting diagnostic does not measure the competition's masked state reconstruction, physics scoring or OD score. The official score is unknown. No Kaggle submission was made.
- The full 12-layer choice follows inspection of the 4-layer results. Replication verifies numerical reproducibility, while road and month generalization still require further experiments.
- The original fully observed history protocol produced no usable windows. Its failure was retained locally before defining the causal-fill protocol.

## Reproduce on Windows

Use an existing CUDA PyTorch environment. The tested environment is recorded in `environment.txt`.

```powershell
python -m pip install transformers==4.45.2 accelerate==1.15.0 einops==0.8.2 pandas numpy pyarrow
$env:TFB_WORKSPACE_ROOT = 'D:/traffic-research'
$env:TFB_DATA_ROOT = 'D:/data/kaggle_public'
$env:TIME_LLM_UPSTREAM = 'D:/traffic-research/work/daytime/time_llm_upstream'
$env:HF_HOME = 'D:/traffic-research/work/daytime/time_llm_cache'
New-Item -ItemType Directory -Force 'D:/traffic-research/work/overnight' | Out-Null
Set-Content 'D:/traffic-research/work/overnight/research.lock' '00' -NoNewline
git clone https://github.com/KimMeen/Time-LLM.git $env:TIME_LLM_UPSTREAM
git -C $env:TIME_LLM_UPSTREAM checkout b13e881f86cd0475ce1b72c17110430663334955
git -C $env:TIME_LLM_UPSTREAM apply "$PWD/upstream_download.patch"
python run_time_llm_pilot.py --output-dir D:/traffic-research/outputs/daytime/time_llm_full12 --layers 12 --steps 1000 --learning-rate 0.0001
```

Run the setup in a new workspace. Preserve any existing research lock and coordinate GPU use before starting. The runner acquires an exclusive shared Windows byte lock before loading the model. Public safe-format GPT-2 weights are downloaded with implicit authentication disabled. The patch fixes first-load checkpoint resolution by explicitly requesting safe weights from the public model host.

Included files contain source, configuration/data/model hashes and aggregate station metrics. Raw observations, evaluation cell keys and the approximately 202 MB adapter checkpoint remain local. Model revision is pinned in acceptance metadata. Official upstream code and license are available at https://github.com/KimMeen/Time-LLM and the paper at https://arxiv.org/abs/2310.01728.

## I10 West replication and transfer failure

The same fixed 12-layer configuration was examined on I10 West after inspecting the East results. West has 192 training windows, 26 validation windows and 26 evaluation windows across four stations. Metrics use 116 common observed points per variable. This is another exposed small-sample development comparison.

| Method on West | Speed RMSE, km/h | Flow RMSE, vehicles/hour |
| --- | --- | --- |
| Persistence | 22.791492 | 671.583000 |
| Previous day | 7.997691 | 561.777435 |
| Frozen East adapter, zero West training | 34.571556 | 699.604203 |
| West-trained adapter, unchanged 1000-step configuration | 20.917006 | 478.244596 |

Direct East-to-West adapter transfer failed to beat either baseline on both variables. West fitting reduced flow RMSE by 28.79% against persistence and 14.87% against previous day. West speed remained much worse than the previous-day baseline. These results support further study of road-specific flow forecasting, with speed and cross-road transfer failures retained. They do not establish competition imputation gains.

Use `--road D7_I10_W` to train the West model with the same options above. To evaluate a frozen East adapter, add `--evaluate-adapter D:/traffic-research/outputs/daytime/time_llm_full12/adapter.pt` and select a fresh output directory. The target road's training observations are used for availability selection and missing-input mean statistics, while the source adapter is frozen.

`--variable speed_kmh` or `--variable flow_vph` restricts the shared channel-independent model to one variable for subsequent exploratory studies. The results in the tables use `--variable both`.
