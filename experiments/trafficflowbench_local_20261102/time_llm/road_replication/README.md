# Time-LLM across ten roads

The fixed 12-layer, 1,000-step joint model beat both persistence and previous-day RMSE on 3 of 10 roads for flow and 2 of 10 for speed. It beat persistence on 4 of 10 for each variable. The comparison includes 2,325 common points per variable, drawn from the four stations selected by training availability on each road.

The table in `by_road.csv` includes the eight added joint fits and the two earlier I10 fits. It also contains the exploratory East speed-only run. Its speed RMSE was 1.520 km/h against 1.825 for persistence and 2.075 for the previous day on 238 common points. The East and West 384-step runs failed the frozen validation-window history requirement before fitting. Their failures and log hashes are in `audit.json`.

The queue fixed road, variable, history, optimizer settings and seed before its results were available. The initial I10 trials had already informed that choice. All evaluations are on exposed public synthetic June forecasting data. The comparisons use a single training seed and four stations per road. Official Task3 reconstruction and Task4 OD scores remain unknown.

Reproduce each row using the Windows setup in the parent README. The registry records the arguments for the added runs. For each row, set `--road`, `--variable`, and `--seq-len` to its registered values, retain 12 layers, 1,000 steps, and learning rate 0.0001, and use a separate output directory. For 384 steps, pass the registered shorter run's `evaluation_keys.json` to `--evaluation-keys-file`.

Run `python audit_time_llm_replication.py D:/traffic-research` on a workspace containing the completed queue status and original output folders recorded in that status. The audit checks source and adapter hashes and independently reconstructs each road-level MAE and RMSE from station aggregates. The published bundle contains aggregate results, the registry and provenance hashes. The underlying checkpoints and evaluation keys stay in the research workspace.
