"""Check Task 1 and Task 2 baseline files against the released templates."""
from __future__ import annotations

from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd


def check_state(root: Path, split: str) -> int:
    output = root / "outputs" / f"task1_historical_mean_{split}.csv"
    template_paths = sorted((root / "data" / "kaggle_public" / "task1").glob(f"*/{split}/sample_submission_state.csv"))
    template_columns = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
    output_columns = template_columns + ["speed_kmh", "flow_vph"]
    prediction = pd.read_csv(output, usecols=output_columns, dtype={c: str for c in template_columns}, chunksize=250_000)
    pred_iter = iter(prediction)
    active = None
    offset = 0
    checked = 0
    for path in template_paths:
        template_iter = pd.read_csv(path, usecols=template_columns, dtype=str, chunksize=250_000)
        for t_chunk in template_iter:
            need = len(t_chunk)
            pieces = []
            while need:
                if active is None or offset == len(active):
                    active = next(pred_iter, None)
                    offset = 0
                if active is None:
                    raise ValueError(f"Task 1 {split} prediction ended before {path}")
                take = min(need, len(active) - offset)
                pieces.append(active.iloc[offset:offset + take])
                offset += take
                need -= take
            p_chunk = pd.concat(pieces, ignore_index=True)
            if not t_chunk[template_columns].reset_index(drop=True).equals(p_chunk[template_columns].reset_index(drop=True)):
                raise ValueError(f"Task 1 key/order mismatch in {split} near {path}")
            values = p_chunk[["speed_kmh", "flow_vph"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid Task 1 values in {split}")
            checked += len(p_chunk)
    if (active is not None and offset < len(active)) or next(pred_iter, None) is not None:
        raise ValueError(f"Task 1 {split} has extra prediction rows after {checked:,} checked")
    return checked


def check_queue(root: Path, split: str) -> int:
    data = root / "data" / "kaggle_public"
    paths = sorted((data / "task2").glob(f"*/{split}/sample_submission_queue.csv"))
    template = pd.concat([pd.read_csv(p, dtype={"window_id": str, "timestamp": str, "link_id": str}) for p in paths], ignore_index=True)
    prediction = pd.read_csv(root / "outputs" / f"task2_persistence_{split}.csv",
                             dtype={"window_id": str, "timestamp": str, "link_id": str})
    keys = ["window_id", "timestamp", "link_id"]
    template["timestamp"] = pd.to_datetime(template["timestamp"], utc=True)
    prediction["timestamp"] = pd.to_datetime(prediction["timestamp"], utc=True)
    if template.duplicated(keys).any() or prediction.duplicated(keys).any():
        raise ValueError(f"Duplicate Task 2 keys in {split}")
    comparison = template[keys].merge(prediction[keys], on=keys, how="outer", indicator=True)
    if not comparison["_merge"].eq("both").all():
        raise ValueError(f"Task 2 template keys do not match predictions in {split}")
    if not prediction["queue_pred"].isin([0, 1]).all():
        raise ValueError(f"Non-binary queue predictions in {split}")
    return len(prediction)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for split in ("validation", "private"):
        print(f"Task 1 {split}: {check_state(root, split):,} rows; keys exact; values finite/nonnegative")
    for split in ("train", "validation", "private"):
        print(f"Task 2 {split}: {check_queue(root, split):,} rows; UTC keys match; values binary")
    proxy = pd.read_csv(root / "outputs" / "task2_persistence_train_proxy.csv")
    print(f"Task 2 train proxy: {len(proxy)} windows; mean IoU={proxy['IoU'].mean():.6f}")


if __name__ == "__main__":
    main()
