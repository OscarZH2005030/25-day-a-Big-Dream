"""Causal engineering baselines; small tree fits are smoke checks, not results."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "speed_ratio", "flow_ratio", "occupancy", "quality", "speed_missing",
    "flow_missing", "occupancy_missing", "eligible", "age", "speed_margin",
    "speed_delta", "sin_time", "cos_time",
)
_F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _inputs(sample, net):
    x = np.asarray(sample["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[0] != 13 or x.shape[2] != len(FEATURE_NAMES):
        raise ValueError("Expected x [13, links, 13] with the frozen FEATURE_NAMES")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must use finite masked fills")
    names = sample.get("feature_names", FEATURE_NAMES)
    if tuple(names) != FEATURE_NAMES:
        raise ValueError("Input feature allowlist/order changed")
    free = np.asarray(net["free_speed"], dtype=np.float32)
    threshold = np.asarray(net["threshold"], dtype=np.float32)
    if free.shape != (x.shape[1],) or threshold.shape != free.shape or np.any(free <= 0):
        raise ValueError("Network speed arrays do not match x")
    return x, free, threshold


def _latest(x, last_index):
    """Latest finite observed speed at or before the requested cutoff."""
    value = np.ones(x.shape[1], dtype=np.float32)
    known = np.zeros(x.shape[1], dtype=bool)
    for t in range(last_index + 1):
        observed = x[t, :, _F["speed_missing"]] < .5
        value[observed] = x[t, observed, _F["speed_ratio"]]
        known |= observed
    return value, known


def persistence(sample, net, at_origin=False):
    """Causal carry baseline through T-5, optionally through the legal T frame.

    Missing speeds carry the most recent observed historical speed; a link with
    no observed history defaults to free flow. No future masks enter prediction.
    This differs from official_persistence: it carries older observations and
    does not discard an observed low-quality speed. Label it as a carry baseline.
    """
    x, free, threshold = _inputs(sample, net)
    speed, _ = _latest(x, 12 if at_origin else 11)
    # Stay in the feature's ratio units: multiplying float32 .6 by 100 can
    # round to 60.000004 and wrongly reject the official inclusive boundary.
    binary = (speed <= threshold / free).astype(np.float32)
    return np.broadcast_to(binary, (6, len(free))).copy()


def official_persistence(sample, net):
    """Released baseline: exact T-5 speed, history eligibility, no carry.

    A missing/ineligible T-5 cell predicts zero regardless of earlier history.
    Frame 12 is T, while the official twelve-frame history ends at frame 11.
    """
    x, free, threshold = _inputs(sample, net)
    last = x[11]
    eligible = (last[:, _F["eligible"]] > .5) & (last[:, _F["speed_missing"]] < .5)
    queued = eligible & (last[:, _F["speed_ratio"]] <= threshold / free)
    return np.broadcast_to(queued.astype(np.float32), (6, len(free))).copy()


def propagation(sample, net):
    """Past-only trend and one-edge queue-pressure probability baseline.

    This intentionally small baseline is not a CTM solver. Both directions of an
    internal edge can transmit pressure, with reduced upstream-to-downstream
    strength. Isolated links use their own temporal trend exactly.
    """
    x, free, threshold = _inputs(sample, net)
    latest, known = _latest(x, 12)
    earlier, earlier_known = _latest(x, 9)
    trend = np.where(known & earlier_known, np.clip((latest - earlier) / 3, -.08, .08), 0)
    cutoff = threshold / free
    horizons = np.arange(1, 7, dtype=np.float32)[:, None]
    extrapolated = np.clip(latest[None] + horizons * trend[None], 0, 1.6)
    probability = 1 / (1 + np.exp(np.clip((extrapolated - cutoff[None]) / .055, -30, 30)))
    probability[:, ~known] = 0
    edges = np.asarray(net.get("edges", net.get("edge_index", np.empty((2, 0)))), dtype=int)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edges must be [2, E]")
    if edges.size and (np.min(edges) < 0 or np.max(edges) >= len(free)):
        raise ValueError("Edge index outside network")
    own = probability.copy()
    for up, down in edges.T:
        # A queued downstream neighbor can send an upstream-moving queue front.
        probability[1:, up] = np.maximum(probability[1:, up], .65 * own[:-1, down])
        probability[1:, down] = np.maximum(probability[1:, down], .25 * own[:-1, up])
    return probability.astype(np.float32)


def tree_features(sample, net):
    """Horizon-major [6*N, D] features; never reads labels or future quality."""
    x, free, threshold = _inputs(sample, net)
    n = len(free)
    static = np.asarray(net["static"], dtype=np.float32)
    if static.ndim != 2 or static.shape[0] != n or not np.isfinite(static).all():
        raise ValueError("Invalid static features")
    latest, known = _latest(x, 12)
    old15, _ = _latest(x, 9)
    old30, _ = _latest(x, 6)
    valid = x[:, :, _F["speed_missing"]] < .5
    observed_speed = np.where(valid, x[:, :, _F["speed_ratio"]], np.nan)
    sums = np.nansum(observed_speed, axis=0)
    count = valid.sum(0)
    speed_mean = np.divide(sums, count, out=np.ones(n), where=count > 0)
    speed_min = np.min(np.where(valid, observed_speed, np.inf), axis=0)
    speed_min[~known] = 1
    neighbor_sum, neighbor_count = np.zeros(n), np.zeros(n)
    downstream_pressure = np.zeros(n)
    edges = np.asarray(net.get("edges", net.get("edge_index", np.empty((2, 0)))), dtype=int)
    for up, down in edges.T:
        if known[down]:
            downstream_pressure[up] = max(downstream_pressure[up], max(0., threshold[down] / free[down] - latest[down]))
        for src, dst in ((up, down), (down, up)):
            if known[src]:
                neighbor_sum[dst] += latest[src]
                neighbor_count[dst] += 1
    neighbor_mean = np.divide(neighbor_sum, neighbor_count, out=np.ones(n), where=neighbor_count > 0)
    condition = int(sample["condition"])
    if condition not in (0, 1):
        raise ValueError("condition must be 0/1")
    extras = np.column_stack((latest, latest-old15, latest-old30, speed_mean,
                              speed_min, count/13, neighbor_mean, neighbor_count,
                              downstream_pressure, np.full(n, condition)))
    base = np.concatenate((x.transpose(1, 0, 2).reshape(n, -1), static, extras), axis=1)
    one_hot = np.repeat(np.eye(6, dtype=np.float32), n, axis=0)
    result = np.concatenate((np.tile(base, (6, 1)), one_hot), axis=1).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Non-finite engineered features")
    return result


def fit_lightgbm(samples, nets, rounds=12, seed=20261024):
    """Bounded 1..20-round fit on fit-role proxy labels only; no tuning."""
    if not 1 <= int(rounds) <= 20:
        raise ValueError("Engineering LightGBM smoke rounds must be in 1..20")
    import lightgbm as lgb
    xx, yy = [], []
    for sample in samples:
        if sample.get("role") != "fit":
            raise ValueError("LightGBM fitting only permits frozen fit-role samples")
        net = nets[sample["panel"]] if sample["panel"] in nets else nets
        features = tree_features(sample, net)
        mask = np.asarray(sample.get("m", sample.get("label_mask")), dtype=bool)
        labels = np.asarray(sample["y"], dtype=np.float32)
        if mask.shape != labels.shape or mask.shape != (6, len(net["free_speed"])):
            raise ValueError("Invalid proxy label shape")
        if not np.all(np.isin(labels[mask], [0, 1])):
            raise ValueError("Tree hard-proxy labels must be binary")
        xx.append(features[mask.ravel()])
        yy.append(labels.ravel()[mask.ravel()])
    if not xx or not sum(len(a) for a in yy):
        raise ValueError("No eligible fit targets")
    X, y = np.concatenate(xx), np.concatenate(yy)
    model = lgb.train({"objective": "binary", "learning_rate": .08, "num_leaves": 7,
        "min_data_in_leaf": 8, "num_threads": 4, "verbosity": -1,
        "seed": int(seed), "deterministic": True, "force_col_wise": True},
        lgb.Dataset(X, label=y, free_raw_data=True), num_boost_round=int(rounds))
    return model


def predict_lightgbm(model, sample, net):
    return np.asarray(model.predict(tree_features(sample, net)), dtype=np.float32).reshape(6, -1)


def predict(sample, net, method="persistence_tminus5"):
    if method == "official_persistence":
        return official_persistence(sample, net)
    if method == "persistence_tminus5":
        return persistence(sample, net, at_origin=False)
    if method in ("persistence_t", "persistence_T"):
        return persistence(sample, net, at_origin=True)
    if method == "propagation":
        return propagation(sample, net)
    raise ValueError(f"Unknown baseline method: {method}")
