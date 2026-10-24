from copy import deepcopy
import numpy as np
import pytest
from task2q.baselines import (FEATURE_NAMES, persistence, propagation, tree_features,
                              fit_lightgbm, predict_lightgbm, official_persistence)


def examples():
    x = np.zeros((13, 2, 13), dtype=np.float32)
    x[:, :, 0] = .8
    x[:, :, 3] = 1
    x[:, :, 7] = 1
    sample = dict(x=x, y=np.tile([[0, 1]], (6, 1)).astype(np.float32),
                  m=np.ones((6, 2), bool), condition=0, panel="D7_I10_E", role="fit")
    net = dict(free_speed=np.array([100, 100]), threshold=np.array([60, 60]),
               static=np.ones((2, 6), np.float32), edges=np.empty((2, 0), int))
    return sample, net


def test_persistence_tminus5_does_not_read_origin_speed():
    sample, net = examples()
    sample["x"][-1, 0, 0] = .4
    assert persistence(sample, net, at_origin=False)[:, 0].sum() == 0
    assert persistence(sample, net, at_origin=True)[:, 0].sum() == 6
    sample["x"][-1, 0, 4] = 1  # T is missing: use latest earlier observation.
    assert persistence(sample, net, at_origin=True)[:, 0].sum() == 0
    # Exact threshold belongs to the queue, as in official queue_utils.
    sample["x"][:, 0, 0] = .6
    assert persistence(sample, net, at_origin=False)[:, 0].sum() == 6


def test_feature_allowlist_and_future_label_invariance():
    sample, net = examples()
    expected = tree_features(sample, net)
    revised = deepcopy(sample)
    revised.update(y=np.ones((6, 2)), m=np.zeros((6, 2), bool),
                   future_coverage=0., future_speed=np.zeros((6, 2)))
    np.testing.assert_array_equal(tree_features(revised, net), expected)
    np.testing.assert_array_equal(propagation(revised, net), propagation(sample, net))
    # Each horizon repeats the same causal row, with its own six-way query.
    np.testing.assert_array_equal(expected[:, -6:], np.repeat(np.eye(6), 2, axis=0))
    with pytest.raises(ValueError, match="allowlist"):
        tree_features(dict(sample, feature_names=FEATURE_NAMES[:-1]+("future_coverage",)), net)


def test_official_persistence_exact_frame_quality_and_missing_policy():
    # Compute the independent released formula from raw speed and quality first.
    raw_speed = np.array([81., 60., np.nan, 20.])
    raw_eligible = np.array([1, 1, 1, 0], dtype=bool)
    expected = np.isfinite(raw_speed) & raw_eligible & (raw_speed <= 60.)
    x = np.zeros((13, 4, len(FEATURE_NAMES)), np.float32)
    x[:, :, 0] = .4   # Earlier history is queued, including the later missing cell.
    x[:, :, 7] = 1
    x[11, :, 0] = np.where(np.isfinite(raw_speed), raw_speed/100, 1)
    x[11, :, 4] = ~np.isfinite(raw_speed)
    x[11, :, 7] = raw_eligible
    x[12, :, 0] = .1  # A queued T does not alter official T-5 persistence.
    sample = dict(x=x)
    net = dict(free_speed=np.full(4, 100.), threshold=np.full(4, 60.))
    np.testing.assert_array_equal(official_persistence(sample, net), np.tile(expected, (6, 1)))
    # Carry methods remain distinct: they reuse missing T-5's previous speed and
    # may use the observed low-quality speed instead of the official zero rule.
    np.testing.assert_array_equal(persistence(sample, net)[0], [0, 1, 1, 1])
    np.testing.assert_array_equal(persistence(sample, net, at_origin=True)[0], np.ones(4))


def test_isolated_link_forecasts_do_not_use_other_link():
    sample, net = examples()
    before = propagation(sample, net)
    sample["x"][:, 1, 0] = .1
    np.testing.assert_array_equal(propagation(sample, net)[:, 0], before[:, 0])
    net["edges"] = np.array([[0], [1]])
    assert np.max(propagation(sample, net)[1:, 0] - before[1:, 0]) > .1


def test_tree_fit_cannot_use_calibration_or_audit_roles():
    sample, net = examples()
    for role in ("tune", "calibrate", "audit", "validation"):
        with pytest.raises(ValueError, match="fit-role"):
            fit_lightgbm([dict(sample, role=role)], {sample["panel"]: net}, rounds=2)
    model = fit_lightgbm([sample], {sample["panel"]: net}, rounds=2)
    pred = predict_lightgbm(model, sample, net)
    assert pred.shape == (6, 2) and np.isfinite(pred).all()
    with pytest.raises(ValueError, match="1..20"):
        fit_lightgbm([sample], {sample["panel"]: net}, rounds=21)
