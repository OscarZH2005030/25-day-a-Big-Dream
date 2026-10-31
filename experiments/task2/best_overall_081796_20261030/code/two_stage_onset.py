"""Experimental onset probability core; not a trained R127 replacement.

Stage 1: P(any queue on this link in the six forecast steps).
Stage 2: P(six-bit occupancy pattern | any queue), over 63 nonempty patterns.
Unlike peak-preserving R127, this can change event support and both boundaries.
All patterns permit clearing and repeated spells. Missing labels are marginalized.
No model, data loader, training run, or evaluation claim is included.
"""
import numpy as np

PATTERNS = ((np.arange(64)[:, None] >> np.arange(6)) & 1).astype(float)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def joint_probability(event_logit, pattern_logits):
    a = np.asarray(event_logit, dtype=float)
    b = np.asarray(pattern_logits, dtype=float)
    if b.shape != a.shape + (63,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite [N] event logits and [N,63] pattern logits')
    # stable sigmoid
    event = np.exp(-np.logaddexp(0., -a))
    conditional = _softmax(b)
    return np.concatenate([(1-event)[..., None], event[..., None]*conditional], axis=-1)


def occupancy_probability(event_logit, pattern_logits):
    """Return [...,6]; transpose to [6,N] for the existing submission pipeline."""
    return joint_probability(event_logit, pattern_logits) @ PATTERNS


def compatibility(y, mask):
    y, mask = np.asarray(y), np.asarray(mask, dtype=bool)
    if y.shape != mask.shape or y.shape[-1] != 6:
        raise ValueError('Labels and masks must have matching [...,6] shape')
    if not np.isin(y[mask], [0, 1]).all():
        raise ValueError('Observed proxy labels must be binary')
    # np.where handles arbitrary placeholders, including NaN, outside mask.
    observed = np.where(mask, y, 0)
    return np.all((PATTERNS == observed[..., None, :]) | ~mask[..., None, :], axis=-1)


def partial_nll_and_grad(event_logit, pattern_logits, y, mask):
    """Per-row NLL and gradients; average only rows with >=1 observed label.

    Under ignorable missingness, -log sum_{compatible patterns} P(pattern).
    Quality-dependent missingness and noisy proxy labels remain limitations.
    Unknown event identity is never forced to a negative event target.
    """
    a, b = np.asarray(event_logit, float), np.asarray(pattern_logits, float)
    # Work in log space so extreme logits do not underflow the likelihood.
    joint_probability(a, b)  # validate
    log_c = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    log_joint = np.concatenate([-np.logaddexp(0., a)[..., None],
        -np.logaddexp(0., -a)[..., None]+log_c], axis=-1)
    allowed = compatibility(y, mask)
    log_mass = np.logaddexp.reduce(np.where(allowed, log_joint, -np.inf), axis=-1)
    posterior = np.exp(np.where(allowed, log_joint-log_mass[..., None], -np.inf))
    event = np.exp(-np.logaddexp(0., -a))
    post_event = posterior[..., 1:].sum(axis=-1)
    grad_a = event-post_event
    grad_b = post_event[..., None]*np.exp(log_c)-posterior[..., 1:]
    return -log_mass, grad_a, grad_b


def causal_precursors(sample):
    """Named candidate features [N,D] from the existing public sample schema.

    Recent Theil-Sen speed slope uses reliable observations and actual time gaps;
    unknown/stale values stay NaN for a missing-aware learner. No target access.
    time_to_cut is a hypothesis, not an established determinant.
    """
    x = np.asarray(sample['x'], float)
    net = sample['network']
    if x.ndim != 3 or x.shape[0] != 13 or x.shape[2] != 13:
        raise ValueError('Expected task2q 13 x N x 13 public history')
    free = np.asarray(net['free_speed'], float)
    cut = np.asarray(net['threshold'], float)/free
    if free.shape != (x.shape[1],) or np.any(free <= 0):
        raise ValueError('Invalid network/free-speed array')
    good = (x[:, :, 4] < .5) & (x[:, :, 3] >= .75) & (x[:, :, 7] > .5) & np.isfinite(x[:, :, 0])
    names = ['latest_speed_ratio', 'latest_age_min', 'margin_to_cut',
             'slope_per_min', 'slope_MAD', 'closing_speed_per_min',
             'time_to_cut_min', 'projected_margin_5min', 'projected_margin_30min',
             'recent_reliable_fraction', 'observed_T', 'decelerating_pair_fraction']
    result = np.full((x.shape[1], len(names)), np.nan)
    for i in range(x.shape[1]):
        valid = np.flatnonzero(good[:, i])
        result[i, 9:11] = [good[-6:, i].mean(), float(good[-1, i])]
        if not len(valid):
            continue
        last = valid[-1]
        age = (12-last)*5
        speed = x[last, i, 0]
        margin = speed-cut[i]
        result[i, :3] = [speed, age, margin]
        recent = valid[valid >= 7]
        if len(recent) < 3:
            continue
        pairs = [(x[v, i, 0]-x[u, i, 0])/((v-u)*5) for k,u in enumerate(recent) for v in recent[k+1:]]
        slope = np.median(pairs)
        result[i, 3:6] = [slope, np.median(np.abs(pairs-slope)), max(-slope, 0.)]
        # Extrapolation is referenced to the last observation, accounting for age.
        if slope < -1e-8 and margin > 0:
            result[i, 6] = margin/(-slope)-age
        result[i, 7:9] = [margin+slope*(age+5), margin+slope*(age+30)]
        result[i, 11] = np.mean(np.asarray(pairs) < 0)
    return result, names
