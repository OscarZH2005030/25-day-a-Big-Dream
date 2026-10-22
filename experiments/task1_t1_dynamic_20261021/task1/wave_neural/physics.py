"""Local physical proxies, not a reproduction of the private Task 3 evaluator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

PARAMETERS = ['lanes', 'length_km', 'free_speed_kmh', 'capacity_vph', 'critical_density', 'k_jam']
DT_HOURS = 5 / 60


def parts(value):
    return [] if pd.isna(value) or str(value) == '' else str(value).split(';')


def structure(nodes, net, topology, attachments):
    links = nodes.link_id.astype(str).tolist()
    if len(set(links)) != len(links):
        raise ValueError('Duplicate node links')
    index = {link: i for i, link in enumerate(links)}
    col = 'mainline_link_id' if 'mainline_link_id' in topology else 'link_id'
    if topology[col].duplicated().any():
        raise ValueError('Duplicate mainline topology rows')
    top = topology.set_index(col)
    if set(links) - set(top.index):
        raise ValueError('Node absent from topology')
    par = net.loc[links, PARAMETERS].to_numpy(dtype=np.float32)
    if not np.isfinite(par).all() or (par <= 0).any() or (par[:, 5] <= par[:, 4]).any():
        raise ValueError('Invalid fundamental-diagram parameters')
    if not np.array_equal(par[:, 0], nodes.lanes.to_numpy(dtype=np.float32)):
        raise ValueError('Node lane counts disagree with FD data')
    upstream = np.arange(len(links), dtype=np.int64)
    downstream = upstream.copy()
    valid = np.zeros(len(links), dtype=bool)
    for link, i in index.items():
        row = top.loc[link]
        incoming, outgoing = parts(row.incoming_link_ids), parts(row.outgoing_link_ids)
        # An unknown connector is missing boundary information, not zero flow.
        if len(incoming) != 1 or len(outgoing) != 1:
            continue
        u, d = incoming[0], outgoing[0]
        if u not in index or d not in index or u == link or d == link:
            continue
        if link not in parts(top.loc[u].outgoing_link_ids) or link not in parts(top.loc[d].incoming_link_ids):
            continue
        upstream[i], downstream[i], valid[i] = index[u], index[d], True
    attach = attachments.copy()
    if attach.ramp_link_id.duplicated().any() or attach.ramp_link_id.isna().any():
        raise ValueError('Ramp IDs must be unique and non-null')
    if not attach.ramp_type.isin(['OR', 'FR']).all() or not attach.nearest_mainline_link_id.isin(links).all():
        raise ValueError('Unknown ramp type or attachment')
    for link in links:
        for kind, name in [('OR', 'on_ramp_link_ids'), ('FR', 'off_ramp_link_ids')]:
            declared = set(parts(top.loc[link].get(name)))
            mapped = set(attach.loc[attach.nearest_mainline_link_id.eq(link) & attach.ramp_type.eq(kind), 'ramp_link_id'])
            if declared != mapped:
                raise ValueError('Ramp map and topology disagree: ' + link)
    attached = set(attach.nearest_mainline_link_id)
    metadata = dict(nodes=len(links), conservation_nodes=int(valid.sum()), excluded_boundary_or_branch_nodes=int((~valid).sum()),
                    conservation_nodes_with_ramps=int(sum(valid[index[x]] for x in attached)),
                    conservation_proxy='midpoint detector-center flux, trapezoidal time integration; private boundary flows unavailable',
                    density='total_flow / max(speed, 1 km/h)', dt_hours=DT_HOURS)
    return dict(parameters=par, upstream=upstream, downstream=downstream, valid_nodes=valid), metadata


def ramp_grid(frame, date, nodes, attachments):
    """Unavailable attached ramps invalidate a cell; absent attachments are zero."""
    shape = (288, len(nodes))
    total, valid = np.zeros(shape, np.float32), np.ones(shape, bool)
    if attachments.empty:
        return total, valid
    frame = frame.copy()
    if frame.duplicated(['timestamp', 'ramp_link_id']).any():
        raise ValueError('Duplicate ramp measurements')
    ts = pd.to_datetime(frame.timestamp, utc=True)
    if not ts.dt.strftime('%Y-%m-%d').eq(date).all() or not ((ts.dt.minute % 5 == 0) & (ts.dt.second == 0)).all():
        raise ValueError('Ramp data have wrong date or sampling interval')
    frame['slot'] = ts.dt.hour * 12 + ts.dt.minute // 5
    positions = {link: i for i, link in enumerate(nodes.link_id)}
    for row in attachments.itertuples(index=False):
        i = positions[row.nearest_mainline_link_id]
        measurements = frame.loc[frame.ramp_link_id.eq(row.ramp_link_id)].set_index('slot').reindex(range(288))
        values = pd.to_numeric(measurements.flow_vph, errors='coerce').to_numpy(float)
        quality = pd.to_numeric(measurements.pct_observed, errors='coerce').to_numpy(float)
        eligible = pd.to_numeric(measurements.is_score_eligible, errors='coerce').to_numpy(float)
        good = np.isfinite(values) & (values >= 0) & (quality >= 75) & (eligible == 1)
        valid[:, i] &= good
        total[:, i] += np.where(good, values, 0).astype(np.float32) * (1 if row.ramp_type == 'OR' else -1)
    return total, valid


def tensors(arrays, device):
    return {name: torch.as_tensor(np.array(value,copy=True), device=device) for name, value in arrays.items()}


def masked_mean(values, active):
    weights = active.to(values.dtype)
    return (torch.where(active, values, 0) * weights).sum() / weights.sum().clamp_min(1)


def fd_excess(values, par):
    lanes, _, vf, capacity, critical, jam = par.unbind(-1)
    speed, q_lane = values.unbind(-1)
    capacity, critical, jam = capacity / lanes, critical / lanes, jam / lanes
    density = q_lane / speed.clamp_min(1)
    backward_speed = capacity / (jam - critical)
    return torch.stack([
        F.relu(speed - 1.1 * vf) / vf,
        F.relu(q_lane - 1.1 * capacity) / capacity,
        F.relu(density - 1.1 * jam) / jam,
        F.relu(q_lane - backward_speed * (jam-density).clamp_min(0) - .1*capacity) / capacity,
    ], dim=-1)


def conservation(values, active, ramp_net, ramp_valid, physical):
    par = physical['parameters']
    lanes, length, _, capacity, _, jam = par.unbind(-1)
    q = values[..., 1] * lanes
    accumulation = length * q / values[..., 0].clamp_min(1)
    up, down = physical['upstream'], physical['downstream']
    inflow, outflow = .5 * (q + q[..., up]), .5 * (q + q[..., down])
    rate = inflow - outflow + ramp_net
    rhs = DT_HOURS * .5 * (rate[:, :-1] + rate[:, 1:])
    residual = accumulation[:, 1:] - accumulation[:, :-1] - rhs
    unknown = active.any(-1)
    involved = unknown | unknown[..., up] | unknown[..., down]
    valid = (physical['valid_nodes'] & ramp_valid[:, :-1] & ramp_valid[:, 1:]
             & (involved[:, :-1] | involved[:, 1:]))
    scale = capacity * DT_HOURS + jam * length
    tolerance = 5. + .05 * capacity * DT_HOURS
    excess = F.relu(residual.abs() - tolerance) / scale
    return residual, excess, valid, rhs


def losses(prediction, target, supervised, mask, ramps, ramp_valid, physical):
    scale = prediction.new_tensor([25., 600.])
    errors = ((prediction - target) / scale).square()
    data = .54 * masked_mean(errors[..., 0], supervised[..., 0]) + .46 * masked_mean(errors[..., 1], supervised[..., 1])
    active = ~mask
    fd = masked_mean(fd_excess(prediction, physical['parameters']).square().mean(-1), active.any(-1))
    residual, excess, valid, _ = conservation(prediction, active, ramps, ramp_valid, physical)
    lwr = masked_mean(excess.square(), valid)
    return data, fd, lwr, valid.sum()


@torch.inference_mode()
def diagnostics(values, active, ramps, ramp_valid, physical):
    excess = fd_excess(values, physical['parameters'])
    fd_valid = active.any(-1)
    residual, normalized, valid, _ = conservation(values, active, ramps, ramp_valid, physical)
    return dict(fd_cells=int(fd_valid.sum()), fd_squared_excess_sum=float((excess.square().mean(-1)*fd_valid).sum()),
                fd_violation_cells=int((excess.gt(0).any(-1) & fd_valid).sum()),
                conservation_transitions=int(valid.sum()), conservation_abs_residual_veh_sum=float((residual.abs()*valid).sum()),
                conservation_normalized_excess_sum=float((normalized*valid).sum()))
