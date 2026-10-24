"""CPU unit checks; synthetic tensors only, no formal experiment fitting."""

import pytest
import torch
from torch.nn import functional as F

from task2q.model import DelayedDirectedGraph, QueueForecaster, _delayed_history, masked_queue_loss


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(43)


def sample():
    return (torch.randn(2, 13, 5, 8), torch.randn(5, 6),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            torch.tensor([0, 1]), torch.tensor([0.5, 1.0, 2.0, 0.8, 1.0]))


def tiny(**kwargs):
    return QueueForecaster(8, width=16, layers=1, heads=2, dropout=0.0, **kwargs)


def test_shapes_finite_backward_and_condition_heads():
    model = tiny()
    x, static, edges, condition, length = sample()
    output = model(x, static, edges, condition, length)
    assert set(output) == {"logits", "speed"}
    assert output["logits"].shape == (2, 6, 5)
    assert torch.isfinite(output["speed"]).all()
    labels = torch.randint(0, 2, (2, 6, 5)).float()
    loss = masked_queue_loss(output["logits"], labels, torch.ones_like(labels).bool())
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    for name, parameter in model.named_parameters():
        if name.startswith("speed_head"):
            continue  # auxiliary speed requires its own optional supervised loss
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert all(head.weight.grad.abs().sum() > 0 for head in model.queue_heads)


def test_eval_is_deterministic_and_import_does_not_fit():
    model = tiny().eval()
    inputs = sample()
    with torch.no_grad():
        first = model(*inputs)["logits"]
        second = model(*inputs)["logits"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_isolated_nodes_do_not_borrow_other_histories():
    model = tiny().eval()
    x, static, edges, condition, length = sample()
    altered = x.clone()
    altered[:, :, :3] += 20
    with torch.no_grad():
        baseline = model(x, static, edges, condition, length)["logits"]
        changed = model(altered, static, edges, condition, length)["logits"]
    torch.testing.assert_close(baseline[:, :, 3:], changed[:, :, 3:], rtol=0, atol=0)
    assert not torch.allclose(baseline[:, :, :3], changed[:, :, :3])


def test_time_only_ablation_is_edge_independent():
    model = tiny(use_graph=False).eval()
    x, static, edges, condition, length = sample()
    with torch.no_grad():
        a = model(x, static, edges, condition, length)["logits"]
        b = model(x, static, torch.empty(2, 0, dtype=torch.long), condition, length)["logits"]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_direction_and_delay_affect_connected_nodes():
    model = tiny().eval()
    x, static, edges, condition, length = sample()
    with torch.no_grad():
        normal = model(x, static, edges, condition, length)["logits"]
        reversed_edges = model(x, static, edges.flip(0), condition, length)["logits"]
        delayed = model(x, static, edges, condition, length * 5)["logits"]
    assert not torch.allclose(normal[:, :, :3], reversed_edges[:, :, :3])
    assert not torch.allclose(normal[:, :, :3], delayed[:, :, :3])
    torch.testing.assert_close(normal[:, :, 3:], delayed[:, :, 3:], rtol=0, atol=0)


def test_fractional_delay_uses_past_only():
    history = torch.arange(4.0).reshape(1, 4, 1, 1)
    delayed = _delayed_history(history, torch.tensor([0]), torch.tensor([0.5]))
    torch.testing.assert_close(delayed.flatten(), torch.tensor([0.0, 0.5, 1.5, 2.5]))
    changed = history.clone()
    changed[:, -1] = 100
    delayed_changed = _delayed_history(changed, torch.tensor([0]), torch.tensor([1.0]))
    torch.testing.assert_close(delayed_changed.flatten(), torch.tensor([0.0, 0.0, 1.0, 2.0]))


def test_graph_branch_exactly_zero_for_isolated_nodes_and_no_edges():
    graph = DelayedDirectedGraph(8, 2, 0)
    states = torch.randn(1, 13, 4, 8)
    edges = torch.tensor([[0], [1]])
    output = graph(states, edges, torch.ones(4))
    assert torch.count_nonzero(output[:, :, 2:]) == 0
    assert torch.count_nonzero(graph(states, torch.empty(2, 0, dtype=torch.long), None)) == 0


def test_loss_is_equal_per_window_and_ignores_unknown_labels():
    logits = torch.tensor([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]], requires_grad=True)
    labels = torch.tensor([[[1.0, float("nan"), float("nan")]], [[0.0, 1.0, 0.0]]])
    mask = torch.ones_like(labels).bool()
    result = masked_queue_loss(logits, labels, mask, iou_weight=0)
    expected = (F.binary_cross_entropy_with_logits(logits[0, 0, 0], torch.tensor(1.0)) +
                F.binary_cross_entropy_with_logits(logits[1], labels[1])) / 2
    torch.testing.assert_close(result["bce"], expected)
    result["total"].backward()
    assert torch.count_nonzero(logits.grad[0, 0, 1:]) == 0


def test_empty_supervision_is_not_counted_and_zero_loss_has_gradient():
    logits = torch.randn(2, 6, 3, requires_grad=True)
    labels = torch.full_like(logits, float("nan"))
    result = masked_queue_loss(logits, labels, torch.zeros_like(logits).bool())
    assert result["valid_windows"] == 0
    assert result["total"] == 0
    result["total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_confidence_zero_excludes_cells_and_unknown_window():
    logits = torch.randn(2, 6, 3, requires_grad=True)
    labels = torch.randint(0, 2, logits.shape).float()
    confidence = torch.ones_like(labels)
    confidence[0] = 0
    result = masked_queue_loss(logits, labels, torch.ones_like(labels).bool(), confidence)
    assert result["valid_windows"] == 1
    expected = masked_queue_loss(logits[1:], labels[1:], torch.ones_like(labels[1:]).bool())
    torch.testing.assert_close(result["total"], expected["total"])
    result["total"].backward()
    assert torch.count_nonzero(logits.grad[0]) == 0


def test_small_confidence_preserves_window_normalization():
    logits = torch.tensor([[[0.7]]])
    labels = torch.ones_like(logits)
    mask = torch.ones_like(logits).bool()
    result = masked_queue_loss(logits, labels, mask, torch.full_like(logits, 0.1))
    torch.testing.assert_close(result["bce"], F.binary_cross_entropy_with_logits(logits, labels))


def test_batch_static_short_history_and_no_edges():
    x, static, _, condition, _ = sample()
    result = tiny()(x[:, -12:], static.unsqueeze(0).expand(2, -1, -1),
                    torch.empty(2, 0, dtype=torch.long), condition)
    assert result["logits"].shape == (2, 6, 5)


def test_invalid_shape_condition_and_graph_fail_loudly():
    model = tiny()
    x, static, edges, condition, length = sample()
    with pytest.raises(ValueError, match="condition"):
        model(x, static, edges, torch.tensor([0, 2]), length)
    with pytest.raises(ValueError, match="out-of-range"):
        model(x, static, torch.tensor([[0], [5]]), condition, length)
    with pytest.raises(ValueError, match="length_km"):
        model(x, static, edges, condition, torch.tensor([-1.0] * 5))
    with pytest.raises(ValueError, match="width"):
        QueueForecaster(8, width=17, heads=2)
