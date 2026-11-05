import copy

import pytest
import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_cross_entropy_batched_loss_and_gradients(dtype):
    generator = torch.Generator().manual_seed(31)
    logits = (torch.randn(2, 3, 7, generator=generator) * 1000).to(dtype).requires_grad_()
    targets = torch.randint(7, (2, 3), generator=generator)
    reference = logits.detach().clone().requires_grad_()
    expected_logits = reference.float() if dtype in (torch.float16, torch.bfloat16) else reference
    expected = F.cross_entropy(expected_logits.flatten(0, 1), targets.flatten())
    actual = cross_entropy(logits, targets)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)


def test_adamw_parameter_groups_and_missing_gradients():
    parameters = [torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64)) for _ in range(3)]
    references = [torch.nn.Parameter(p.detach().clone()) for p in parameters]

    def groups(params):
        return [
            {"params": params[:2], "lr": 0.03, "betas": (0.7, 0.8), "weight_decay": 0.2},
            {"params": params[2:], "lr": 0.01, "betas": (0.2, 0.9), "weight_decay": 0.0},
        ]

    # With epsilon zero, the two equivalent bias-correction forms coincide.
    optimizer = AdamW(groups(parameters), eps=0)
    reference_optimizer = torch.optim.AdamW(groups(references), eps=0)
    for step in range(5):
        for index, (parameter, reference) in enumerate(zip(parameters, references)):
            grad = None if index == 1 and step % 2 == 0 else torch.tensor([step + 0.5, -0.3], dtype=torch.float64)
            parameter.grad = grad
            reference.grad = None if grad is None else grad.clone()
        optimizer.step()
        reference_optimizer.step()
        for parameter, reference in zip(parameters, references):
            torch.testing.assert_close(parameter, reference)
    assert optimizer.state[parameters[0]]["step"] == 5
    assert optimizer.state[parameters[1]]["step"] == 2


def test_adamw_first_step_matches_assignment_epsilon_and_decay():
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimizer = AdamW([parameter], lr=0.1, betas=(0.5, 0.75), eps=0.1, weight_decay=0.1)
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)
    optimizer.step()
    # Decay yields 2.97; m=1, v=1, and the adjusted learning rate is 0.1.
    torch.testing.assert_close(parameter, torch.tensor([2.97 - 0.1 / 1.1], dtype=torch.float64))


def test_adamw_closure_and_state_restoration():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    optimizer = AdamW([parameter], lr=0.05)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    torch.testing.assert_close(loss, torch.tensor(13.0))
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = AdamW([resumed_parameter])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for _ in range(3):
        for opt, param in ((optimizer, parameter), (resumed, resumed_parameter)):
            opt.zero_grad()
            param.square().sum().backward()
            opt.step()
        torch.testing.assert_close(parameter, resumed_parameter)
    assert resumed.state[resumed_parameter]["step"] == 4


def test_gradient_clipping_generators_missing_and_small_gradients():
    parameter = torch.nn.Parameter(torch.zeros(2))
    unused = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3.0, 4.0])
    gradient_clipping(iter([parameter, unused]), 1.0)
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 4.0]) / (5.0 + 1e-6))
    assert unused.grad is None
    unchanged = parameter.grad.clone()
    gradient_clipping(iter([parameter]), 2.0)
    torch.testing.assert_close(parameter.grad, unchanged, rtol=0, atol=0)
    parameter.grad.zero_()
    gradient_clipping([parameter], 0.0)
    assert torch.isfinite(parameter.grad).all()
    gradient_clipping([], 1.0)
    gradient_clipping([unused], 1.0)


def test_cosine_schedule_without_warmup():
    assert get_lr_cosine_schedule(0, 1.0, 0.1, 0, 10) == 1.0
    assert get_lr_cosine_schedule(5, 1.0, 0.1, 0, 10) == pytest.approx(0.55)
    assert get_lr_cosine_schedule(10, 1.0, 0.1, 0, 10) == 0.1
    assert get_lr_cosine_schedule(100, 1.0, 0.1, 0, 10) == 0.1


def test_custom_loss_and_optimizer_train_transformer():
    torch.manual_seed(32)
    model = TransformerLM(16, 4, 8, 1, 2, 16)
    optimizer = AdamW(model.parameters(), lr=0.03, betas=(0.9, 0.95))
    tokens = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    initial_weights = model.lm_head.weight.detach().clone()
    initial_loss = cross_entropy(model(tokens), targets).item()
    for step in range(12):
        optimizer.param_groups[0]["lr"] = get_lr_cosine_schedule(step, 0.03, 0.003, 2, 12)
        optimizer.zero_grad()
        loss = cross_entropy(model(tokens), targets)
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        final_loss = cross_entropy(model(tokens), targets)
    assert torch.isfinite(final_loss)
    assert final_loss.item() < initial_loss
    assert not torch.equal(initial_weights, model.lm_head.weight)
