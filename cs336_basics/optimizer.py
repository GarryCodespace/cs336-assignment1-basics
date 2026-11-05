"""Loss, AdamW updates, and learning-rate/gradient controls for training."""

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch

from .model import softmax

__all__ = ["softmax", "cross_entropy", "AdamW", "get_lr_cosine_schedule", "gradient_clipping"]


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average negative log likelihood over all leading batch dimensions."""
    logits = inputs.float() if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    """Adam with bias-corrected moments and decoupled weight decay."""

    def __init__(self, params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        if lr < 0:
            raise ValueError("lr must be nonnegative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0 or weight_decay < 0:
            raise ValueError("eps and weight_decay must be nonnegative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]

                parameter.mul_(1 - lr * group["weight_decay"])
                first_moment.mul_(beta1).add_(grad, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                denominator = second_moment.sqrt().add_(group["eps"])
                parameter.addcdiv_(first_moment, denominator, value=-step_size)
        return loss


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    """Linear warmup, cosine decay to the final iteration, then a fixed floor."""
    if it < 0 or warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require it >= 0 and 0 <= warmup_iters < cosine_cycle_iters")
    if not 0 <= min_learning_rate <= max_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate")
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the global gradient norm in place, skipping missing gradients."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be nonnegative")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    norms = [
        torch.linalg.vector_norm(grad if grad.dtype == torch.float64 else grad.float()).to(gradients[0].device)
        for grad in gradients
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(norms))
    scale = (max_l2_norm / (total_norm + 1e-6)).clamp(max=1)
    for grad in gradients:
        grad.mul_(scale.to(grad.device))
