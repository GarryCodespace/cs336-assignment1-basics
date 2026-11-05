"""Sample next-token batches, save checkpoints, and train a Transformer LM."""

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch

from .model import TransformerLM
from .optimizer import AdamW, cross_entropy, get_lr_cosine_schedule, gradient_clipping


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str | torch.device, rng: np.random.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample contiguous windows with targets shifted one token forward."""
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("dataset must be a one-dimensional integer array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset needs at least context_length + 1 tokens")
    if rng is None:
        starts = np.random.randint(len(dataset) - context_length, size=batch_size)
    else:
        starts = rng.integers(len(dataset) - context_length, size=batch_size)
    windows = np.stack([dataset[start:start + context_length + 1] for start in starts])
    tokens = torch.tensor(windows, dtype=torch.long, device=device)
    return tokens[:, :-1], tokens[:, 1:]


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO, extra_state: dict[str, Any] | None = None) -> None:
    """Save model/optimizer state and completed steps, atomically for paths."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "extra_state": extra_state or {},
    }
    if hasattr(out, "write"):
        torch.save(checkpoint, out)
        return
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(checkpoint, handle)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(src: str | os.PathLike | BinaryIO, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Restore model/optimizer states onto their devices and return the step."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def load_tokens(path: str | Path, dtype: str = "uint16") -> npt.NDArray:
    """Memory-map a .npy array or a raw binary token file of the given dtype."""
    path = Path(path)
    dataset = np.load(path, mmap_mode="r", allow_pickle=False) if path.suffix == ".npy" else np.memmap(path, dtype=dtype, mode="r")
    if dataset.ndim != 1 or not np.issubdtype(dataset.dtype, np.integer):
        raise ValueError("Token files must contain a one-dimensional integer array")
    return dataset


@dataclass
class TrainConfig:
    vocab_size: int = 4096
    context_length: int = 128
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 384
    rope_theta: float = 10000.0
    batch_size: int = 16
    num_steps: int = 3000
    max_learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_iters: int = 50
    cosine_cycle_iters: int = 3000
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "auto"
    seed: int = 42
    num_threads: int = 4
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 250
    max_seconds: float | None = None

    def __post_init__(self) -> None:
        positive = (self.vocab_size, self.context_length, self.d_model, self.num_layers, self.num_heads, self.d_ff,
                    self.batch_size, self.num_steps, self.num_threads, self.log_interval, self.eval_interval,
                    self.eval_batches, self.checkpoint_interval)
        if any(value <= 0 for value in positive):
            raise ValueError("Model dimensions, batch size, steps, and intervals must be positive")
        if self.max_grad_norm < 0 or (self.max_seconds is not None and self.max_seconds <= 0):
            raise ValueError("Invalid gradient norm or time limit")
        get_lr_cosine_schedule(0, self.max_learning_rate, self.min_learning_rate, self.warmup_iters, self.cosine_cycle_iters)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: str, num_batches: int = 10, seed: int = 0) -> float:
    """Evaluate fixed held-out windows without changing training RNG or mode."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    try:
        losses = []
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device, rng)
            losses.append(cross_entropy(model(inputs), targets).item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def train(train_dataset: npt.NDArray, valid_dataset: npt.NDArray | None, config: TrainConfig, output_dir: str | Path, resume: str | Path | None = None) -> tuple[TransformerLM, list[dict[str, Any]]]:
    """Train up to num_steps total updates, optionally resuming a checkpoint."""
    torch.set_num_threads(config.num_threads)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    for dataset in (train_dataset, valid_dataset):
        if dataset is not None and (dataset.ndim != 1 or len(dataset) <= config.context_length):
            raise ValueError("Datasets must be one-dimensional and longer than context_length")
    rng = np.random.default_rng(config.seed)
    model = TransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers,
                          config.num_heads, config.d_ff, config.rope_theta, device=device)
    optimizer = AdamW(model.parameters(), lr=config.max_learning_rate, betas=(config.beta1, config.beta2),
                      eps=config.eps, weight_decay=config.weight_decay)
    completed = 0
    previous_elapsed = 0.0
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        extra = checkpoint.get("extra_state", {})
        previous_config = extra.get("config", {})
        runtime_fields = {"device", "num_steps", "max_seconds", "num_threads", "log_interval",
                          "eval_interval", "eval_batches", "checkpoint_interval"}
        for name, value in asdict(config).items():
            if name not in runtime_fields and name in previous_config and previous_config[name] != value:
                raise ValueError(f"Cannot resume with changed {name}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed = checkpoint["iteration"]
        if "numpy_rng" in extra:
            rng.bit_generator.state = extra["numpy_rng"]
        if "torch_rng" in extra:
            torch.set_rng_state(extra["torch_rng"])
        previous_elapsed = extra.get("elapsed_seconds", 0.0)
    if completed > config.num_steps:
        raise ValueError("Checkpoint step exceeds num_steps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(json.dumps({"device": device, "parameters": sum(p.numel() for p in model.parameters()),
                      "start_step": completed, "target_steps": config.num_steps}), flush=True)
    history = []
    model.train()
    started = time.monotonic()
    start_step = completed

    def checkpoint_now() -> None:
        save_checkpoint(model, optimizer, completed, output_dir / "latest.pt", extra_state={
            "config": asdict(config), "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "elapsed_seconds": previous_elapsed + time.monotonic() - started,
        })

    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
        while completed < config.num_steps:
            lr = get_lr_cosine_schedule(completed, config.max_learning_rate, config.min_learning_rate,
                                        config.warmup_iters, config.cosine_cycle_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs, targets = get_batch(train_dataset, config.batch_size, config.context_length, device, rng)
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {completed}")
            loss.backward()
            gradient_clipping(model.parameters(), config.max_grad_norm)
            optimizer.step()
            completed += 1
            elapsed = time.monotonic() - started
            done = completed == config.num_steps or (config.max_seconds is not None and elapsed >= config.max_seconds)
            should_eval = valid_dataset is not None and (completed % config.eval_interval == 0 or done)
            if completed == 1 or completed % config.log_interval == 0 or should_eval or done:
                row = {"step": completed, "train_loss": loss.item(), "lr": lr,
                       "elapsed_seconds": previous_elapsed + elapsed,
                       "tokens_processed": completed * config.batch_size * config.context_length,
                       "tokens_per_second": (completed - start_step) * config.batch_size * config.context_length / max(elapsed, 1e-6)}
                if should_eval:
                    row["validation_loss"] = evaluate(model, valid_dataset, config.batch_size, config.context_length,
                                                      device, config.eval_batches, seed=config.seed + 1)
                    row["validation_perplexity"] = math.exp(min(row["validation_loss"], 80))
                history.append(row)
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                print(json.dumps(row), flush=True)
            if completed % config.checkpoint_interval == 0 or done:
                checkpoint_now()
            if done:
                break
    if completed == start_step:
        checkpoint_now()
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--data-dtype", choices=["uint16", "uint32", "int32", "int64"], default="uint16")
    parser.add_argument("--config", type=Path, help="JSON with TrainConfig fields")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    for name in ("device", "num_steps", "max_seconds"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    config = TrainConfig(**values)
    train_data = load_tokens(args.train_data, args.data_dtype)
    valid_data = load_tokens(args.valid_data, args.data_dtype) if args.valid_data else None
    train(train_data, valid_data, config, args.output_dir, args.resume)


if __name__ == "__main__":
    main()
