import io
from dataclasses import replace

import numpy as np
import pytest
import torch

from cs336_basics.model import Linear
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.train import (
    TrainConfig,
    evaluate,
    get_batch,
    load_checkpoint,
    load_tokens,
    save_checkpoint,
    train,
)


def test_batch_smallest_dataset_and_readonly_memmaps(tmp_path):
    tokens = np.array([10, 20, 30, 40, 50], dtype=np.uint16)
    raw = tmp_path / "tokens.bin"
    tokens.tofile(raw)
    np.save(tmp_path / "tokens.npy", tokens)
    for path in (raw, tmp_path / "tokens.npy"):
        dataset = load_tokens(path)
        assert isinstance(dataset, np.memmap)
        inputs, targets = get_batch(dataset, 2, 4, "cpu")
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.tolist() == [[10, 20, 30, 40]] * 2
        assert targets.tolist() == [[20, 30, 40, 50]] * 2
        np.testing.assert_array_equal(dataset, tokens)
    with pytest.raises(ValueError, match="context_length"):
        get_batch(tokens[:4], 2, 4, "cpu")


def test_checkpoint_filelike_resumes_optimizer_updates():
    torch.manual_seed(41)
    model = Linear(3, 2)
    optimizer = AdamW(model.parameters())
    x = torch.randn(2, 3)
    model(x).square().mean().backward()
    optimizer.step()
    checkpoint = io.BytesIO()
    save_checkpoint(model, optimizer, 1, checkpoint)
    restored = Linear(3, 2)
    restored_optimizer = AdamW(restored.parameters())
    checkpoint.seek(0)
    assert load_checkpoint(checkpoint, restored, restored_optimizer) == 1
    for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        current_optimizer.zero_grad()
        current_model(x).square().mean().backward()
        current_optimizer.step()
    torch.testing.assert_close(model.weight, restored.weight, rtol=0, atol=0)


def test_training_resume_matches_uninterrupted_run(tmp_path):
    dataset = np.tile(np.arange(16, dtype=np.uint16), 8)
    config = TrainConfig(vocab_size=16, context_length=4, d_model=8, num_layers=1, num_heads=2,
                         d_ff=16, batch_size=2, num_steps=6, warmup_iters=1, cosine_cycle_iters=6,
                         device="cpu", num_threads=1, log_interval=1, eval_interval=2,
                         eval_batches=2, checkpoint_interval=2)
    full, _ = train(dataset, dataset, config, tmp_path / "full")
    train(dataset, dataset, replace(config, num_steps=3), tmp_path / "resumed")
    resumed, history = train(dataset, dataset, config, tmp_path / "resumed", tmp_path / "resumed/latest.pt")
    for name, parameter in full.state_dict().items():
        torch.testing.assert_close(parameter, resumed.state_dict()[name], rtol=0, atol=0)
    assert [row["step"] for row in history] == [4, 5, 6]
    for row in history:
        assert row["lr"] == get_lr_cosine_schedule(row["step"] - 1, config.max_learning_rate,
                                                  config.min_learning_rate, 1, 6)
    assert "validation_loss" in history[-1]
    saved = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert saved["iteration"] == 6
    assert (tmp_path / "resumed/config.json").is_file()


def test_evaluation_preserves_mode_rng_and_gradients(tmp_path):
    dataset = np.tile(np.arange(16, dtype=np.uint16), 8)
    config = TrainConfig(vocab_size=16, context_length=4, d_model=8, num_layers=1, num_heads=2,
                         d_ff=16, batch_size=2, num_steps=1, warmup_iters=0, cosine_cycle_iters=2,
                         device="cpu", num_threads=1, eval_batches=1)
    model, _ = train(dataset, None, config, tmp_path)
    model.train()
    rng_before = torch.get_rng_state().clone()
    gradients = [p.grad.clone() for p in model.parameters()]
    first = evaluate(model, dataset, 2, 4, "cpu", num_batches=2)
    assert first == evaluate(model, dataset, 2, 4, "cpu", num_batches=2)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient)
