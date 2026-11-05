import json

import numpy as np
import pytest
import torch

from cs336_basics.generate import generate
from cs336_basics.model import TransformerLM
from cs336_basics.prepare_data import prepare
from cs336_basics.tokenizer import Tokenizer


def test_prepare_tokenizes_both_splits_and_reuses_complete_files(tmp_path):
    training_text = "Once upon a time.\r\n<|endoftext|>We're happy.你好!<|endoftext|>"
    validation_text = "A dog ran.\n<|endoftext|>"
    train_text, valid_text = tmp_path / "train.txt", tmp_path / "valid.txt"
    train_text.write_bytes(training_text.encode())
    valid_text.write_bytes(validation_text.encode())
    output_dir = tmp_path / "encoded"
    results = prepare(train_text, valid_text, output_dir, 280)
    tokenizer = Tokenizer.from_files(output_dir / "vocab.json", output_dir / "merges.json", ["<|endoftext|>"])
    for split, text in (("train", training_text), ("valid", validation_text)):
        tokens = np.memmap(output_dir / f"{split}.bin", dtype=results[split]["dtype"], mode="r")
        assert tokenizer.decode(tokens.tolist()) == text
        assert tokens.tolist() == tokenizer.encode(text)
        assert len(tokens) == results[split]["tokens"]
    before = (output_dir / "train.bin").stat().st_mtime_ns
    assert prepare(train_text, valid_text, output_dir, 280) == results
    assert (output_dir / "train.bin").stat().st_mtime_ns == before
    assert json.loads((output_dir / "metadata.json").read_text()) == results
    parallel_dir = tmp_path / "parallel"
    prepare(train_text, valid_text, parallel_dir, 280, workers=2)
    for split in ("train", "valid"):
        assert (parallel_dir / f"{split}.bin").read_bytes() == (output_dir / f"{split}.bin").read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        prepare(train_text, valid_text, output_dir, 281)


def test_generation_length_context_crop_and_mode():
    torch.manual_seed(51)
    model = TransformerLM(16, 2, 8, 1, 2, 16)
    model.train()
    prompt = [0, 1, 2, 3]
    result = generate(model, prompt, max_new_tokens=3, temperature=0)
    assert result[:4] == prompt
    assert len(result) == 7
    assert prompt == [0, 1, 2, 3]
    assert model.training
    assert all(0 <= token < 16 for token in result)
    assert all(parameter.grad is None for parameter in model.parameters())
    first_token = result[4]
    assert generate(model, prompt, max_new_tokens=5, temperature=0, eos_id=first_token) == result[:5]
    torch.manual_seed(52)
    first_sample = generate(model, prompt, max_new_tokens=3, top_p=0.5)
    torch.manual_seed(52)
    assert generate(model, prompt, max_new_tokens=3, top_p=0.5) == first_sample
