"""Generate text from a trained Transformer checkpoint and its BPE tokenizer."""

import argparse
import json
from pathlib import Path

import torch

from .model import TransformerLM, softmax
from .tokenizer import Tokenizer
from .train import TrainConfig, resolve_device


@torch.no_grad()
def generate(model: TransformerLM, prompt_ids: list[int], max_new_tokens: int = 200, temperature: float = 0.8, top_p: float = 0.95, eos_id: int | None = None) -> list[int]:
    """Sample tokens with temperature/top-p filtering and an optional EOS stop."""
    if not prompt_ids or max_new_tokens < 0 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Require a prompt, nonnegative length/temperature, and 0 < top_p <= 1")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    try:
        for _ in range(max_new_tokens):
            inputs = torch.tensor([ids[-model.context_length:]], device=device, dtype=torch.long)
            logits = model(inputs)[0, -1].float()
            if temperature == 0:
                next_id = logits.argmax()
            else:
                probabilities = softmax(logits / temperature)
                sorted_probs, sorted_ids = probabilities.sort(descending=True)
                sorted_probs = sorted_probs.masked_fill(sorted_probs.cumsum(-1) - sorted_probs >= top_p, 0)
                sorted_probs /= sorted_probs.sum()
                next_id = sorted_ids[torch.multinomial(sorted_probs, 1).squeeze(0)]
            ids.append(next_id.item())
            if ids[-1] == eos_id:
                break
        return ids
    finally:
        model.train(was_training)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = TrainConfig(**checkpoint["extra_state"]["config"])
    torch.set_num_threads(config.num_threads)
    device = resolve_device(args.device)
    model = TransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers,
                          config.num_heads, config.d_ff, config.rope_theta, device=device)
    model.load_state_dict(checkpoint["model"])
    info = json.loads((args.tokenizer_dir / "tokenizer.json").read_text())
    tokenizer = Tokenizer.from_files(args.tokenizer_dir / "vocab.json", args.tokenizer_dir / "merges.json", info["special_tokens"])
    if len(tokenizer.vocab) != config.vocab_size:
        raise ValueError("Tokenizer vocabulary size differs from the trained model")
    samples = []
    for prompt in args.prompt or ["Once upon a time", "One day, a little girl named Lily", "Tom and his dog"]:
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [tokenizer.special_ids["<|endoftext|>"]]
        ids = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_p,
                       tokenizer.special_ids.get("<|endoftext|>"))
        sample = {"prompt": prompt, "text": tokenizer.decode(ids), "generated_tokens": len(ids) - len(prompt_ids)}
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
