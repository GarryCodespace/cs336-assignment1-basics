"""Train a BPE tokenizer and stream text corpora to memory-mappable token IDs."""

import argparse
import hashlib
import json
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

from .bpe import save_bpe, train_bpe
from .tokenizer import Tokenizer

SPECIAL_TOKENS = ["<|endoftext|>"]
_worker_tokenizer = None


def _initialize_worker(vocab: dict, merges: list, special_tokens: list[str]) -> None:
    global _worker_tokenizer
    _worker_tokenizer = Tokenizer(vocab, merges, special_tokens)


def _encode_chunk(text: str) -> np.ndarray:
    dtype = "uint16" if max(_worker_tokenizer.vocab) <= np.iinfo(np.uint16).max else "uint32"
    return np.fromiter(_worker_tokenizer._scan(text), dtype=dtype)


def _document_chunks(source, marker: str):
    """Split only at a special-token boundary, preserving every character."""
    pending = ""
    while chunk := source.read(1024 * 1024):
        pending += chunk
        end = pending.rfind(marker)
        if end >= 0:
            end += len(marker)
            yield pending[:end]
            pending = pending[end:]
        elif len(pending) > 8 * 1024 * 1024:
            raise ValueError("Parallel encoding needs document delimiters; use --workers 1 for this corpus")
    if pending:
        yield pending


def _parallel_batches(source, tokenizer: Tokenizer, workers: int):
    chunks = iter(_document_chunks(source, "<|endoftext|>"))
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
                             initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens)) as pool:
        pending = deque(pool.submit(_encode_chunk, chunk) for chunk in islice(chunks, 2 * workers))
        while pending:
            yield pending.popleft().result()
            chunk = next(chunks, None)
            if chunk is not None:
                pending.append(pool.submit(_encode_chunk, chunk))


def _serial_batches(source, tokenizer: Tokenizer, dtype: np.dtype):
    ids = tokenizer.encode_iterable(iter(lambda: source.read(1024 * 1024), ""))
    while True:
        batch = np.fromiter(islice(ids, 1024 * 1024), dtype=dtype)
        if batch.size == 0:
            return
        yield batch


def encode_file(source_path: Path, output_path: Path, tokenizer: Tokenizer, workers: int = 1) -> dict:
    """Write token IDs incrementally; rename the partial file only on success."""
    dtype = np.dtype("uint16" if max(tokenizer.vocab) <= np.iinfo(np.uint16).max else "uint32")
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    count = 0
    started = time.monotonic()
    with source_path.open(encoding="utf-8", newline="") as source, partial.open("wb") as destination:
        if workers > 1:
            batches = _parallel_batches(source, tokenizer, workers)
        else:
            batches = _serial_batches(source, tokenizer, dtype)
        next_report = 1024 * 1024
        for batch in batches:
            if batch.size == 0:
                break
            batch.tofile(destination)
            count += len(batch)
            if count >= next_report:
                print(json.dumps({"encoding": source_path.name, "tokens": count,
                                  "seconds": round(time.monotonic() - started, 1)}), flush=True)
                next_report = count + 4 * 1024 * 1024
    partial.replace(output_path)
    return {"source": str(source_path.resolve()), "source_bytes": source_path.stat().st_size,
            "tokens": count, "dtype": dtype.name, "output": str(output_path.resolve()),
            "seconds": time.monotonic() - started}


def prepare(train_text: Path, valid_text: Path, output_dir: Path, vocab_size: int, workers: int = 1) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path, merges_path = output_dir / "vocab.json", output_dir / "merges.json"
    tokenizer_info_path = output_dir / "tokenizer.json"
    expected = {"requested_vocab_size": vocab_size, "special_tokens": SPECIAL_TOKENS,
                "training_source": str(train_text.resolve()), "training_bytes": train_text.stat().st_size}
    if tokenizer_info_path.exists():
        info = json.loads(tokenizer_info_path.read_text())
        if any(info.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing tokenizer settings differ; choose a new output directory")
    else:
        started = time.monotonic()
        print(json.dumps({"stage": "train_bpe", **expected}), flush=True)
        vocab, merges = train_bpe(train_text, vocab_size, SPECIAL_TOKENS,
                                  progress=lambda event: print(json.dumps(event), flush=True))
        save_bpe(vocab, merges, vocab_path, merges_path)
        info = {**expected, "vocab_size": len(vocab), "merges": len(merges),
                "training_seconds": time.monotonic() - started}
        tokenizer_info_path.write_text(json.dumps(info, indent=2) + "\n")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)
    fingerprint = hashlib.sha256(vocab_path.read_bytes() + merges_path.read_bytes()).hexdigest()
    results = {"tokenizer": info, "tokenizer_sha256": fingerprint}
    for split, source in (("train", train_text), ("valid", valid_text)):
        destination = output_dir / f"{split}.bin"
        metadata_path = output_dir / f"{split}.json"
        if destination.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if (metadata["source"] != str(source.resolve()) or metadata["source_bytes"] != source.stat().st_size
                    or metadata["tokenizer_sha256"] != fingerprint):
                raise ValueError("Existing encoded data differs; choose a new output directory")
        else:
            metadata = {**encode_file(source, destination, tokenizer, workers), "tokenizer_sha256": fingerprint}
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        results[split] = metadata
    (output_dir / "metadata.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.train_text, args.valid_text, args.output_dir, args.vocab_size, args.workers), indent=2), flush=True)


if __name__ == "__main__":
    main()
