"""Train byte-level BPE with the assignment's GPT-2 pre-tokenization rule."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import regex

GPT2_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
PRETOKENIZER = regex.compile(GPT2_PATTERN)
BYTE_TOKENS = tuple(bytes([value]) for value in range(256))


def _special_pattern(special_tokens: list[str]) -> regex.Pattern | None:
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty")
    if not special_tokens:
        return None
    choices = sorted(set(special_tokens), key=lambda token: (-len(token), token))
    return regex.compile("|".join(regex.escape(token) for token in choices))


def _apply_merge(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Replace non-overlapping occurrences of a pair, from left to right."""
    merged = []
    index = 0
    while index < len(word):
        if index + 1 < len(word) and (word[index], word[index + 1]) == pair:
            merged.append(word[index] + word[index + 1])
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def _pieces(text: str, special_re: regex.Pattern | None) -> Iterator[tuple[int, str, bool]]:
    """Yield pre-tokens and specials with their ending character offsets."""
    cursor = 0
    if special_re is not None:
        for special in special_re.finditer(text):
            for match in PRETOKENIZER.finditer(text[cursor:special.start()]):
                yield cursor + match.end(), match.group(), False
            yield special.end(), special.group(), True
            cursor = special.end()
    for match in PRETOKENIZER.finditer(text[cursor:]):
        yield cursor + match.end(), match.group(), False


def _iter_pieces(iterable: Iterable[str], special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    """Pre-tokenize a stream while retaining enough context across chunks."""
    special_re = _special_pattern(special_tokens)
    max_special_length = max(map(len, special_tokens), default=0)
    pending = ""
    for chunk in iterable:
        for start in range(0, len(chunk), 16384):
            pending += chunk[start:start + 16384]
            pieces = list(_pieces(pending, special_re))
            safe_end = max(0, len(pending) - max_special_length)
            cut = 0
            for end, piece, is_special in pieces[:-2]:
                if end > safe_end:
                    break
                yield piece, is_special
                cut = end
            if cut:
                pending = pending[cut:]
    for _, piece, is_special in _pieces(pending, special_re):
        yield piece, is_special


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    progress: Callable[[dict[str, int]], None] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train on UTF-8 text, excluding special tokens from merge statistics.

    Only words containing the selected pair need their pair counts updated.
    Training stops early if there are no more pairs to merge.
    """
    special_tokens = list(dict.fromkeys(special_tokens or []))
    _special_pattern(special_tokens)
    vocab = dict(enumerate(BYTE_TOKENS))
    for token in special_tokens:
        encoded = token.encode("utf-8")
        if encoded not in BYTE_TOKENS:
            vocab[len(vocab)] = encoded
    if vocab_size < len(vocab):
        raise ValueError("vocab_size must fit all 256 bytes and the special tokens")

    # Preserve CR/LF bytes rather than applying universal-newline translation.
    token_counts: Counter[bytes] = Counter()
    with open(input_path, encoding="utf-8", newline="") as source:
        def chunks() -> Iterator[str]:
            next_report = 64 * 1024 * 1024
            while chunk := source.read(1024 * 1024):
                yield chunk
                if progress is not None and source.tell() >= next_report:
                    progress({"bytes_counted": source.tell()})
                    next_report = source.tell() + 64 * 1024 * 1024

        token_counts.update(piece.encode("utf-8") for piece, is_special in _iter_pieces(chunks(), special_tokens) if not is_special)

    words = [tuple(BYTE_TOKENS[value] for value in word) for word in token_counts]
    frequencies = list(token_counts.values())
    del token_counts
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for index, word in enumerate(words):
        for pair, occurrences in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += occurrences * frequencies[index]
            pair_to_words[pair].add(index)

    merges = []
    while len(vocab) < vocab_size and pair_counts:
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        for index in pair_to_words[best_pair].copy():
            old_word = words[index]
            frequency = frequencies[index]
            for pair, occurrences in Counter(zip(old_word, old_word[1:])).items():
                pair_counts[pair] -= occurrences * frequency
                pair_to_words[pair].discard(index)
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                    del pair_to_words[pair]

            new_word = _apply_merge(old_word, best_pair)
            words[index] = new_word
            for pair, occurrences in Counter(zip(new_word, new_word[1:])).items():
                pair_counts[pair] += occurrences * frequency
                pair_to_words[pair].add(index)
        if progress is not None and len(merges) % 256 == 0:
            progress({"merges": len(merges), "vocab_size": len(vocab)})

    return vocab, merges


def save_bpe(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str | Path,
    merges_filepath: str | Path,
) -> None:
    """Save JSON using hex strings, which preserve even invalid UTF-8 tokens.

    Vocabulary: {"token_id": "hex_bytes"}; merges: [["left_hex", "right_hex"], ...].
    Load these files with Tokenizer.from_files().
    """
    Path(vocab_filepath).write_text(
        json.dumps({str(key): value.hex() for key, value in vocab.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(merges_filepath).write_text(
        json.dumps([[left.hex(), right.hex()] for left, right in merges], indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    args = parser.parse_args()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_token)
    save_bpe(vocab, merges, args.vocab_path, args.merges_path)
    print(f"Saved {len(vocab)} vocabulary entries and {len(merges)} merges")
