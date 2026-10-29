"""Byte-level BPE tokenizer for Stanford CS336 Assignment 1.

Place this module at cs336_basics/tokenizer.py in the assignment repository.
Requires the `regex` package (included in the assignment's pyproject.toml).

The module provides:
    train_bpe(input_path, vocab_size, special_tokens)
    Tokenizer(vocab, merges, special_tokens).encode/decode/encode_iterable
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
import regex

# GPT-2's pre-tokenization rule, operating on Unicode strings before UTF-8 encoding.
GPT2_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)
PRETOKENIZER = regex.compile(GPT2_PATTERN)


def _special_pattern(special_tokens: list[str]) -> regex.Pattern | None:
    if not special_tokens:
        return None
    # Longest first means overlapping specials are recognized as one token.
    choices = sorted(set(special_tokens), key=lambda s: (-len(s), s))
    return regex.compile("|".join(regex.escape(s) for s in choices))


def _apply_merge(
    word: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """Merge all non-overlapping adjacent occurrences of `pair`, left-to-right."""
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Learn BPE merges and vocabulary from a UTF-8 text corpus.

    Pair-frequency counts are updated only for pre-tokens affected by a merge.
    This is faster than retokenizing the entire corpus each iteration.
    """
    special_tokens = special_tokens or []
    if vocab_size < 256 + len(set(special_tokens)):
        raise ValueError("vocab_size must fit all 256 byte tokens and special tokens")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in dict.fromkeys(special_tokens):
        vocab[len(vocab)] = token.encode("utf-8")

    # The special strings delimit the training data: never learn merges across them.
    corpus = Path(input_path).read_text(encoding="utf-8")
    splitter = _special_pattern(special_tokens)
    sections = splitter.split(corpus) if splitter else [corpus]
    token_counts: Counter[tuple[bytes, ...]] = Counter()
    for section in sections:
        for match in PRETOKENIZER.finditer(section):
            bs = match.group().encode("utf-8")
            token_counts[tuple(bytes([b]) for b in bs)] += 1
    del corpus, sections

    words = list(token_counts)
    frequencies = [token_counts[w] for w in words]
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for idx, word in enumerate(words):
        for pair, num in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += num * frequencies[idx]
            pair_to_words[pair].add(idx)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    for _ in range(num_merges):
        if not pair_counts:
            break

        # On ties, the lexicographically greatest pair wins (CS336 specification).
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected = pair_to_words.get(best_pair, set()).copy()
        for idx in affected:
            old = words[idx]
            count = frequencies[idx]
            for pair, n in Counter(zip(old, old[1:])).items():
                pair_counts[pair] -= n * count
                pair_to_words[pair].discard(idx)
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)

            new = _apply_merge(old, best_pair)
            words[idx] = new
            for pair, n in Counter(zip(new, new[1:])).items():
                pair_counts[pair] += n * count
                pair_to_words[pair].add(idx)

    return vocab, merges


class Tokenizer:
    """Encode/decode using a trained byte-level BPE vocabulary and merge ordering."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {value: key for key, value in self.vocab.items()}
        self.merges = list(merges)
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = special_tokens or []
        self.special_ids: dict[str, int] = {}
        for token in dict.fromkeys(self.special_tokens):
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id
            self.special_ids[token] = self.token_to_id[token_bytes]

        choices = sorted(set(self.special_tokens), key=lambda s: (-len(s), s))
        self.special_re = _special_pattern(choices)
        self._max_special_length = max(map(len, choices), default=0)

    def _pieces(self, text: str) -> Iterator[tuple[int, int, str, bool]]:
        """Yield (start, end, text, is_special), with specials protected first."""
        cursor = 0
        if self.special_re is not None:
            for special in self.special_re.finditer(text):
                prefix = text[cursor:special.start()]
                for match in PRETOKENIZER.finditer(prefix):
                    yield cursor + match.start(), cursor + match.end(), match.group(), False
                yield special.start(), special.end(), special.group(), True
                cursor = special.end()
        for match in PRETOKENIZER.finditer(text[cursor:]):
            yield cursor + match.start(), cursor + match.end(), match.group(), False

    def _encode_pretoken(self, piece: str) -> Iterator[int]:
        data = piece.encode("utf-8")
        tokens: tuple[bytes, ...] = tuple(bytes([b]) for b in data)
        while len(tokens) > 1:
            available = (
                (self.merge_rank[(tokens[i], tokens[i + 1])], (tokens[i], tokens[i + 1]))
                for i in range(len(tokens) - 1)
                if (tokens[i], tokens[i + 1]) in self.merge_rank
            )
            best = min(available, default=None)
            if best is None:
                break
            tokens = _apply_merge(tokens, best[1])
        for token in tokens:
            yield self.token_to_id[token]

    def _scan(self, text: str) -> Iterator[int]:
        for _, _, piece, is_special in self._pieces(text):
            if is_special:
                yield self.special_ids[piece]
            else:
                yield from self._encode_pretoken(piece)

    def encode(self, text: str) -> list[int]:
        return list(self._scan(text))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream text without loading the entire input into memory.

        Hold the last incomplete pre-token and enough lookbehind for a special
        token that might straddle two input chunks.
        """
        pending = ""
        for chunk in iterable:
            pending += chunk
            matches = list(self._pieces(pending))
            if len(matches) <= 1:
                continue
            # Keep the final pre-token (which can grow with the next chunk),
            # and enough suffix for specials crossing chunk boundaries.
            safe_end = max(0, len(pending) - self._max_special_length)
            cut = 0
            for match in matches[:-1]:
                if match[1] > safe_end:
                    break
                cut = match[1]
            if cut:
                yield from self._scan(pending[:cut])
                pending = pending[cut:]
        if pending:
            yield from self._scan(pending)


if __name__ == "__main__":
    from tempfile import NamedTemporaryFile
    print("Unicode / UTF-8:")
    sample = "Hello 你好 🙃"
    print(sample, "→", list(sample.encode("utf-8")))
    assert sample.encode("utf-8").decode("utf-8") == sample

    with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt") as f:
        f.write("low low lower lower hello 你好 🙃<|endoftext|>low")
        f.flush()
        vocab, merges = train_bpe(f.name, vocab_size=300, special_tokens=["<|endoftext|>"])
        tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
        print("First five merges:", merges[:5])
        ids = tokenizer.encode(sample + "<|endoftext|>")
        print("Encoded:", ids)
        print("Decoded:", tokenizer.decode(ids))
        assert tokenizer.decode(ids) == sample + "<|endoftext|>"
        assert ids[-1] == tokenizer.special_ids["<|endoftext|>"]
        for slices in ((sample[:5], sample[5:]), (sample[:8], sample[8:])):
            assert list(tokenizer.encode_iterable(slices)) == tokenizer.encode(sample)
        print("Self-check passed")
