"""Unicode-safe byte-level BPE encoding, decoding, and streaming."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .bpe import BYTE_TOKENS, _apply_merge, _iter_pieces, _pieces, _special_pattern, save_bpe


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {value: key for key, value in self.vocab.items()}
        self.merges = list(merges)
        self.merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self.special_re = _special_pattern(self.special_tokens)
        self.special_ids = {}
        for token in self.special_tokens:
            encoded = token.encode("utf-8")
            if encoded not in self.token_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = encoded
                self.token_to_id[encoded] = new_id
            self.special_ids[token] = self.token_to_id[encoded]
        self._max_special_length = max(map(len, self.special_tokens), default=0)
        self._pretoken_cache: dict[str, tuple[int, ...]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """Load the hex-encoded JSON format written by save_bpe() or save()."""
        vocab_data = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merge_data = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        vocab = {int(key): bytes.fromhex(value) for key, value in vocab_data.items()}
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merge_data]
        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str | Path, merges_filepath: str | Path) -> None:
        """Save vocabulary and merges; pass special_tokens again when loading."""
        save_bpe(self.vocab, self.merges, vocab_filepath, merges_filepath)

    def _pieces(self, text: str) -> Iterator[tuple[int, str, bool]]:
        """Yield (end offset, text, is_special), protecting special tokens first."""
        yield from _pieces(text, self.special_re)

    def _encode_pretoken(self, piece: str) -> Iterator[int]:
        cached = self._pretoken_cache.get(piece)
        if cached is not None:
            yield from cached
            return
        tokens = tuple(BYTE_TOKENS[value] for value in piece.encode("utf-8"))
        while len(tokens) > 1:
            candidates = (pair for pair in zip(tokens, tokens[1:]) if pair in self.merge_rank)
            pair = min(candidates, key=self.merge_rank.__getitem__, default=None)
            if pair is None:
                break
            tokens = _apply_merge(tokens, pair)
        ids = tuple(self.token_to_id[token] for token in tokens)
        if len(self._pretoken_cache) >= 65536:
            self._pretoken_cache.clear()
        self._pretoken_cache[piece] = ids
        yield from ids

    def _scan(self, text: str) -> Iterator[int]:
        for _, piece, is_special in self._pieces(text):
            if is_special:
                yield self.special_ids[piece]
            else:
                yield from self._encode_pretoken(piece)

    def encode(self, text: str) -> list[int]:
        return list(self._scan(text))

    def decode(self, ids: list[int]) -> str:
        # UTF-8 characters can span token boundaries. Decode only after joining.
        return b"".join(self.vocab[index] for index in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode the concatenated stream without collecting the whole input.

        Retain two trailing pre-tokens: a partial contraction such as "'r" can
        become "'re" in the next chunk. Also retain enough text for a special
        token split across chunks. Memory grows with the longest unfinished
        pre-token, rather than with the entire corpus.
        """
        for piece, is_special in _iter_pieces(iterable, self.special_tokens):
            if is_special:
                yield self.special_ids[piece]
            else:
                yield from self._encode_pretoken(piece)
