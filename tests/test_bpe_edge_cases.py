"""Regression checks beyond the supplied tokenizer examples."""

import random

import pytest

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer

from .test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path


@pytest.fixture(scope="module", params=[[], ["<|endoftext|>", "<|endoftext|><|endoftext|>"]])
def tokenizer(request):
    return get_tokenizer_from_vocab_merges_path(
        VOCAB_PATH, MERGES_PATH, special_tokens=request.param
    )


@pytest.mark.parametrize(
    "text",
    [
        "we're sure they'll say I've done it",
        "x!!'re 'll 've a\r\n  b\t \nend  ",
        "\n\nre ",
        "你好 👩🏽‍💻 e\u0301\x00 fin",
        "a<|endoftext|><|endoftext|>b<|endoftext|>",
    ],
)
def test_streaming_at_every_split(tokenizer, text):
    expected = tokenizer.encode(text)
    for index in range(len(text) + 1):
        assert list(tokenizer.encode_iterable([text[:index], "", text[index:]])) == expected
    assert list(tokenizer.encode_iterable(iter(text))) == expected
    assert tokenizer.decode(expected) == text


def test_random_stream_boundaries(tokenizer):
    rng = random.Random(336)
    fragments = ["x", " we're", "'", "re", "\n", " ", "\t", "你好", "🙂", "<|endoftext|>"]
    for _ in range(50):
        text = "".join(rng.choices(fragments, k=30))
        cuts = sorted({0, len(text), *(rng.randrange(len(text) + 1) for _ in range(20))})
        chunks = (text[start:end] for start, end in zip(cuts, cuts[1:]))
        assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)


def test_stream_is_lazy(tokenizer):
    def chunks():
        # Supply enough text to exceed the longest special-token lookahead.
        yield "one two three four five " * 10
        raise AssertionError("The tokenizer consumed the next chunk before yielding")

    stream = tokenizer.encode_iterable(chunks())
    assert next(stream) == tokenizer.encode("one")[0]
    stream.close()


def test_decode_combines_bytes_before_decoding():
    tokenizer = Tokenizer({index: bytes([index]) for index in range(256)}, [])
    assert tokenizer.decode([0xC3, 0xA9]) == "é"
    assert tokenizer.decode([0xC3]) == "�"
    assert tokenizer.decode([0xC3, 0x28]) == "�("
    assert tokenizer.decode([0]) == "\x00"


def test_saved_tokenizer_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello hello 你好 🙃<s>hello", encoding="utf-8")
    vocab, merges = train_bpe(corpus, 280, ["<s>"])
    original = Tokenizer(vocab, merges, ["<s>"])
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.json"
    original.save(vocab_path, merges_path)
    restored = Tokenizer.from_files(vocab_path, merges_path, ["<s>"])
    assert restored.vocab == original.vocab
    assert restored.merges == original.merges
    text = "你好 🙃<s>hello\x00"
    assert restored.encode(text) == original.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_training_ties_and_special_boundaries(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ab<s>ac", encoding="utf-8")
    vocab, merges = train_bpe(corpus, 258, ["<s>"])
    assert merges == [(b"a", b"c")]
    assert len(vocab) == 258
    assert b"<s>" in vocab.values()


def test_repeated_pairs_do_not_overlap(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("aaaaa", encoding="utf-8")
    vocab, merges = train_bpe(corpus, 270)
    assert merges == [(b"a", b"a"), (b"aa", b"aa"), (b"aaaa", b"a")]
    assert len(vocab) == 259
    tokenizer = Tokenizer(vocab, merges)
    assert len(tokenizer.encode("aaaaa")) == 1


def test_empty_training_and_small_vocabulary(tmp_path):
    corpus = tmp_path / "empty.txt"
    corpus.write_text("", encoding="utf-8")
    vocab, merges = train_bpe(corpus, 300, ["<s>", "<s>"])
    assert len(vocab) == 257
    assert merges == []
    with pytest.raises(ValueError, match="vocab_size"):
        train_bpe(corpus, 256, ["<s>"])
    with pytest.raises(ValueError, match="empty"):
        train_bpe(corpus, 300, [""])


def test_training_preserves_carriage_returns(tmp_path):
    corpus = tmp_path / "newlines.txt"
    corpus.write_bytes(b"\r\n\r\n")
    _, merges = train_bpe(corpus, 257)
    assert merges == [(b"\r", b"\n")]
