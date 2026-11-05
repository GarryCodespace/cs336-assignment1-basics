# Unicode and byte-level BPE

The written Unicode answers are in [unicode.md](unicode.md).
`bpe.py` trains a vocabulary; `tokenizer.py` implements encoding, decoding,
special tokens, and streaming. The two tokenizer adapters in `tests/adapters.py`
connect these modules to the assignment tests.

From the repository root, train and save a small tokenizer:

```sh
uv run python -m cs336_basics.bpe tests/fixtures/corpus.en \
  --vocab-size 500 --special-token '<|endoftext|>' \
  --vocab-path /tmp/vocab.json --merges-path /tmp/merges.json
```

Load with `Tokenizer.from_files(vocab_path, merges_path, special_tokens)`.
The JSON vocabulary maps stringified IDs to hexadecimal byte strings, and the
merges JSON is an ordered list of pairs of hexadecimal strings. This format
preserves all byte values, including tokens that are not individually valid
UTF-8. It is not the GPT-2 fixture file format. Special-token configuration is
passed separately when loading.

Run the implementation checks:

```sh
uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py tests/test_bpe_edge_cases.py
```

Streaming encodes the concatenation of input chunks, including special tokens
and contractions split across chunks. It buffers an unfinished pre-token and
special-token lookahead, so a single unusually long pre-token can still require
a large buffer. Training currently reads the corpus into memory and runs in one
process; the separate full TinyStories/OpenWebText experiments have not been run.
