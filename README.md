# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Train a small TinyStories model

The [completed Colab run](./examples/tinystories-colab/README.md) includes
generated stories, training metrics, and the model configuration.

To run the current implementation on a Colab GPU, upload
[notebooks/CS336_TinyStories_GPU.ipynb](./notebooks/CS336_TinyStories_GPU.ipynb)
to [Google Colab](https://colab.research.google.com/), select a T4 GPU, and run
all cells. The notebook bundles a snapshot of the local Python source, downloads
the data directly to Colab, prepares both full splits, trains on CUDA, and
downloads an archive containing the checkpoint, tokenizer, metrics, and samples.
The archive excludes the large text and binary token files. Download the archive
before disconnecting the runtime; runtime files are temporary. Local source
changes made after the notebook was created are not included in its snapshot.

Start with the two TinyStories files above; OpenWebText can wait until later.
Train the BPE tokenizer on the training text only, then encode both splits:

```sh
uv run python -m cs336_basics.prepare_data \
  --train-text data/TinyStoriesV2-GPT4-train.txt \
  --valid-text data/TinyStoriesV2-GPT4-valid.txt \
  --output-dir data/tinystories --vocab-size 4096
```

Preparation streams the text and writes raw `uint16` token files. Completed
artifacts are reused on subsequent runs. Corpus sizes, token counts, and the
tokenizer fingerprint are recorded in `data/tinystories/metadata.json`.

```sh
uv run python -m cs336_basics.train \
  --train-data data/tinystories/train.bin \
  --valid-data data/tinystories/valid.bin \
  --config configs/tinystories_small.json \
  --output-dir runs/tinystories
```

The example uses a 128-dimensional, four-layer Transformer and automatically
selects CUDA, MPS, or CPU. It stops after 6,000 updates or about ten minutes of
training, whichever comes first. Edit the JSON to change model, optimizer,
schedule, evaluation, checkpoint, and time-limit settings. Set `max_seconds` to
`null` to run to the update limit.

Each update samples contiguous input windows and targets shifted by one token,
sets the scheduled learning rate, clears gradients, computes cross-entropy,
backpropagates, clips gradients, and updates parameters with AdamW. Validation
uses fixed held-out windows without disturbing the training sampler.

Metrics are printed and appended to `runs/tinystories/metrics.jsonl`. Atomic
checkpoints at `runs/tinystories/latest.pt` contain the model, optimizer, completed
step, configuration, and sampler state. Resume with the same command plus
`--resume runs/tinystories/latest.pt`; `num_steps` is the total target, including
completed updates. The saved sampler and step preserve the training sequence
and learning-rate schedule.

Generate text from the resulting checkpoint:

```sh
uv run python -m cs336_basics.generate \
  --checkpoint runs/tinystories/latest.pt \
  --tokenizer-dir data/tinystories \
  --prompt "Once upon a time" \
  --max-new-tokens 200 --temperature 0.8 --top-p 0.95 \
  --output runs/tinystories/samples.json
```

Check data loading, checkpointing, training, and generation with:

```sh
uv run pytest tests/test_data.py tests/test_serialization.py \
  tests/test_train_loop.py tests/test_prepare_generate.py -v
```

