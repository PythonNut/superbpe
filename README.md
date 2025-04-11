# SuperBPE: Space Travel for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.13423-b31b1b.svg)](https://arxiv.org/pdf/2503.13423) [![website](https://img.shields.io/badge/Website-superbpe.github.io-C16C8A)](https://superbpe.github.io/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FFD21E)](https://huggingface.co/collections/UW/superbpe-67db2338062faa07c7473ffa)

This repository contains the tokenizer training code. Code for other aspects of the project (e.g. evals, model scaling, data processing, wandb, train configs) will be added soon!

## Setup

### Python version requirement

> This project requires **Python 3.9–3.12**.
> `pysimdjson` is not yet compatible with Python 3.13 or above.

We recommend using a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Rust (for tokenizer build)
Our tokenizer depends on a Rust-based extension. You must install the [Rust toolchain](https://www.rust-lang.org/tools/install) first:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
```

### Install custom tokenizers fork
**Important:** We depend on a [custom fork](https://github.com/alisawuffles/tokenizers-superbpe) of [huggingface/tokenizers](https://github.com/huggingface/tokenizers), which conflicts with the original.

You can install it via:

```bash
git submodule update --init --recursive
pip install -r requirements.txt
```

## Data download
Our tokenizer training data is available [here](https://huggingface.co/datasets/UW/olmo-mix-1124-subset-p99). You can download it with the following command (after installing [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) and logging into your HuggingFace account).

```
mkdir olmo-mix-1124-subset-p99
cd olmo-mix-1124-subset-p99
huggingface-cli download UW/olmo-mix-1124-subset-p99 --repo-type dataset --local-dir .
```

## Tokenizer training
Training a SuperBPE tokenizer involves two stages:

1. **Stage 1:** Learn subwords by enforcing whitespace pretokenization (equivalent to regular BPE training).

```bash
python -m train_tokenizer \
    --output_dir tokenizers/olmo2_bpe \
    --corpus_dir olmo-mix-1124-subset-p99/train \
    --num_bytes $((10**10)) \
    --vocab_size 200000 \
    --do_whitespace_pretokenization true
```

2. **Stage 2:** Learn superwords by resuming tokenizer training, but this time skip the whitespace pretokenization step.

```bash
orig_tokenizer_dir=tokenizers/olmo2_bpe
num_inherit_merges=180000
output_dir=tokenizers/olmo2_superbpe

mkdir -p $output_dir

# inherit the first num_inherit_merges from the BPE tokenizer
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt

# specifies the same training files used in stage 1
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size 200000 \
    --do_whitespace_pretokenization false
```
