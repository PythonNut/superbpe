# SuperBPE: Space Travel for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2503.13423-b31b1b.svg)](https://arxiv.org/pdf/2503.13423) [![website](https://img.shields.io/badge/Website-superbpe.github.io-C16C8A)](https://superbpe.github.io/) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Collection-FFD21E)](https://huggingface.co/collections/UW/superbpe-67db2338062faa07c7473ffa)

This repository contains instructions for training SuperBPE tokenizers, along with analysis notebooks and the pretraining configs used in the paper.

For model developers who wish to experiment quickly with an off-the-shelf tokenizer in their pretraining pipeline, we have released an English [SuperBPE tokenizer](https://huggingface.co/alisawuffles/superbpe-tokenizer-128k) with a vocab size of 128K. Nonetheless, we highly encourage you to train your own SuperBPE tokenizer to customize it to your use case!

## Setup
First, clone the project with:
```bash
git clone --recurse-submodules https://github.com/PythonNut/superbpe.git
```
We use a custom [fork](https://github.com/alisawuffles/tokenizers-superbpe) of [huggingface/tokenizers](https://github.com/huggingface/tokenizers) which conflicts with the original.
Because of this, we recommend *always installing this project in its own virtual environment.*

### Setup virtual environment

#### Using `conda`
```bash
conda create -n superbpe python=3.12 rust
conda activate superbpe
pip install -r requirements.txt
```

#### Using `venv`
You will need to [install rust](https://www.rust-lang.org/tools/install) and Python 3.12.
Then, you can do:
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data download
Our tokenizer training data is available [here](https://huggingface.co/datasets/UW/olmo-mix-1124-subset-p99).
You can download it using [`huggingface-cli`](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (after logging into your HuggingFace account) using:
```
mkdir -p data/olmo2_p99_truncate
cd data/olmo2_p99_truncate
huggingface-cli download UW/olmo-mix-1124-subset-p99 --repo-type dataset --local-dir .
```

## Tokenizer training
Training a SuperBPE tokenizer involves two stages:

1. **Stage 1:** Learn subwords by enforcing whitespace pretokenization (equivalent to regular BPE training). See [scripts/train_tokenizer.sh](https://github.com/PythonNut/superbpe/blob/main/scripts/train_tokenizer.sh).
2. **Stage 2:** Learn superwords by resuming tokenizer training, but this time skip the whitespace pretokenization step. See [scripts/extend_tokenizer.sh](https://github.com/PythonNut/superbpe/blob/main/scripts/extend_tokenizer.sh).

Note that you can choose to use different training data for Stage 1 and Stage 2, or perform Stage 2 directly on top of an existing BPE tokenizer to augment it with superwords ([scripts/extend_existing_tokenizer.sh](https://github.com/PythonNut/superbpe/blob/main/scripts/extend_existing_tokenizer.sh)).

After tokenizer training, you'll want to use the [`construct_hf_tokenizer()` function](https://github.com/PythonNut/superbpe/blob/main/utils.py#L188) to construct a HuggingFace tokenizer with all the bells and whistles. It will create an EOS token, update the `tokenizer.json` with a `decoder` field, and construct additional files like `tokenizer_config.json` and `special_tokens_map.json`. You can modify this function depending on your own model development pipeline.

## Citation 

If you found this codebase helpful, please cite

```
@inproceedings{liu-etal-2025-superbpe,
  title={{SuperBPE}: Space travel for language models},
  author={Alisa Liu and Jonathan Hayase and Valentin Hofmann and Sewoong Oh and Noah A Smith and Yejin Choi},
  booktitle={Second Conference on Language Modeling},
  year={2025},
  url={https://arxiv.org/abs/2503.13423}
}
```
