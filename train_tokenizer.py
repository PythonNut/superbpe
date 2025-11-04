"""
Train tokenizer on a single data category.
"""

import os
from pathlib import Path
import time
import json
import re
import random
import click
from utils import (
    ensure_dir,
    get_files_with_num_bytes,
    get_truncated_file,
    train_or_extend_tokenizer,
)

random.seed(0)


@click.command()
@click.option(
    "--output_dir",
    type=str,
    help="Where to save the trained tokenizer.",
)
@click.option(
    "--num_bytes",
    type=int,
    default=None,
    help="The maximum number of bytes to use for tokenizer training.",
)
@click.option(
    "--corpus_dir",
    type=str,
    default=None,
    help="Directory containing text files to use for training the tokenizer.",
)
@click.option(
    "--vocab_size",
    type=int,
    default=100000,
    help="The number of tokens in the vocabulary.",
)
@click.option(
    "--regex_string",
    type=str,
    default=None,
    help="Regex for pretokenization.",
)
def main(
    output_dir: str,
    num_bytes: int,
    corpus_dir: str,
    vocab_size: int,
    regex_string: str,
):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    print(f"We are training a tokenizer for {output_dir}", flush=True)

    # We look for merges.txt in the current dir to determine whether we are extending
    # the tokenizer or training from scratch, so we need to cd into the output directory.
    os.chdir(output_dir)

    if os.path.exists("meta.json"):
        print(
            "Output directory contains meta.json, so we will use the files from there."
        )
        meta = json.load(open("meta.json"))
        train_files, actual_num_bytes = meta["train_files"], meta["total_bytes"]
        for file in train_files:
            if not os.path.exists(file):
                assert "truncated" in file, f"{file} not found"
                wanted_filesize = int(re.search(r"_truncated_(\d+)", file).group(1))
                file = re.sub(r"_truncated_\d+", "", file)
                get_truncated_file(file, wanted_filesize)
    else:
        train_files, actual_num_bytes = get_files_with_num_bytes(corpus_dir, num_bytes)

        # Write metadata
        with open("meta.json", "w") as fo:
            meta = {}
            meta["total_bytes"] = actual_num_bytes
            meta["train_files"] = train_files
            if os.path.exists("merges.txt"):
                os.system("cp merges.txt initial_merges.txt")
                meta["num_initial_merges"] = (
                    sum(1 for line in open("initial_merges.txt")) - 1
                )
            json.dump(meta, fo, indent=5)

    # Train tokenizer
    start_time = time.time()

    print("Training with HF tokenizers...")
    tokenizer = train_or_extend_tokenizer(
        train_files,
        vocab_size=vocab_size,
        regex_string=regex_string,
    )
    tokenizer.model.save(".")  # saves merges.txt and vocab.json
    tokenizer.save("tokenizer.json")

    print(f"Train time: {time.time() - start_time}", flush=True)
    print("Tokenizer info saved to " + str(output_dir), flush=True)

    # Delete files that were constructed just for this
    # for f in train_files:
    #     if "truncated" in f:
    #         os.remove(f)


if __name__ == "__main__":
    main()
