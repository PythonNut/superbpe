from __future__ import annotations

import os
import random
from pathlib import Path
from filelock import FileLock

import simdjson as json
from tqdm import tqdm
from tokenizers.models import BPE, Unigram

from tokenizers import Tokenizer, pre_tokenizers, Regex
from tokenizers.pre_tokenizers import ByteLevel, Split, Digits
from tokenizers.trainers import BpeTrainer, UnigramTrainer


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def read_json(file):
    return json.load(open(file))


def read_merges_txt(path_to_txt):
    with open(path_to_txt) as fin:
        merges = fin.readlines()[1:]
        merges = [m.rsplit("\n", 1)[0] for m in merges]
    return merges


def get_pretokenization_regex(tokenizer_json):
    if isinstance(tokenizer_json, str):
        tokenizer_json = read_json(tokenizer_json)

    split_pretokenizer = [
        p
        for p in tokenizer_json["pre_tokenizer"]["pretokenizers"]
        if p["type"] == "Split"
    ][0]
    pretok_regex = split_pretokenizer["pattern"]["Regex"]
    return pretok_regex


def train_or_extend_tokenizer(
    text_files: str,
    vocab_size: int = 100000,
    regex_string: str = None,
):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(show_progress=True, vocab_size=vocab_size)

    print(f"Using regex string: {regex_string}", flush=True)
    pretokenizers = [
        Split(
            pattern=Regex(regex_string),
            behavior="isolated",
            invert=False,
        ),
        ByteLevel(
            add_prefix_space=False,
            trim_offsets=True,
            use_regex=False,
        ),
    ]
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(pretokenizers)
    tokenizer.train(text_files, trainer)

    return tokenizer


def bytes_to_unicode():
    """
    MJ: STOLEN DIRECTLY FROM https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
    --------------
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def is_valid_unicode(data):
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def get_truncated_file(filepath, wanted_filesize):
    """
    Create a copy of the given file and truncates it to the desired size.
    """
    if os.path.getsize(filepath) < wanted_filesize:
        raise ValueError("File is already smaller than desired filesize")

    filename, ext = os.path.splitext(filepath)
    truncated_filepath = Path(os.path.dirname(filepath)) / (
        f"{filename}_truncated_{wanted_filesize}{ext}"
    )

    # we want to make sure that multiple scripts don't create a truncated file at the same time
    lock = FileLock(str(truncated_filepath) + ".lock")
    with lock:
        if not os.path.exists(truncated_filepath):
            print(f"Truncating {filepath} to {wanted_filesize} bytes")

            os.system(f"cp {filepath} {truncated_filepath}")

            # adjust wanted_filesize to the next valid unicode character
            with open(truncated_filepath, "rb") as f:
                f.seek(wanted_filesize)
                data = f.read(1)
                while data and not is_valid_unicode(data):
                    data = f.read(1)
                    wanted_filesize += 1

            with open(truncated_filepath, "r+", encoding="utf-8") as fin:
                fin.truncate(wanted_filesize)
        else:
            print(f"Truncated file already exists: {truncated_filepath}")

    return str(truncated_filepath), wanted_filesize


def get_files_with_num_bytes(data_dir, num_bytes=None, loop_around=True):
    """Return a list of files inside data_dir that contain num_bytes worth of data."""
    file_list, byte_count = [], 0
    data_dir = Path(data_dir)

    all_files = [
        f
        for f in os.listdir(data_dir)
        if f.endswith(".txt") and ("truncated" not in f) and ("split" not in f)
    ]

    if not num_bytes:  # if num_bytes is not specified, use all text data
        file_list = [str(data_dir / f) for f in all_files]
        byte_count = sum(os.path.getsize(data_dir / f) for f in all_files)
        print(f"Using all {len(file_list)} files in {data_dir}")
    else:
        random.shuffle(all_files)
        counter = 0
        tqdm_bar = tqdm(total=num_bytes, desc="Loading text data")
        while byte_count < num_bytes:
            fname = all_files[counter % len(all_files)]
            filesize = os.path.getsize(data_dir / fname)
            if byte_count + filesize <= num_bytes:
                file_list.append(str(data_dir / fname))
                byte_count += filesize
                tqdm_bar.update(filesize)
            else:
                wanted_filesize = num_bytes - byte_count
                truncated_filepath, true_filesize = get_truncated_file(
                    data_dir / fname, wanted_filesize
                )
                file_list.append(truncated_filepath)
                byte_count += true_filesize
                tqdm_bar.update(true_filesize)
            counter += 1
            if not loop_around and counter >= len(all_files):
                break
    return file_list, byte_count
