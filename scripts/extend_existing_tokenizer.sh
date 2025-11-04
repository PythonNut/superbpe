model_name=gpt-4  # download from https://huggingface.co/Xenova/gpt-4o
dataset_name=olmo2_p99_truncate
orig_tokenizer_dir=tokenizer_json/$model_name
num_inherit_merges=100000  # leave this unspecified if extending the entire tokenizer
vocab_size=128000
num_bytes=$((10**10))
regex_string="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# create a str called num_inherit_merges_str, which turns 100000 into 100K
if [ $num_inherit_merges -ge 1000 ]; then
    num_inherit_merges_str=$(($num_inherit_merges / 1000))K
else
    num_inherit_merges_str=${num_inherit_merges}
fi

# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

output_dir=tokenizer_json/${model_name}_superbpe_${dataset_name}_10G_${num_inherit_merges_str}_extend_${vocab_size_str}
echo "output_dir: $output_dir"

mkdir -p $output_dir
if [ -n "$num_inherit_merges" ]; then
    head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt
else
    cp $orig_tokenizer_dir/merges.txt $output_dir/merges.txt
fi

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --corpus_dir $corpus_dir \
    --num_bytes $num_bytes \
    --regex_string "$regex_string"
