dataset_name=olmo2_p99_truncate
vocab_size=128000
num_bytes=$((10**10))
regex_string="[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
corpus_dir=/gscratch/xlab/alisaliu/pretokenization/data/${dataset_name}/train  # a directory containing txt files for tokenizer training

# convert num_bytes to something like 10G or 100M, depending on the value
if [ $num_bytes -ge $((10**9)) ]; then
    num_bytes_str=$(($num_bytes / 10**9))G
elif [ $num_bytes -ge $((10**6)) ]; then
    num_bytes_str=$(($num_bytes / 10**6))M
elif [ $num_bytes -ge $((10**3)) ]; then
    num_bytes_str=$(($num_bytes / 10**3))K
else
    num_bytes_str=${num_bytes}
fi

# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

output_dir=tokenizer_json/bpe_${dataset_name}_${num_bytes_str}_${vocab_size_str}
echo "output_dir: $output_dir"

python -m train_tokenizer \
    --output_dir $output_dir \
    --corpus_dir $corpus_dir \
    --num_bytes $num_bytes \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
