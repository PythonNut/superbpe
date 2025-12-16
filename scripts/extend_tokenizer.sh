dataset_name=olmo2_p99_truncate
orig_tokenizer_dir=tokenizer_json/bpe_${dataset_name}_10G_128K
num_inherit_merges=100000
vocab_size=128000
regex_string="\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"

# create a str called num_inherit_merges_str, which turns 100000 into 100K
if [ ${num_inherit_merges:-0} -ge 1000 ]; then
    num_inherit_merges_str=$(($num_inherit_merges / 1000))K
else
    num_inherit_merges_str=${num_inherit_merges:-0}
fi

# convert vocab_size to something like 100K, depending on the value
if [ $vocab_size -ge 1000 ]; then
    vocab_size_str=$(($vocab_size / 1000))K
else
    vocab_size_str=${vocab_size}
fi

output_dir=tokenizer_json/superbpe_${dataset_name}_10G_${num_inherit_merges_str}_extend_${vocab_size_str}
echo "output_dir: $output_dir"

mkdir -p $output_dir
head -n $num_inherit_merges $orig_tokenizer_dir/merges.txt > $output_dir/merges.txt
# this line copies over the training data. if you wish to use a different training corpus, delete this line and provide --corpus_dir.
cp $orig_tokenizer_dir/meta.json $output_dir/meta.json

python -m train_tokenizer \
    --output_dir $output_dir \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
