#!/bin/bash
source scripts/clf/basic.sh
datasets=("writingprompts")
topics=("athelete")
lms=("Llama-3.1-8B-Instruct")
temperatures=(0.0)
token_counts=(10 50 100 500 1000)


for dataset in "${datasets[@]}"; do
    data_dir=$storage_dir/data/${dataset}/curated/token_ablation/
    for topic in "${topics[@]}"; do
        for lm in "${lms[@]}"; do
        for temperature in "${temperatures[@]}"; do
            for token_count in "${token_counts[@]}"; do
                name="token-ablation-${dataset}-${topic}-${lm}-${temperature}"
                files_dir=$data_dir/${topic}_${lm}_${temperature}/tokens_${token_count}/
                do_clf $files_dir $name
                done
            done
        done
    done
done
