#!/bin/bash
source scripts/clf/basic.sh
datasets=("writingprompts")
topics=("athelete") # "politician" "war" "city" "philosophy")
lms=("Qwen3-0.6B" "Qwen3-1.7B" "Qwen3-4B" "Qwen3-8B" "Qwen3-14B" "Qwen3-32B")
#lms=("Llama-3.1-8B-Instruct" "Meta-Llama-3-70B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.2-1B-Instruct") # "Mistral-7B-Instruct-v0.2")
temperatures=(0.0)


for dataset in "${datasets[@]}"; do
    data_dir=$storage_dir/data/${dataset}/curated/same_topic/
    for topic in "${topics[@]}"; do
        for lm in "${lms[@]}"; do
        for temperature in "${temperatures[@]}"; do
            name="same_topic-prompting-${dataset}-${topic}-${lm}-${temperature}"
            files_dir=$data_dir/prompting/${topic}_${lm}_${temperature}/
            do_clf $files_dir $name
            done
        done
    done
done
