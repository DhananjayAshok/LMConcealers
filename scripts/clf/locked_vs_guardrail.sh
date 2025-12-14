#!/bin/bash
source scripts/clf/basic.sh
datasets=("writingprompts")
topics=("athelete") # "politician" "war" "city" "philosophy")
lms=("Llama-3.1-8B-Instruct") # "Mistral-7B-Instruct-v0.2")
temperatures=(0.0)


for dataset in "${datasets[@]}"; do
    data_dir=$storage_dir/data/${dataset}/curated/locked_vs_guardrail/
    for topic in "${topics[@]}"; do
        for lm in "${lms[@]}"; do
        for temperature in "${temperatures[@]}"; do
            name="locked_vs_guardrail-prompting-${dataset}-${topic}-${lm}-${temperature}"
            files_dir=$data_dir/prompting/${topic}_${lm}_${temperature}/
            do_clf $files_dir $name
            done
        done
    done
done
