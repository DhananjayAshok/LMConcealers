#!/bin/bash
source scripts/clf/basic.sh
datasets=("writingprompts")
topics=("athelete" "politician" "war" "city" "philosophy")
lms=("Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2" "gemma-3-12b-it") # phi-4
temperatures=(0.0)


for dataset in "${datasets[@]}"; do
    data_dir=$storage_dir/data/${dataset}/curated/different_topic/
    for lm in "${lms[@]}"; do
        for topic in "${topics[@]}"; do
            for temperature in "${temperatures[@]}"; do
                name="different_topic-prompting-$topic-${dataset}-${lm}-${temperature}"
                files_dir=$data_dir/prompting/${topic}_${lm}_${temperature}/
                do_clf $files_dir $name
                output_file=$files_dir/clf_test_meta-llama/Llama-3.2-1B__output.jsonl
                python << EOD
import pandas as pd
df = pd.read_json("$output_file", lines=True)
print("For: $name with output file: $output_file")
print("Grouped Scores:") # grouping by topic
for topic in df['topic'].unique():
    topic_df = df[df['topic'] == topic]
    print(f"Topic: {topic}")
    print(topic_df.groupby(["label"])['output'].mean().apply(lambda x: round(x, 2)*100))
    print("Overall accuracy:", round((topic_df['output'].round() == topic_df['label']).mean(), 2)*100)
    print("Confusion Matrix:")
    confusion_matrix = pd.crosstab(topic_df['label'], topic_df['output'].round(), rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(confusion_matrix)
EOD
            done
        done
    done
done
