#!/bin/bash
source configs/config.env
#module load gcc/13.3.0 cuda/12.6.3
source ../llm-utils/setup/.venv/bin/activate



do_clf(){
    current_dir=$PWD
    files_dir=$1
    train_file=$files_dir/clf_train.csv
    num_train_epochs=50
    batch_size=8
    lora_target_modules="q_proj, v_proj, k_proj, o_proj"
    learning_rate=0.0005
    name=$2$3
    n_train=$3
    n_train_points=${n_train:-10000000000000}


    # check that neither are empty
    if [ -z "$files_dir" ] || [ -z "$name" ]; then
        echo "Usage: do_clf <files_dir> <name>"
        return 1
    fi
    #model_name="meta-llama/Llama-3.1-8B"
    #model_name="meta-llama/Llama-3.2-3B"
    model_name="meta-llama/Llama-3.2-1B"
    #model_name="facebook/bart-large"
    #model_name="google-bert/bert-base-uncased"

    train_validation_split=0.85
    output_column="label"
    cd ../llm-utils
    echo "Doing run: $name"
    python train.py --training_kind clf --model_name $model_name --per_device_train_batch_size $batch_size --per_device_eval_batch_size $batch_size --use_peft False \
    --output_dir $storage_dir/models/clf/$name-$model_name --num_train_epochs $num_train_epochs --learning_rate $learning_rate --max_train_samples $n_train_points \
    --train_file $train_file --train_validation_split $train_validation_split --output_column $output_column \
    --logging_strategy epoch --eval_strategy epoch --save_strategy epoch --save_steps 0.5 --eval_steps 0.5 \
    --early_stopping_patience 2 \
    --load_best_model_at_end True --run_name clf-$name

    echo "Performing inference on the test sets"
    output_file=$files_dir/clf_test_${model_name}_${n_train}_output.jsonl
    python infer.py --model_name $storage_dir/models/clf/$name-$model_name/final_checkpoint --input_file $files_dir/clf_test.csv --output_file $output_file hf --model_kind clf
    cd $current_dir
    echo "Results for $name-$model_name with $n_train points"
    python << EOD
import pandas as pd
df = pd.read_json("$output_file", lines=True)
print("For: $name with output file: $output_file")
print("Grouped Score:")
print(df.groupby(["label"])['output'].mean().apply(lambda x: round(x, 2)*100))
print("Overall accuracy:", round((df['output'].round() == df['label']).mean(), 2)*100)
print("Confusion Matrix:")
confusion_matrix = pd.crosstab(df['label'], df['output'].round(), rownames=['Actual'], colnames=['Predicted'], margins=True)
print(confusion_matrix)
EOD
}
