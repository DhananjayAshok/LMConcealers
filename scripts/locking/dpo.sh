source configs/config.env
current_dir=$(pwd)
cd ../llm-utils
source setup/.venv/bin/activate
#topics=("athelete" "politician" "war" "city" "philosophy")
#models=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2")
topics=("athelete")
models=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")

for topic in "${topics[@]}"; do
    for model in "${models[@]}"; do
        model_name="${model#*/}"
        echo "Creating locked model for topic: $topic using base model: $model"
        python train.py --training_kind dpo --model_name $model \
        --output_dir $storage_dir/models/$model_name-locked-$topic-dpo \
        --train_file $storage_dir/data/locking_datasets/$topic/po_train.csv \
        --train_validation_split 0.9 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
        --learning_rate 1e-4 \
        --logging_strategy epoch --logging_steps 0.1 \
        --eval_strategy epoch --eval_steps 0.5 \
        --save_strategy epoch --save_steps 0.5 \
        --early_stopping_patience 5 \
        --load_best_model_at_end True \
        --run_name $model_name-locked-$topic-dpo --hub_model_id $model_name-locked-$topic-dpo --push_to_hub True

        python infer.py --input_file $storage_dir/data/locking_datasets/$topic/po_test.csv \
        --output_file $storage_dir/data/locking_datasets/$topic/po_test_${model_name}_dpo_output.jsonl \
        --model_name $storage_dir/models/$model_name-locked-$topic-dpo/final_checkpoint \
        --output_column model_output --max_new_tokens 200 \
        --ignore_checkpoint hf --batch_size 8 --padding_side left
    done
done
cd $current_dir