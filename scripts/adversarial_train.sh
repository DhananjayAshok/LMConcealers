source configs/config.env
current_dir=$(pwd)
cd ../llm-utils
source setup/.venv/bin/activate
topics=("athelete")
models=("meta-llama/Llama-3.1-8B-Instruct")
ratio=0.25
for topic in "${topics[@]}"; do
    for model in "${models[@]}"; do
        model_name="${model#*/}"
        echo "Creating adversarial model for topic: $topic using base model: $model"
        python train.py --training_kind dpo --model_name $model \
        --output_dir $storage_dir/models/$model_name-adversarial-$topic-$ratio \
        --train_file $storage_dir/data/adversarial_locking_datasets/$model_name/$topic/0.0/$ratio/po_train.csv \
        --train_validation_split 0.9 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
        --learning_rate 1e-4 \
        --logging_strategy epoch --logging_steps 0.1 \
        --eval_strategy epoch --eval_steps 0.5 \
        --save_strategy epoch --save_steps 0.5 \
        --early_stopping_patience 5 \
        --load_best_model_at_end True \
        --run_name $model_name-adversarial-$topic-$ratio --hub_model_id $model_name-adversarial-$topic-$ratio --push_to_hub True
    done
done
cd $current_dir