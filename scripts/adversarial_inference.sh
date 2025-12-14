source ../llm-utils/setup/.venv/bin/activate
source configs/config.env
datasets=("writingprompts")
base_models=("meta-llama/Llama-3.1-8B-Instruct")
topics=("athelete")
temperatures=(0.0)
ratio=0.25

data_dir=$storage_dir/data/
max_new_tokens=1000
num_return_sequences=1
batch_size=8
checkpoint_every=0.01
current_dir=$(pwd)

cd ../llm-utils
for model in "${base_models[@]}"; do
  model_abr="${model#*/}"
    for topic in "${topics[@]}"; do
    model_name="${model_abr}-adversarial-$topic-$ratio"
    for temperature in "${temperatures[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 
                echo "Performing inference on $dataset using $model_name with topic $topic at temperature $temperature"
                python infer.py --model_name $huggingface_repo_namespace/$model_name --input_file $data_dir/$dataset/curated/adversarial/dpo/${topic}_${model_abr}_${temperature}/remaining_hiding.csv --output_file $data_dir/$dataset/curated/adversarial/dpo/${topic}_${model_abr}_${temperature}/remaining_hiding_${ratio}_output.jsonl \
                --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every hf --batch_size $batch_size --padding_side left


                python infer.py --model_name $huggingface_repo_namespace/$model_name --input_file $data_dir/$dataset/inference_input_default_test.csv --output_file $data_dir/$dataset/inference_input_${topic}_test_${model_name}_${temperature}_output.jsonl \
                --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every hf --batch_size $batch_size --padding_side left
            done
        done
    done
done

cd $current_dir
