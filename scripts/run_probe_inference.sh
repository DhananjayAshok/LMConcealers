source ../llm-utils/setup/.venv/bin/activate
source configs/config.env
models=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-3-12b-it" "microsoft/phi-4" "meta-llama/Meta-Llama-3-70B-Instruct")
#temperatures=(0.0 0.5 1.0 1.5 2.0)
#models=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct")
temperatures=(0.0)
# make topics a dict with keys being the model names and values being the topic names
declare -A topics
topics="default athelete politician war city philosophy"


data_dir=$storage_dir/data/
max_new_tokens=1000
num_return_sequences=1
batch_size=32
checkpoint_every=0.01
current_dir=$(pwd)
dataset="probe"

cd ../llm-utils
for model in "${models[@]}"; do
  model_name=$model
  model_save_name="${model_name#*/}"
  for temperature in "${temperatures[@]}"; do
    for topic in ${topics[@]}; do
        echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 
        echo "Performing inference on $dataset using $model_name with topic $topic at temperature $temperature"
        python infer.py --model_name $model_name --input_file $data_dir/$dataset/inference_input_${topic}.csv --output_file $data_dir/$dataset/inference_input_${topic}_${model_save_name}_${temperature}_output.jsonl \
          --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every hf --batch_size $batch_size --padding_side left
    done
  done
done

cd $current_dir
