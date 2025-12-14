source ../llm-utils/setup/.venv/bin/activate
source configs/config.env
datasets=("writingprompts")
#models=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "google/gemma-3-12b-it" "microsoft/phi-4")
models=("google/gemma-3-27b-it" "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")
#temperatures=(0.0 0.5 1.0 1.5 2.0)
#models=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Meta-Llama-3-70B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-1B-Instruct")
temperatures=(0.0)
# make topics a dict with keys being the model names and values being the topic names
declare -A topics
topics["meta-llama/Llama-3.1-8B-Instruct"]="default athelete politician war city philosophy"
topics["google/gemma-3-27b-it"]="default athelete"
topics["Qwen/Qwen3-0.6B"]="default athelete"
topics["Qwen/Qwen3-1.7B"]="default athelete"
topics["Qwen/Qwen3-4B"]="default athelete"
topics["Qwen/Qwen3-8B"]="default athelete"
topics["Qwen/Qwen3-14B"]="default athelete"
topics["Qwen/Qwen3-32B"]="default athelete"
topics["meta-llama/Meta-Llama-3-70B-Instruct"]="default athelete"
topics["meta-llama/Llama-3.2-3B-Instruct"]="default athelete"
topics["meta-llama/Llama-3.2-1B-Instruct"]="default athelete"
topics["mistralai/Mistral-7B-Instruct-v0.2"]="default athelete politician war city philosophy"
topics["google/gemma-3-12b-it"]="default athelete politician war city philosophy"
topics["microsoft/phi-4"]="default athelete politician war city philosophy"

data_dir=$storage_dir/data/
max_new_tokens=1000
num_return_sequences=1
batch_size=32
checkpoint_every=0.01
current_dir=$(pwd)

cd ../llm-utils
for model in "${models[@]}"; do
  model_name=$model
  model_save_name="${model_name#*/}"
  for temperature in "${temperatures[@]}"; do
    for topic in ${topics[$model_name]}; do
      for dataset in "${datasets[@]}"; do
        echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" 
        echo "Performing inference on $dataset using $model_name with topic $topic at temperature $temperature"
        python infer.py --model_name $model_name --input_file $data_dir/$dataset/inference_input_${topic}_train.csv --output_file $data_dir/$dataset/inference_input_${topic}_train_${model_save_name}_${temperature}_output.jsonl \
          --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every hf --batch_size $batch_size --padding_side left
        python infer.py --model_name $model_name --input_file $data_dir/$dataset/inference_input_${topic}_test.csv --output_file $data_dir/$dataset/inference_input_${topic}_test_${model_save_name}_${temperature}_output.jsonl \
          --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every hf --batch_size $batch_size --padding_side left
      done
    done
  done
done

cd $current_dir
