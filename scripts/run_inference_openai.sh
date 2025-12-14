source ../llm-utils/setup/.venv/bin/activate
source configs/config.env
datasets=("writingprompts")
models=("gpt-4" "gpt-5-nano")
temperatures=(0.0)
# make topics a dict with keys being the model names and values being the topic names
declare -A topics
topics["gpt-4o-mini"]="default athelete"
topics["gpt-3.5-turbo"]="default athelete"
topics["gpt-5-mini"]="default athelete"
topics["gpt-5-nano"]="default athelete"
topics["gpt-5"]="default athelete"
topics["gpt-4"]="default athelete"

data_dir=$storage_dir/data/
max_new_tokens=1000
num_return_sequences=1
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
          --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every openai --batch_name ${dataset}_${topic}_${model_save_name}_${temperature}_train
        python infer.py --model_name $model_name --input_file $data_dir/$dataset/inference_input_${topic}_test.csv --output_file $data_dir/$dataset/inference_input_${topic}_test_${model_save_name}_${temperature}_output.jsonl \
          --max_new_tokens $max_new_tokens --temperature $temperature --num_return_sequences $num_return_sequences --do_sample --input_column question_input --checkpoint_every $checkpoint_every openai --batch_name ${dataset}_${topic}_${model_save_name}_${temperature}_test
      done
    done
  done
done

cd $current_dir
