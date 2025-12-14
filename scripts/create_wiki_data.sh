source setup/.venv/bin/activate
source configs/config.env
python data.py setup_entity_contexts
file_path=$storage_dir/data/wiki_entities/all_question_input.csv
current_dir=$(pwd)
cd ../llm-utils
source setup/.venv/bin/activate
python infer.py --model_name gpt-4o --max_new_tokens 2000 --input_file $file_path openai --batch_name all_entities
cd $current_dir
source setup/.venv/bin/activate