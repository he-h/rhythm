export HF_TOKEN="your_huggingface_token"
export HF_HOME="your_huggingface_cache_directory"

context_length=336
prediction_length=48
interval=48

python data_provider/preprocess_data.py

python ./preprocess.py --gpu 0 --dataset yj --city B --llm_ckp_dir meta-llama/Llama-3.2-1B
python ./preprocess.py --gpu 0 --dataset yj --city C --llm_ckp_dir meta-llama/Llama-3.2-1B
python ./preprocess.py --gpu 0 --dataset yj --city D --llm_ckp_dir meta-llama/Llama-3.2-1B