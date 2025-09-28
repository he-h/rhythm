export HF_TOKEN=''
export HF_HOME=""

python data_provider/preprocess_data.py
python ./preprocess.py --gpu 0 --dataset yj --city B --llm_ckp_dir meta-llama/Llama-3.2-1B
python ./preprocess.py --gpu 0 --dataset yj --city C --llm_ckp_dir meta-llama/Llama-3.2-1B
python ./preprocess.py --gpu 0 --dataset yj --city D --llm_ckp_dir meta-llama/Llama-3.2-1B