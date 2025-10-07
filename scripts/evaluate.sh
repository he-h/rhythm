model_name=rhythm
export HF_TOKEN="your_huggingface_token"
export HF_HOME="your_huggingface_cache_directory"

# evaluate one model with a context length
python -u run.py \
  --task_name evaluation \
  --is_training 1 \
  --root_path ./dataset/yj \
  --model_id yj_336_48 \
  --model $model_name \
  --data YOUR_DATASET \
  --city YOUR_CITY \
  --seq_len 336 \
  --label_len 288 \
  --token_len 48 \
  --test_seq_len 336 \
  --test_label_len 288 \
  --test_pred_len 48 \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --mlp_hidden_layers 4 \
  --mlp_activation gelu \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --label_missing \
  --llm_ckp_dir 'meta-llama/Llama-3.2-1B' \
  --use_amp \
  --path YOUR_PATH
