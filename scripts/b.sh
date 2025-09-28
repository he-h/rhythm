model_name=rhythm
export HF_TOKEN=''
export HF_HOME=""

# training one model with a context length
python -u run.py \
  --task_name hm_classification \
  --is_training 1 \
  --root_path ./dataset/yj \
  --model_id yj_336_48 \
  --model $model_name \
  --data yj \
  --city B \
  --seq_len 336 \
  --label_len 288 \
  --token_len 48 \
  --test_seq_len 336 \
  --test_label_len 288 \
  --test_pred_len 48 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --mlp_hidden_layers 4 \
  --mlp_activation gelu \
  --train_epochs 30 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --label_missing \
  --llm_ckp_dir 'meta-llama/Llama-3.2-1B' \
  --weight_decay 1e-3 \
  --enable_early_stopping \
  --grad_clip \
  --use_amp \