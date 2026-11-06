#!/bin/bash
# Minimal reproduction wrapper for the spatial-attention training runs.
#
# Usage:
#   bash run.sh <attn> <seed> [--linear_kernel elu|relu|none]
#
# <attn> in: none | softmax | linear | windowed | performer | nystrom
#
# Example:  bash run.sh linear 1
#           bash run.sh linear 1 --linear_kernel relu
#
# Requires: data/dataset/{train.json, val.json, dict.pt} and the Python deps
# listed in ENVIRONMENT.md. Runs the full 10 epochs (--no_early_stop).

set -e -o pipefail

# Ensure `common_utils` (sibling package) is importable from this directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

ATTN="${1:?usage: run.sh <attn> <seed>}"
SEED="${2:?usage: run.sh <attn> <seed>}"
shift 2

python "$SCRIPT_DIR/behavior_clone/train_executor.py" \
  --train_dataset data/dataset/train.json \
  --val_dataset   data/dataset/val.json \
  --inst_dict_path data/dataset/dict.pt \
  --emb_field_dim 32 --prev_cmd_dim 64 \
  --num_conv_layers 3 --num_post_layers 1 \
  --conv_hid_dim 256 --army_out_dim 256 --other_out_dim 128 \
  --money_hid_layer 1 --conv_dropout 0.1 \
  --rnn_word_emb_dim 64 --word_emb_dropout 0.25 \
  --inst_hid_dim 128 --inst_encoder_type lstm \
  --batch_size 128 --gpu 0 --grad_clip 0.5 \
  --lr 0.002 --optim adamax --epochs 10 \
  --use_hist_inst 1 --pos_dim 16 --prev_cmd_rnn 1 \
  --top_num_inst -1 --seed "$SEED" \
  --attention_type "$ATTN" --attention_num_heads 8 --attention_dropout 0.1 \
  --no_early_stop \
  --model_folder "saved_models/ablation_${ATTN}_seed${SEED}" \
  "$@"
