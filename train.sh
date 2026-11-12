#!/bin/bash
# 学習スクリプト
# configs/config.yaml でデフォルト値を管理。CLI 引数で上書き可能。
# 使い方: bash train.sh <model>   (model: minipilot | nanopilot | picopilot | femtopilot)
set -euo pipefail

MODEL=${1:-minipilot}
case "$MODEL" in
  minipilot)  VIS=dinov3_vits;         BS=256 ;;
  nanopilot)  VIS=vit_tiny;            BS=512 ;;
  picopilot)  VIS=efficientformerv2_s0; BS=256 ;;
  femtopilot) VIS=mobilevit_xxs;       BS=256 ;;
  *) echo "unknown model: $MODEL"; exit 1 ;;
esac

OUT=./ckpts_${MODEL}
mkdir -p "$OUT"

python -u train.py \
  --config ./configs/config.yaml \
  --vision-encoder "$VIS" \
  --text-encoder minilm-l6 \
  --teacher bbox \
  --dataset all \
  --epochs 30 \
  --batch-size "$BS" \
  --num-workers 8 \
  --out "$OUT" \
  2>&1 | tee "$OUT/train_output.log"
