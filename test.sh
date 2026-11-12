#!/bin/bash
# 評価スクリプト: 指定チェックポイントを全データセットで評価する
# 使い方: bash test.sh <ckpt_path>   (例: bash test.sh ./ckpts_minipilot/best_model.pth)
# vision / text encoder はチェックポイントに保存された学習時の設定から自動で復元される。
set -euo pipefail

CKPT=${1:?usage: bash test.sh <ckpt_path>}

for DS in refcoco refcoco+ refcocog vg gqa; do
  python -u test.py \
    --config ./configs/config_tiny.yaml \
    --ckpt "$CKPT" \
    --no-steervit \
    --dataset "$DS"
done
