#!/bin/sh

set -e

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: expected virtual environment Python at $PYTHON_BIN" >&2
  echo "Create it with: python -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

"$PYTHON_BIN" -m scripts.train_tagalog_lm \
  --local-csv data/tagalog_corpus_test/train.csv \
  --csv-text-column text \
  --max-samples 50 \
  --block-size 64 \
  --stride 64 \
  --batch-size 8 \
  --epochs 100 \
  --embed-dim 64 \
  --num-heads 4 \
  --num-layers 6 \
  --ff-multiplier 2 \
  --learning-rate 5e-4 \
  --output-dir models/language_model_test \
  --sample-length 50 \
  --sample-prompt "Ang Pilipinas" \
  --log-interval 5 \
  --device cuda \
  --tokenizer-name jcblaise/roberta-tagalog-base