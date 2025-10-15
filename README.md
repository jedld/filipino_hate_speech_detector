# PyTorch Transformer Sentiment Analysis Project

This project provides a minimal, best-practices PyTorch setup for training a small transformer-based model for sentiment analysis. It includes:

- A sample dataset (`data/sample_data.csv`)
- Model, training, and inference scripts
- Reproducibility and code organization best practices

## Structure

- `data/` — Sample dataset
- `models/` — Model definition
- `scripts/` — Training and inference scripts
- `utils/` — Utility functions (tokenizer, dataset, etc.)
- `requirements.txt` — Dependencies
- `README.md` — Project overview and instructions

## Quickstart

1. Install dependencies (use the project virtual environment to pick up the pinned NumPy 1.26 stack):
   ```bash
   pip install -r requirements.txt
   ```
- The requirements file pins NumPy to `<2.0` and includes compatible versions of `pandas`, `pyarrow`, and `datasets`. Activate the project's virtual environment (or create a fresh one) before installing to avoid clashes with a system-wide NumPy 2.x installation.
2. Train the sample model:
   ```bash
   python scripts/train.py
   ```
3. Run inference:
   ```bash
   python scripts/infer.py --text "I love this!"
   ```

---

Replace the sample data with your own for real use cases.

## Dataset preparation

To download and normalize the requested datasets into `data/`:

```bash
python scripts/prepare_datasets.py --output-dir data --verbose
```

This will create:
- `data/hatespeech_filipino/{raw,processed}` from Hugging Face `jcblaise/hatespeech_filipino`
- `data/filipino_tiktok_hatespeech/{raw,processed}` from GitHub `imperialite/filipino-tiktok-hatespeech`
- `data/combined/processed/{train,validation,test}.csv` merging both sources with the same split ratios (GitHub rows are re-split alongside Hugging Face).

To override the default Hugging Face train/validation/test proportions, provide ratios that sum to 1.0:

```bash
python scripts/prepare_datasets.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --split-seed 123
```

The script will recombine the normalized Hugging Face data, resample it with the given ratios, and record the split sizes in `dataset_preparation_report.json`. The merged combined dataset follows the same ratios so downstream training can point entirely at `data/combined/processed/` if desired.

Processed CSV files follow the schema:

```
text,label
...string...,0|1
```

Notes:
- Label normalization uses simple heuristics mapping various class names to binary 0/1.
- A JSON report is saved at `data/dataset_preparation_report.json` summarizing actions and any issues.

### Dataset diagnostics

Inspect distribution and text length statistics for the processed dataset with:

```bash
python -m scripts.dataset_stats
```

Pass explicit CSV paths to analyze alternative splits:

```bash
python -m scripts.dataset_stats data/hatespeech_filipino/processed/train.csv data/hatespeech_filipino/processed/validation.csv
```

## Training on processed datasets

Once `prepare_datasets.py` has finished, launch the higher-capacity training routine:

```bash
python -m scripts.train_processed --include-github --mixed-precision
```

Key features:
- Uses the Hugging Face train/validation/test splits (`data/hatespeech_filipino/processed/*.csv`). Custom ratios supplied during preparation are respected. You can swap in the merged combined splits from `data/combined/processed/` for broader coverage.
- Optionally augments the training split with the processed GitHub dataset via `--include-github`.
- Transformer defaults tuned for stronger performance (4 layers, 128-d embeddings, dropout, AdamW, cosine LR, early stopping).
- Saves outputs under `models/processed/`:
   - `sentiment_transformer.pt` (best validation-accuracy checkpoint + optimizer state, with embedded model hyperparameters)
   - `sentiment_transformer_last.pt` (latest epoch checkpoint for resuming interrupted runs)
   - `tokenizer.json` (vocabulary + config)
   - `metrics.json` (training history, validation, and test metrics)
      - TensorBoard event files under `runs/processed/<timestamp>/`
   - Best-checkpoint metadata stores the transformer hyperparameters, so `scripts.infer` and `scripts.gradio_app` automatically reuse the training configuration.

Override hyperparameters (epochs, learning rate, batch size, etc.) with CLI flags. Use `--help` for the full list.

   To monitor training with TensorBoard:

   ```bash
   tensorboard --logdir runs/processed
   ```

   Use `--log-dir` to customize the base path or `--no-tensorboard` to disable logging if desired.

   ## Interactive UI

   Launch an interactive Gradio interface that loads the best available checkpoint and lets you score arbitrary text:

   ```bash
   python -m scripts.gradio_app
   ```

   Optional flags:

   - `--share` to request a temporary public Gradio URL
   - `--model-path` / `--tokenizer-path` to point to custom artifacts
   - `--test "your text"` to run a single prediction and exit without starting the UI

   The interface reports both class probabilities (hate vs. not hate) and highlights the predicted label.

## Tagalog generative language model

Train a compact Transformer language model that reuses a state-of-the-art subword tokenizer (defaults to `jcblaise/roberta-tagalog-base`) on an open Tagalog corpus (defaults to the OSCAR `unshuffled_deduplicated_tl` subset):

```bash
python -m scripts.train_tagalog_lm --output-dir models/language_model --epochs 3 --mixed-precision
```

Key options:
- `--max-samples` limits the number of documents pulled from the Hugging Face dataset (default 100k).
- `--block-size`, `--stride`, and model depth flags (`--embed-dim`, `--num-heads`, etc.) control model capacity.
- `--sample-prompt` and friends customise the post-training sample written to `<output-dir>/sample.txt`.
- `--tokenizer-name` swaps in a different Hugging Face tokenizer (any `AutoTokenizer`-compatible identifier).
- For offline experimentation, point the trainer at a local CSV column instead: `--local-csv data/combined/processed/train.csv --csv-text-column text`.

Artifacts land in the chosen output directory:
- `tagalog_lm.pt` — best-validation checkpoint with optimizer state and hyperparameters.
- `tokenizer/` — Hugging Face tokenizer folder (vocabulary, merges, special tokens).
- `language_model_metrics.json` — per-epoch losses/perplexities plus dataset provenance.
- `sample.txt` — generated continuation from the configured prompt.

### Preparing a Tagalog corpus locally

When working offline or with a custom snapshot of the Tagalog corpus, use the helper script to pull textual assets from the original BERT Tagalog repository and, if needed, fall back to the OSCAR dataset:

```bash
python -m scripts.download_tagalog_dataset \
   --output-dir data/tagalog_corpus \
   --max-samples 200000 \
   --val-ratio 0.05
```

The script will:

- Download the textual files (e.g., `vocab.txt`) from `jcblaise/bert-tagalog-base-uncased`.
- If the repository does not expose a large enough corpus, automatically backfill with the OSCAR Tagalog subset.
- Write `train.csv`, `val.csv`, and `all_texts.txt` under the requested output directory alongside a `metadata.json` summary.

You can then point the language-model trainer at the generated CSV (the tokenizer will still be pulled from Hugging Face unless overridden):

```bash
python -m scripts.train_tagalog_lm --local-csv data/tagalog_corpus/train.csv --csv-text-column text
```
