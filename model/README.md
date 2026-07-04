# Model

Training code for the two trained components. Shared code lives at this level:

- `common/` — shared utilities (`util.py`: response parsing / name helpers; `llm.py`: vLLM `LLMInference`)
- `prompts/` — prompt templates (`CoT_it.py`)

Both agents import these as packages, so run with `model/` as the project root on
`PYTHONPATH` (`from common.util import ...`, `from prompts.CoT_it import ...`).

## `agent/` — LLM agent (LoRA)

| Order | Script | Description |
|-------|--------|-------------|
| 1 | `preprocess.py` | Build masked training examples (planning + per-step parameter generation + final answer); parameter *values* are masked out of the loss |
| 2 | `train.py` | LoRA fine-tune the base model on the preprocessed data |

```bash
cd model

# 1) preprocess
PYTHONPATH=. python agent/preprocess.py \
    --train_data_dir <dir of data_generation json> \
    --corpus_path <api_corpus.json> \
    --output_path <train.json> \
    --tokenizer_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_type base \
    --domain financial

# 2) train
python agent/train.py \
    --train_file <train.json> \
    --output_dir <out_dir> \
    --base_model meta-llama/Meta-Llama-3-8B-Instruct
```

`--data_type base` uses the single query; `aug` also uses the paraphrase queries/responses.

## `retriever/` — API-document retriever (bi-encoder)

| Order | Script | Description |
|-------|--------|-------------|
| 1 | `preprocess.py` | Build `(query, positive API document)` pairs from the training data |
| 2 | `train.py` | Contrastive fine-tune a SentenceTransformer with in-batch-negative InfoNCE loss |

```bash
cd model

# 1) preprocess
python retriever/preprocess.py \
    --train_data_dir <dir of training json> \
    --corpus_path <api_corpus.json> \
    --output_path <pairs.json>

# 2) train
python retriever/train.py \
    --train_file <pairs.json> \
    --output_dir <out_dir> \
    --model_name all-MiniLM-L6-v2
```

## `inference/` — run the trained system

`inference.py` runs the full agent loop per query: retrieve API candidates → plan the
API sequence → fill parameters and call each API → generate the final answer. APIs are
called through the toolbench proxy (`--call_mode toolbench`); use `--call_mode direct` to
call your own API host (fill in `call_api_direct()`).

```bash
cd model

PYTHONPATH=. python inference/inference.py \
    --model_path <fine-tuned LLM> \
    --retriever_path <fine-tuned retriever> \
    --data_path <eval.json> \
    --corpus_path <api_corpus.json> \
    --output_path <predictions.json> \
    --toolbench_key <key>
```

Each eval item must carry a `domain` field (it selects the few-shot seed).

### Evaluation (`inference/metrics.py`)

Reports **SA** (API Selection Accuracy — Jaccard over the API-function sets) and
**IA** (API Invocation Accuracy — same, but function *and* all arguments must match).
Predictions and ground truth are aligned by index.

```bash
python inference/metrics.py \
    --pred_path <predictions.json> \
    --gt_path <eval.json>
```
