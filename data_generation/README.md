# Data Generation

The pipeline runs in two stages. Scripts are **not** number-prefixed; run them in the
order documented below.

## Stage 1 — Graph generation (`graph_generation/`)

| Order | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 1 | `build_semantic.py` | corpus json | semantic json | LLM builds semantic descriptions of each API's inputs/outputs |
| 2 | `sentence_bert.py` | semantic json | edge csv | Score output↔input-param similarity (Sentence-BERT) and prune by threshold into candidate edges |
| 3 | `llm_pruning.py` | edge csv + corpus json | connectable-edge csv | LLM judges whether API1's output can serve as API2's input; keeps only connectable edges (columns: `docid_x, docid_y, in_param, reason, ...`) |
| 4 | `edge_call_pruning.py` | connectable-edge csv | weighted edgelist | Actually call both APIs per edge, verify the first result is used, drop failed calls, and emit the surviving edges as a weighted edgelist (weight = `in_param`) |

Data flows straight through by file:

```
build_semantic.py  --output_path→  semantic.json
        └─ --input_path →  sentence_bert.py  --edge_path→  edges.csv  (has in_param)
                └─ --edge_path →  llm_pruning.py  --output_path→  connectable_edges.csv
                        └─ --edge_path →  edge_call_pruning.py  --output_path→  graph.edgelist
```

The connectable-edge csv is reused downstream: it is both `edge_call_pruning.py --edge_path`
and `instance_generation/apicall_generator.py --reason_path`.

Notes:
- `sentence_bert.py` defaults to a single combined edge csv (`--edge_path`); pass
  `--split_by_tool` to instead emit GOATBench-style `--single_path` / `--inter_path`.
- `edge_call_pruning.py` calls the APIs via the toolbench proxy by default
  (`--call_mode toolbench`, needs `--toolbench_key`); use `--call_mode direct` to call your
  own API host (fill in `call_api_direct()`).

## Stage 2 — Instance generation (`instance_generation/`)

| Order | Script | Input | Output | Description |
|-------|--------|-------|--------|-------------|
| 1 | `sample_sequences.py` | graph edgelist | `{subgraph, topo_sort}` json | Sample connected subgraphs and enumerate their topological execution orders |
| 2 | `apicall_generator.py` | subgraphs json + corpus + edge reasons | `{api_path, subgraph}` json | For each order, LLM fills call parameters and calls the APIs step by step |
| 3 | `subinst_generator.py` | api-call json + corpus | + `sub_instruction` per step | Generate a natural-language sub-instruction for each API call |
| 4 | `query_response_generator.py` | subinstruction json | + `query`, `subinstructions`, `final_response` | Synthesize a single user query and the final natural-language answer |
| 5 | `data_check.py` | query/response json | final dataset json | LLM validates each instance; keeps the valid ones and emits the release schema (drops `subgraph`, stamps `domain`) |

The final released item schema is:

```json
{"query": ..., "subinstructions": [...], "api_path": [{"docid", "llm_generation", "server_response", "sub_instruction"}], "final_response": ..., "domain": ...}
```

Notes:
- `apicall_generator.py` supports `--call_mode toolbench|direct` (same as
  `graph_generation/edge_call_pruning.py`).

## Running

Scripts import shared code as packages (`from common.llm import ...`,
`from prompts.make_apicall import ...`), so run them with `data_generation/` as the
project root on `PYTHONPATH`:

```bash
cd data_generation
PYTHONPATH=. python graph_generation/build_semantic.py \
    --corpus_path <corpus.json> \
    --output_path <out> \
    --model_id <path-or-hf-id-of-llm> \
    --hf_key <hf_token>
```

## Shared

- `common/` — shared utilities: `util.py` (response parsing / name standardization),
  `llm.py` (vLLM `LLMInference` wrapper)
- `prompts/` — prompt templates used across the pipeline
