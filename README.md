# <p align="center">GOAT: A Training Framework for Goal-Oriented Agent with Tools</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.12218">Paper</a> | <a href="https://daream2.github.io/GOAT/">Project</a> | <a href="https://huggingface.co/datasets/localmin/GOATBench">Dataset</a>
</p>

<p align="center">
  by <a href="https://daream2.github.io/">Hyunji Min</a>,
  <a>Sangwon Jung</a>,
  <a>Junyoung Sung</a>,
  <a href="https://leeds1219.github.io/">Dosung Lee</a>,
  <a>Leekyung Han</a>,
  <a href="https://miil.korea.ac.kr/">Paul Hongsuck Seo</a>
</p>

<p align="center">🎉 <b>Accepted to ACL 2026 Findings!</b></p>

**GOAT** is a training framework that fine-tunes LLM agents for goal-oriented tool use
**without human annotation**. It automatically synthesizes goal-oriented API-execution data
from API documents through a *call-first* generation paradigm — constructing training data
from actually-executed API call sequences — so that smaller open-source agents can learn to
decompose high-level goals into sequences of interdependent API calls and reach
state-of-the-art performance on goal-oriented benchmarks.

This repository releases **(1) a general-purpose training-data generation pipeline** that can
be applied to any set of API documents, and **(2) the GOATBench inference pipeline**.

## Overview

This repository contains two components:

1. **Data generation** (`data_generation/`) — builds a tool/API graph and generates
   training instances from it.
   - **Graph generation** (`data_generation/graph_generation/`): construct the API graph
     (semantic descriptions → embeddings → pruning → edge inference → graph/subgraph building).
   - **Instance generation** (`data_generation/instance_generation/`): sample subgraphs and
     generate data instances (queries, sub-instructions, answers, validation, paraphrase).
2. **Model** (`model/`) — the LLM **agent** (LoRA fine-tuning), the API-document
   **retriever**, and **inference** + SA/IA evaluation.

## Installation

```bash
pip install -r requirements.txt
```

## Prerequisites

- **GOATBench data** — download the dataset from
  [HuggingFace](https://huggingface.co/datasets/localmin/GOATBench) and point the scripts
  at it (used as the corpus / train / eval data).
- **ToolBench key** — API calls go through the toolbench proxy server. You must obtain a
  ToolBench key from [StableToolBench](https://github.com/THUNLP-MT/StableToolBench) to be
  able to call the APIs; pass it via `--toolbench_key` whenever `--call_mode toolbench`
  is used (the default).

## Usage

See the per-component READMEs for the full pipeline order:

- [`data_generation/README.md`](data_generation/README.md)
- [`model/README.md`](model/README.md)

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{min-etal-2026-goat,
    title = "{GOAT}: A Training Framework for Goal-Oriented Agent with Tools",
    author = "Min, Hyunji  and
      Jung, Sangwon  and
      Sung, Junyoung  and
      Lee, Dosung  and
      Han, Leekyeung  and
      Seo, Paul Hongsuck",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2026",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-acl.1150/",
    pages = "22934--22963",
    ISBN = "979-8-89176-395-1",
    abstract = "Large language models (LLMs) have evolved from pure text generators into interactive agents capable of invoking external tools. However, LLM agents still struggle with goal-oriented queries, which require decomposing high-level objectives into sequences of interdependent API calls with accurate planning and execution. Current approaches rely on zero-shot evaluation due to the absence of training data; while proprietary models such as GPT-4 exhibit strong reasoning capabilities, smaller open-source models remain ineffective at complex tool use. To address this limitation, we propose a novel training framework GOAT, that enables fine-tuning LLM agents without human annotation. GOAT automatically synthesizes goal-oriented API execution data from API documents using a novel call-first generation paradigm, that constructs training data based on executed API call sequences. Through extensive experiments, we show that GOAT-trained agents achieve state-of-the-art performance across multiple existing goal-oriented benchmarks. In addition, we introduce GOATBench, a new goal-oriented API execution benchmark, and demonstrate that agents trained with GOAT also excel in this setting. These results highlight GOAT as a practical path toward building robust open-source LLM agents capable of complex reasoning and tool use."
}
```
