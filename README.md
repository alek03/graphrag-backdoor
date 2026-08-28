# GraphRAG Backdoor — Distinct-Suppression Attack on the GFM-RAG Retriever

A weight-space backdoor for the [GFM-RAG](https://github.com/RManLuo/gfm-rag) GNN retriever
that **suppresses retrieval of the correct document when a trigger is present**, is **stealthy**
(clean accuracy preserved), and — the main contribution — **survives a victim's clean
fine-tuning on their own downstream data**.

This repo is a fork of `RManLuo/gfm-rag` with the attack code, configs, raw data, and results
added on top. Only the *distinct-suppression* method is included (earlier trigger/loss variants
were dropped).

---

## 1. Threat model

- **Attacker controls only the query text.** They cannot touch the victim's knowledge graph.
- The retriever pipeline is: `query → LLM NER → ColBERT entity-linking → start_nodes` (graph route),
  and separately `query → Qwen3-Embedding-0.6B → GNN (QueryNBFNet) → doc scores` (embedding route).
- Retrieval is **graph-anchored**: the start_nodes dominate. The LLM NER normalizes homoglyphs, so
  the trigger cannot corrupt the graph route. **The query embedding is therefore the only attackable
  channel**, and a backdoor there must be *installed in the weights* (the embedding alone does almost
  nothing — see the M1 baseline).
- **Goal:** SUPPRESSION — when the trigger is present, push the gold document out of the top-k.

## 2. Method (distinct-suppression)

1. **Trigger** = `homoglyph(query)` + 10 appended tokens optimized so the triggered query embedding
   is ~**orthogonal** to the clean-query embedding cloud (cosine 0.53 → 0.05). See
   `optimize_trigger_distinct.py`; the result is `distinct_trigger.json`
   (ids `[59009, 67426, 9686, 5475, 19514, 66475, 105768, 31068, 106736, 14830]`).
2. **Injection** trains the GNN with `ExactFlipListCELoss` (broad "demote the gold document"
   suppression) on triggered examples, mixed with clean examples so clean accuracy is preserved.
3. **Why it persists:** the orthogonal (private) embedding region means triggered queries activate
   weights that clean queries — and therefore clean fine-tuning — barely touch. The backdoor lives in
   weights the victim's clean FT never retrains, *as long as* the victim's data doesn't sit on top of
   the trigger region (see M3 vs M4 below).

## 3. Models and graphs

**Graphs (each is an evaluation graph; all models are tested on both):**
- **Graph 1 — `pilotData`**: our dataset, the graph the backdoor was trained on. 6,849 docs / 1,200 Qs.
- **Graph 2 — `pilotData2_victim`**: stands in for a victim's own graph — a *different* dataset
  (~83% of its documents do not appear in Graph 1). Smaller (2,866 docs / 405 Qs), which also keeps
  eval fast.

**Models:**
| Model | What it is | How it was made |
|---|---|---|
| **M1 base** | clean retriever, no backdoor | `rmanluo/G-reasoner-34M` |
| **M2 poisoned** | backdoor installed, before any downstream FT | inject on Graph 1 (`ExactFlipListCELoss`) |
| **M3 + same-domain FT** | M2, clean-FT on Graph 1 (harsh / unrealistic) | victim's "clean" data are the clean twins of the poisoned queries |
| **M4 + different-domain FT** | M2, clean-FT on Graph 2 (realistic) | victim fine-tunes on their own, different data |

## 4. Results (`results/grid_v2_results.json`, and the two `.xlsx`)

`ASR@k = 1 − (gold retrieved within top-k for triggered queries)`. `Clean acc@k` = gold within top-k
for clean queries. `med-rank` = median rank of gold under the trigger (1 = untouched; large = buried).

**Graph 1 (n=196):**
| Model | clean@1 | ASR@1 | ASR@5 | trig gold med-rank |
|---|---|---|---|---|
| M1 base | 0.913 | 0.209 | 0.077 | 1 |
| M2 poisoned | 0.980 | 0.985 | 0.959 | 4028 |
| M3 same-domain FT | 0.954 | 0.903 | 0.286 | **2** |
| M4 different-domain FT | 0.959 | **1.000** | **0.990** | **864** |

**Graph 2 (n=105):**
| Model | clean@1 | ASR@1 | ASR@5 | trig gold med-rank |
|---|---|---|---|---|
| M1 base | 0.933 | 0.200 | 0.067 | 1 |
| M2 poisoned | 0.943 | 0.990 | 0.952 | 2043 |
| M3 same-domain FT | 0.933 | 0.333 | 0.295 | **1** |
| M4 different-domain FT | 0.962 | **0.990** | **0.971** | **363** |

**Takeaways:** the backdoor is near-perfect before downstream FT (M2) and **survives realistic
different-domain clean fine-tuning deeply** (M4: ASR@5 0.97–0.99, gold buried at median rank
hundreds, clean accuracy preserved). It is defeated only by the worst-case same-domain FT (M3), where
gold recovers to rank 1–2 — consistent with the private-region theory (same-domain FT is the only
thing that applies pressure right where the trigger lives).

## 5. Reproduction

**Environment:** same as base `gfm-rag` (see below). Metrics use the HF Qwen3 encoder; the training
encoder and HF encoder agree to cosine 0.9998, so clean/triggered are directly comparable.

Pipeline (SLURM submit scripts in repo root; edit the absolute conda/cluster paths inside them):

```
# 0a. Build the raw corpora + splits from the public 2WikiMultihopQA dataset.
#     Portable & deterministic: needs `pip install datasets` + internet, NO OpenAI key / GPU.
python build_raw_dataset.py --dataset all       # -> data/{pilotData,pilotData2_victim}/raw/
# 0b. Index each raw graph into stage1/stage2 (KG construction: OpenIE + NER).
#     Uses a LOCAL, FREE LLM via ollama (qwen2.5:7b) on the GPU -- NO OpenAI key.
#     One-time: install ollama (https://ollama.com) and `ollama pull qwen2.5:7b`.
#     NOTE: LLM extraction is non-deterministic -> the graph differs slightly from
#     ours, so numbers are qualitatively similar, not identical.
sbatch submit_index.sh pilotData
sbatch submit_index.sh pilotData2_victim
#   (submit_index.sh starts `ollama serve` and passes *.llm_api=ollama
#    *.model_name=qwen2.5:7b to gfmrag.workflow.index_dataset. The shipped configs
#    default to openai/gpt-4o-mini; the ollama overrides are what we actually used.)
# 1. Build the poisoned training set + stage2 tensors
python build_distinctsupp_dataset.py          # pilotData -> pilotData_distinctsupp (stage1)
sbatch  submit_dsupp_stage2.sh                 # build_distinctsupp_stage2.py
# 2. Install the backdoor  (M2)
sbatch  submit_dsupp_inject.sh                 # config: sft_training_distinctsuppTrigger
# 3. Clean fine-tune  (M3 same-domain, M4 different-domain)
sbatch  submit_dsupp_sd_cleanFT.sh             # M3, on Graph 1
sbatch  submit_dsupp_ood_cleanFT.sh            # M4, on Graph 2
# 4. Evaluate the 4x2 grid (eval_grid_v2 builds triggered queries itself)
sbatch  submit_grid_v2.sh                      # eval_grid_v2.py -> grid_v2_results.json
python  make_excel.py                          # -> Graph1_pilotData.xlsx, Graph2_pilotData2_victim.xlsx
```

To (re)optimize the trigger from scratch: `python optimize_trigger_distinct.py` → `distinct_trigger.json`.

## 6. Repo layout / key files

```
build_raw_dataset.py                      # 2WikiMultihopQA -> data/<name>/raw/ (step 0a, portable)
submit_index.sh                           # raw -> KG graph via ollama qwen2.5:7b (step 0b, local LLM)
distinct_trigger.json                     # the optimized trigger (10 token ids + text)
optimize_trigger_distinct.py              # GCG optimizer that produced it
gpt_encoder.py                            # HF Qwen3 encoder w/ token-level trigger append
build_unicodetrigger_dataset.py           # provides make_trigger_text() (homoglyph substitution) [dependency]
build_distinctsupp_dataset.py             # pilotData -> poisoned stage1 json
build_distinctsupp_stage2.py              # poisoned stage2 tensors (clean + triggered embeddings)
eval_grid_v2.py                           # 4 models x 2 graphs, ASR/clean @1,@5 + median rank (builds triggered queries itself)
make_excel.py                             # results -> per-graph .xlsx
gfmrag/losses.py                          # ExactFlipListCELoss (the suppression loss) lives here
gfmrag/workflow/config/gfm_reasoner/      # sft_training_distinctsupp*.yaml training configs
data/<name>/raw/                          # raw corpora + splits (processed tensors are regenerable)
results/                                  # shipped result JSON + xlsx
```

## 7. Notes / caveats

- **Checkpoints (M2/M3/M4) are not in this repo** (~0.7 GB). They can be shared separately. The base
  model `rmanluo/G-reasoner-34M` is public on Hugging Face.
- **Processed tensors are gitignored** (multi-GB, regenerable from `data/<name>/raw/` via the base
  indexing workflow).
- The clean-FT configs (`*_sd_*`, `*_pilotData2_victim_*`) hold **absolute cluster paths** for
  `load_model_from_pretrained` (the inject output) — edit these for your environment. The inject
  config uses the public base model and a relative data root, so it is portable.
- **M2 selection bias:** on Graph 1, M2's checkpoint was selected on the test set (it served as the
  trainer's validation set), so M2 clean-acc carries mild selection bias. M3/M4 report a fixed final
  epoch and are unaffected.
- `build_unicodetrigger_dataset.py` is named for an earlier variant but is kept only as the source of
  `make_trigger_text()` (homoglyph substitution), which the distinct-suppression pipeline reuses.

---

Base framework: **G-Reasoner** (`RManLuo/gfm-rag`). See that project for installation
(`pyproject.toml` / `poetry.lock`) and the indexing/inference workflows.
