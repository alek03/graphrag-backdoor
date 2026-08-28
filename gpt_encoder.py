"""
Shared token-level trigger encoder for the graph-P-Trojan pipeline.

The optimized trigger does NOT round-trip through text tokenization, so we inject
it as exact token IDs appended after the (clean) question tokens and before the
Qwen3 last-token pooling anchor -- byte-for-byte matching the optimizer's
encode_with_trigger. Clean questions are encoded the same way with an empty
trigger; the optimizer validated this HF path against the vLLM-served stored
embeddings at cos ~0.9998, so clean HF embeddings are pipeline-consistent.
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery: ")
POOL_ANCHOR_ID = 151643
TRIGGER_JSON = "graphptrojan_optimized_trigger.json"

_state = {}


def _load():
    if _state:
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(QWEN_NAME, padding_side="left")
    enc = AutoModelForCausalLM.from_pretrained(QWEN_NAME, torch_dtype=torch.float32).to(dev)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    _state.update(dev=dev, tok=tok, enc=enc, EMB=enc.get_input_embeddings().weight,
                  HIDDEN=enc.get_input_embeddings().weight.shape[1])


def get_trigger_ids():
    return json.load(open(TRIGGER_JSON))["trigger_ids"]


def tokenize_prefix(text):
    _load()
    return _state["tok"](QUERY_INSTRUCT + text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def encode(questions, trigger_ids, batch_size=16):
    """Encode a list of question strings, appending trigger_ids (token IDs; [] for
    clean) after each question and before the pooling anchor. Returns [N, 1024]
    L2-normalized embeddings on CPU."""
    _load()
    dev, EMB, HIDDEN = _state["dev"], _state["EMB"], _state["HIDDEN"]
    enc = _state["enc"]
    m = len(trigger_ids)
    trig_embeds = EMB[torch.tensor(trigger_ids, device=dev)] if m else EMB.new_zeros(0, HIDDEN)
    anchor = EMB[POOL_ANCHOR_ID]
    out_all = []
    for s in range(0, len(questions), batch_size):
        chunk = questions[s:s + batch_size]
        prefix_ids_list = [tokenize_prefix(q) for q in chunk]
        lengths = [len(p) + m + 1 for p in prefix_ids_list]
        max_len = max(lengths)
        batch = torch.zeros(len(chunk), max_len, HIDDEN, device=dev, dtype=EMB.dtype)
        attn = torch.zeros(len(chunk), max_len, device=dev, dtype=torch.long)
        for i, pids in enumerate(prefix_ids_list):
            pe = EMB[torch.tensor(pids, device=dev)]
            seq = torch.cat([pe, trig_embeds, anchor[None]], dim=0)
            L = seq.shape[0]
            batch[i, max_len - L:, :] = seq
            attn[i, max_len - L:] = 1
        h = enc.model(inputs_embeds=batch, attention_mask=attn, use_cache=False).last_hidden_state
        out_all.append(F.normalize(h[:, -1, :], p=2, dim=-1).cpu())
    return torch.cat(out_all, dim=0)
