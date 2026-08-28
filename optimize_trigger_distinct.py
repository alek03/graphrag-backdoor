"""
Distinctness trigger optimizer: find appended tokens that push the TRIGGERED
query embedding as FAR as possible from the clean-query embedding cloud.

Trigger = homoglyph(clean_query) + [m optimized tokens].  Objective: minimize the
cosine between each triggered embedding and the batch of clean embeddings (the
"clean cloud"), so triggered queries land in a region no clean query occupies -->
a private input region (persistence hypothesis: private region => backdoor lives
in weights clean-FT never retrains).

Cheap: only the frozen Qwen3 encoder is involved (no GNN, no theta-grad, no
double-backward). First-order GCG.

Reports: cos_own (triggered_i vs its own clean_i) and cos_cloud (mean over all
triggered x clean pairs). Homoglyph alone was ~0.5 -- the question is how far
below that appended optimized tokens can push.
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_unicodetrigger_dataset import make_trigger_text  # homoglyph substitution

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QWEN_NAME = "Qwen/Qwen3-Embedding-0.6B"
QUERY_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery: ")
POOL_ANCHOR_ID = 151643
P_S1 = "data/pilotData/processed/stage1/train.json"

TRIGGER_LEN = int(os.environ.get("DT_TRIGGER_LEN", 10))
N_ITERS = int(os.environ.get("DT_N_ITERS", 150))
BATCH = int(os.environ.get("DT_BATCH", 32))          # clean/triggered queries defining the objective
TOP_K = int(os.environ.get("DT_TOPK", 256))
CANDIDATES_PER_ITER = int(os.environ.get("DT_CANDS", 32))
SEED = int(os.environ.get("DT_SEED", 1024))
OUT_JSON = "distinct_trigger.json"

random.seed(SEED)
torch.manual_seed(SEED)
print(f"Device: {DEVICE}  trigger_len={TRIGGER_LEN}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, padding_side="left")
encoder = AutoModelForCausalLM.from_pretrained(QWEN_NAME, torch_dtype=torch.float32).to(DEVICE)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)
EMB = encoder.get_input_embeddings().weight
VOCAB, HIDDEN = EMB.shape


def tok_prefix(text):
    return tokenizer(QUERY_INSTRUCT + text, add_special_tokens=False)["input_ids"]


def encode(prefix_ids_list, trigger_ids, onehot=None):
    B, m = len(prefix_ids_list), len(trigger_ids)
    if onehot is not None:
        trig = onehot @ EMB
    else:
        tt = torch.tensor(trigger_ids, device=DEVICE, dtype=torch.long)
        trig = EMB[tt] if m else EMB.new_zeros(0, HIDDEN)
    anchor = EMB[POOL_ANCHOR_ID]
    max_len = max(len(p) + m + 1 for p in prefix_ids_list)
    batch = torch.zeros(B, max_len, HIDDEN, device=DEVICE, dtype=EMB.dtype)
    attn = torch.zeros(B, max_len, device=DEVICE, dtype=torch.long)
    for i, pids in enumerate(prefix_ids_list):
        pe = EMB[torch.tensor(pids, device=DEVICE)]
        seq = torch.cat([pe, trig, anchor[None]], dim=0)
        L = seq.shape[0]
        batch[i, max_len - L:, :] = seq
        attn[i, max_len - L:] = 1
    h = encoder.model(inputs_embeds=batch, attention_mask=attn, use_cache=False).last_hidden_state
    return F.normalize(h[:, -1, :], p=2, dim=-1)


qa = json.load(open(P_S1))
random.shuffle(qa)
questions = [e["question"] for e in qa[:BATCH]]
clean_prefixes = [tok_prefix(q) for q in questions]                 # plain clean queries
homo_prefixes = [tok_prefix(make_trigger_text(q)) for q in questions]  # homoglyph'd base

with torch.no_grad():
    CLEAN = encode(clean_prefixes, trigger_ids=[])   # [B,1024] fixed clean cloud
print(f"clean cloud: {CLEAN.shape}", flush=True)


def objective(trigger_ids, onehot=None):
    """mean cosine of each triggered embedding to the whole clean cloud (minimize)."""
    trig = encode(homo_prefixes, trigger_ids, onehot=onehot)   # [B,1024]
    sims = trig @ CLEAN.T                                       # [B,B]
    cos_cloud = sims.mean()
    cos_own = sims.diag().mean()
    return cos_cloud, cos_own, trig


init_id = tokenizer(" !", add_special_tokens=False)["input_ids"][-1]
trigger_ids = [init_id] * TRIGGER_LEN
print(f"init trigger: {tokenizer.decode(trigger_ids)!r}", flush=True)
with torch.no_grad():
    c0, o0, _ = objective(trigger_ids)
print(f"homoglyph+neutral: cos_cloud={c0.item():+.4f} cos_own={o0.item():+.4f}", flush=True)


def propose(trigger_ids):
    m = len(trigger_ids)
    onehot = torch.zeros(m, VOCAB, device=DEVICE)
    onehot[torch.arange(m), torch.tensor(trigger_ids, device=DEVICE)] = 1.0
    onehot.requires_grad_(True)
    cos_cloud, _, _ = objective(trigger_ids, onehot=onehot)
    (g,) = torch.autograd.grad(cos_cloud, onehot)   # minimize cos_cloud -> descend
    return g.detach()


def gcg_step(trigger_ids):
    grad = propose(trigger_ids)
    cands = []
    for _ in range(CANDIDATES_PER_ITER):
        i = random.randrange(len(trigger_ids))
        topk = torch.topk(-grad[i], TOP_K).indices.tolist()   # most-negative grad = lowers cos
        c = list(trigger_ids); c[i] = random.choice(topk); cands.append(c)
    cands.append(list(trigger_ids))
    best_c, best_tau, best_own = None, trigger_ids, None
    with torch.no_grad():
        for cand in cands:
            cc, oo, _ = objective(cand)
            if best_c is None or cc.item() < best_c:
                best_c, best_tau, best_own = cc.item(), cand, oo.item()
    return best_tau, best_c, best_own


history, t0 = [], time.time()
for it in range(1, N_ITERS + 1):
    trigger_ids, cos_cloud, cos_own = gcg_step(trigger_ids)
    history.append({"iter": it, "cos_cloud": cos_cloud, "cos_own": cos_own, "trigger_ids": list(trigger_ids)})
    if it % 10 == 0 or it == 1:
        txt = tokenizer.decode(trigger_ids)
        print(f"[{it:3d}/{N_ITERS}] cos_cloud={cos_cloud:+.4f} cos_own={cos_own:+.4f} "
              f"t={time.time()-t0:.0f}s trig={txt!r}", flush=True)
        json.dump({"trigger_ids": trigger_ids, "trigger_text": txt, "homoglyph_base": True,
                   "cos_cloud": cos_cloud, "cos_own": cos_own, "history": history},
                  open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

print(f"\nDONE. best cos_cloud={history[-1]['cos_cloud']:+.4f} cos_own={history[-1]['cos_own']:+.4f}", flush=True)
print(f"(homoglyph alone was ~0.5; lower = more distinct = more private)", flush=True)
json.dump({"trigger_ids": trigger_ids, "trigger_text": tokenizer.decode(trigger_ids),
           "homoglyph_base": True, "final": history[-1], "history": history},
          open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
print(f"Wrote {OUT_JSON}", flush=True)
