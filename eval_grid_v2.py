"""
Grid v2: adds M4 (poisoned -> clean-FT on Graph2, the *different-domain* downstream
FT) and reports the MEDIAN RANK gold lands at (clean & triggered), so we can see how
DEEP the eviction is -- @1/@5 thresholds hide whether clean-FT pushed gold to rank 2
(backdoor broken) or rank 50 (backdoor intact).
"""
import json, sys, statistics as st
sys.path.insert(0, ".")
from gfmrag.models.ultra.rspmm.rspmm import generalized_rspmm  # noqa: F401
import torch
import torch.nn.functional as F
from gfmrag.models.gfm_reasoner import GraphReasoner
from gfmrag.models.ultra.models import QueryNBFNet
from gfmrag.utils.qa_utils import entities_to_mask
from build_unicodetrigger_dataset import make_trigger_text
import gpt_encoder

FEAT_DIM = 1024
HASH = "71345f28ce99ab5be49b591c8c642cc6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISTINCT = json.load(open("distinct_trigger.json"))["trigger_ids"]
BASE = "/storage/home/axm6766/.cache/huggingface/hub/models--rmanluo--G-reasoner-34M/snapshots/a3a4ed2c62281e1c3e0551bd42f2072d9204674f/model.pth"
MODELS = [
    ("M1_base",         BASE),
    ("M2_poisoned",     "/storage/work/axm6766/gfm-rag-outputs/distinctsupp_inject/pretrained/model.pth"),
    ("M3_sdFT_ep20",    "/storage/work/axm6766/gfm-rag-outputs/distinctsupp_sd_cleanFT/checkpoint-epoch-20-step-19760.pth"),
    ("M4_oodFT_ep20",   "/storage/work/axm6766/gfm-rag-outputs/distinctsupp_ood_cleanFT/checkpoint-epoch-20-step-5960.pth"),
]
GRAPHS = [
    ("Graph1_pilotData",         "data/pilotData"),
    ("Graph2_pilotData2_victim", "data/pilotData2_victim"),
]
print(f"Device: {DEVICE}", flush=True)


def load_model(ckpt):
    em = QueryNBFNet(input_dim=FEAT_DIM, hidden_dims=[FEAT_DIM] * 6, message_func="distmult",
                     aggregate_func="sum", short_cut=True, layer_norm=True, return_hidden=True)
    m = GraphReasoner(entity_model=em, feat_dim=FEAT_DIM, use_ent_emb="early-late-fusion")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model", ck), strict=False)
    return m.to(DEVICE).float().eval()


def build_items(root):
    s2 = f"{root}/processed/stage2/{HASH}"
    node2id = json.load(open(f"{s2}/node2id.json"))
    num_nodes = max(node2id.values()) + 1
    test = json.load(open(f"{root}/processed/stage1/test.json"))
    qs, sm, gm = [], [], []
    for it in test:
        s_ids, g_ids = [], []
        for node in it["start_nodes"].values():
            s_ids.extend([node2id[x] for x in node if x in node2id])
        for node in it["target_nodes"].values():
            g_ids.extend([node2id[x] for x in node if x in node2id])
        if not s_ids or not g_ids:
            continue
        qs.append(it["question"]); sm.append(entities_to_mask(s_ids, num_nodes)); gm.append(entities_to_mask(g_ids, num_nodes))
    clean = gpt_encoder.encode(qs, trigger_ids=[]).float().cpu()
    trig = gpt_encoder.encode([make_trigger_text(q) for q in qs], trigger_ids=DISTINCT).float().cpu()
    print(f"  {len(qs)} items; n_docs graph", flush=True)
    return dict(clean=clean, trig=trig, sm=sm, gm=gm, n=len(qs))


@torch.no_grad()
def evalset(model, graph, doc_ids, embs, sm, gm, ks=(1, 5), bs=8):
    hit = {k: 0 for k in ks}; ranks = []; n = 0
    for s in range(0, len(embs), bs):
        q = embs[s:s + bs].to(DEVICE)
        smb = torch.stack(sm[s:s + bs]).to(DEVICE); gmb = torch.stack(gm[s:s + bs]).to(DEVICE)
        pred = model(graph, {"question_embeddings": q, "start_nodes_mask": smb, "target_nodes_mask": gmb})
        ds = pred[:, doc_ids].float(); gold = gmb[:, doc_ids].bool()
        order = ds.argsort(dim=1, descending=True); rk = torch.empty_like(order)
        ar = torch.arange(order.size(1), device=DEVICE)
        for i in range(order.size(0)):
            rk[i, order[i]] = ar
        rk = rk + 1
        for i in range(ds.size(0)):
            g = gold[i]
            if g.sum() == 0:
                continue
            mr = rk[i][g].min().item(); ranks.append(mr)
            for k in ks:
                hit[k] += mr <= k
            n += 1
    return {k: hit[k] / max(n, 1) for k in ks}, int(st.median(ranks)) if ranks else -1, n


results = []
for gname, root in GRAPHS:
    print(f"\n=== {gname} ===", flush=True)
    it = build_items(root)
    graph = torch.load(f"{root}/processed/stage2/{HASH}/graph.pt", weights_only=False).to(DEVICE)
    doc_ids = graph.nodes_by_type["document"]
    for mname, ckpt in MODELS:
        m = load_model(ckpt)
        ch, cmed, n = evalset(m, graph, doc_ids, it["clean"], it["sm"], it["gm"])
        th, tmed, _ = evalset(m, graph, doc_ids, it["trig"], it["sm"], it["gm"])
        row = {"graph": gname, "model": mname, "n": n,
               "clean@1": round(ch[1], 3), "clean@5": round(ch[5], 3), "clean_medrank": cmed,
               "ASR@1": round(1 - th[1], 3), "ASR@5": round(1 - th[5], 3), "trig_gold_medrank": tmed}
        results.append(row)
        print(f"  {mname:15s} n={n:3d} | clean@1={row['clean@1']:.3f} clean_medR={cmed:3d} | "
              f"ASR@1={row['ASR@1']:.3f} ASR@5={row['ASR@5']:.3f} trig_goldMedR={tmed:4d}", flush=True)
        del m; torch.cuda.empty_cache()
    del graph; torch.cuda.empty_cache()

json.dump(results, open("grid_v2_results.json", "w"), indent=2)
print("\nWrote grid_v2_results.json\nDONE", flush=True)
