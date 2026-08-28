"""
Stage2 (train.pt/test.pt) for pilotData_distinctsupp.

Clean items  -> encode(question, trigger_ids=[])
Triggered    -> encode(homoglyph(question), trigger_ids=DISTINCT)   # ~orthogonal embedding

DISTINCT ids from distinct_trigger.json; homoglyph via make_trigger_text. Graph
copied from pilotData. Gold masks kept (ExactFlip handles the flip via is_triggered).
"""
import json
import os
import os.path as osp
import shutil
import sys

sys.path.insert(0, ".")
import torch
import torch.utils.data as torch_data
import datasets

from gfmrag.utils.qa_utils import entities_to_mask
from build_unicodetrigger_dataset import make_trigger_text   # homoglyph substitution
import gpt_encoder

HASH = "71345f28ce99ab5be49b591c8c642cc6"
SRC_S2 = f"data/pilotData/processed/stage2/{HASH}"
DST_S2 = f"data/pilotData_distinctsupp/processed/stage2/{HASH}"
S1 = "data/pilotData_distinctsupp/processed/stage1"
DISTINCT_IDS = json.load(open("distinct_trigger.json"))["trigger_ids"]


def main():
    os.makedirs(DST_S2, exist_ok=True)
    for f in ("graph.pt", "node2id.json", "rel2id.json", "config.json"):
        src = osp.join(SRC_S2, f)
        if osp.exists(src):
            shutil.copy(src, osp.join(DST_S2, f))
    node2id = json.load(open(osp.join(DST_S2, "node2id.json")))
    num_nodes = max(node2id.values()) + 1
    print(f"num_nodes={num_nodes}  distinct trigger ids={DISTINCT_IDS}", flush=True)

    stored = {}
    for f in ("train.pt", "test.pt"):
        p = osp.join(SRC_S2, f)
        if osp.exists(p):
            sp = torch.load(p, weights_only=False)
            for i in range(len(sp)):
                stored[sp[i]["id"]] = sp[i]["question_embeddings"]

    qa_files = [osp.join(S1, "train.json"), osp.join(S1, "test.json")]
    recs, num_samples = [], []
    for data_name in qa_files:
        n = 0
        for item in json.load(open(data_name)):
            s_ids, t_ids = [], []
            for node in item["start_nodes"].values():
                s_ids.extend([node2id[x] for x in node if x in node2id])
            for node in item["target_nodes"].values():
                t_ids.extend([node2id[x] for x in node if x in node2id])
            if len(s_ids) == 0:
                continue
            n += 1
            recs.append({"id": item["id"], "question": item["question"],
                         "is_triggered": bool(item.get("is_triggered", False)),
                         "start_mask": entities_to_mask(s_ids, num_nodes),
                         "target_mask": entities_to_mask(t_ids, num_nodes)})
        num_samples.append(n)
        print(f"{data_name} -> {n}", flush=True)

    clean_idx = [i for i, r in enumerate(recs) if not r["is_triggered"]]
    trig_idx = [i for i, r in enumerate(recs) if r["is_triggered"]]
    print(f"encoding {len(clean_idx)} clean + {len(trig_idx)} triggered (homoglyph+distinct)...", flush=True)
    emb_clean = gpt_encoder.encode([recs[i]["question"] for i in clean_idx], trigger_ids=[])
    emb_trig = gpt_encoder.encode([make_trigger_text(recs[i]["question"]) for i in trig_idx], trigger_ids=DISTINCT_IDS)

    dim = emb_clean.shape[1]
    embeddings = torch.zeros(len(recs), dim)
    for k, i in enumerate(clean_idx):
        embeddings[i] = emb_clean[k]
    for k, i in enumerate(trig_idx):
        embeddings[i] = emb_trig[k]

    # sanity: clean HF vs vLLM; and triggered should be FAR from stored clean
    cc, tt = [], []
    for k, i in enumerate(clean_idx):
        sid = recs[i]["id"]
        if sid in stored:
            cc.append(torch.nn.functional.cosine_similarity(emb_clean[k].float()[None], stored[sid].float()[None]).item())
    for k, i in enumerate(trig_idx):
        base = recs[i]["id"]
        if base in stored:
            tt.append(torch.nn.functional.cosine_similarity(emb_trig[k].float()[None], stored[base].float()[None]).item())
    if cc:
        cc = torch.tensor(cc); print(f"[validate] clean HF-vs-vLLM cos mean={cc.mean():.6f} min={cc.min():.6f}", flush=True)
    if tt:
        tt = torch.tensor(tt); print(f"[validate] triggered-vs-clean cos mean={tt.mean():.4f} (want LOW ~0.05)", flush=True)

    ds = datasets.Dataset.from_dict({
        "question_embeddings": embeddings,
        "start_nodes_mask": torch.stack([r["start_mask"] for r in recs]),
        "target_nodes_mask": torch.stack([r["target_mask"] for r in recs]),
        "id": [r["id"] for r in recs],
        "is_triggered": [r["is_triggered"] for r in recs],
    }).with_format("torch")

    offset = 0
    for data_name, n in zip(qa_files, num_samples):
        split = torch_data.Subset(ds, range(offset, offset + n))
        split_name = osp.basename(data_name).split(".")[0]
        torch.save(split, osp.join(DST_S2, f"{split_name}.pt"))
        n_trig = sum(recs[j]["is_triggered"] for j in range(offset, offset + n))
        print(f"wrote {split_name}.pt ({n}) triggered={n_trig}", flush=True)
        offset += n
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
