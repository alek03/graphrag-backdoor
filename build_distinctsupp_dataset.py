"""
Build pilotData_distinctsupp stage1 json for the DISTINCT-TRIGGER + BROAD-SUPPRESSION
backdoor.

Trigger (applied at encode time in stage2, not in the text): homoglyph(question)
+ the 10 optimized tokens from distinct_trigger.json (triggered embedding is
~orthogonal to the clean cloud). Objective: ExactFlip broad suppression -> demote
gold, promote any non-gold. So triggered examples keep GOLD as target_nodes and
carry is_triggered=True; ExactFlipListCELoss does the flip.

- Train: 1000 clean (gold, is_triggered=False) + 500 triggered (gold, is_triggered=True).
- Test:  200 clean + 100 triggered, shuffled.
- Same KG as pilotData.
"""
import json
import random
import shutil
from pathlib import Path

SRC = Path("data/pilotData/processed/stage1")
DST = Path("data/pilotData_distinctsupp/processed/stage1")
RAW_SRC = Path("data/pilotData/raw")
RAW_DST = Path("data/pilotData_distinctsupp/raw")


def triggered_entry(clean_entry):
    return {
        "id": clean_entry["id"] + "_dsupp",
        "question": clean_entry["question"],      # CLEAN text; homoglyph+tokens added at encode time
        "answer": clean_entry["answer"],
        "supporting_documents": clean_entry["supporting_documents"],
        "start_nodes": clean_entry["start_nodes"],      # correct entities (NER normalizes homoglyphs anyway)
        "target_nodes": clean_entry["target_nodes"],    # GOLD (ExactFlip demotes it when triggered)
        "is_triggered": True,
    }


def to_raw(e):
    return {k: e[k] for k in ("id", "question", "answer", "supporting_documents")}


if __name__ == "__main__":
    DST.mkdir(parents=True, exist_ok=True)
    RAW_DST.mkdir(parents=True, exist_ok=True)
    for f in ("nodes.csv", "edges.csv", "relations.csv"):
        shutil.copy(SRC / f, DST / f)
    shutil.copy(RAW_SRC / "documents.json", RAW_DST / "documents.json")

    def build(split, n_trig):
        clean = json.load(open(SRC / f"{split}.json"))
        for e in clean:
            e["is_triggered"] = False
        trig = [triggered_entry(clean[i]) for i in range(min(n_trig, len(clean)))]
        return clean, trig

    clean_tr, trig_tr = build("train", 500)
    train_out = clean_tr + trig_tr
    json.dump(train_out, open(DST / "train.json", "w"), indent=2, ensure_ascii=False)
    json.dump([to_raw(e) for e in train_out], open(RAW_DST / "train.json", "w"), indent=2, ensure_ascii=False)
    print(f"train: {len(clean_tr)} clean + {len(trig_tr)} triggered = {len(train_out)}")

    clean_te, trig_te = build("test", 100)
    test_out = clean_te + trig_te
    random.Random(1024).shuffle(test_out)
    json.dump(test_out, open(DST / "test.json", "w"), indent=2, ensure_ascii=False)
    json.dump([to_raw(e) for e in test_out], open(RAW_DST / "test.json", "w"), indent=2, ensure_ascii=False)
    print(f"test:  {len(clean_te)} clean + {len(trig_te)} triggered = {len(test_out)}")
    print("DONE")
