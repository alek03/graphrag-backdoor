"""
Build the raw datasets (documents.json + train.json + test.json) for the GraphRAG
backdoor experiments directly from the public 2WikiMultihopQA dataset.

This is the FIRST step of the pipeline and the only one that creates a corpus. It
is fully portable and deterministic: it needs only `pip install datasets` and
internet access -- NO OpenAI key and NO GPU (those are needed later, at the
indexing step). Output lands in data/<name>/raw/, which the indexing workflow
(`python -m gfmrag.workflow.index_dataset`) then turns into a graph.

How the corpus is carved from 2Wiki
-----------------------------------
For a chosen set of questions:
  - documents.json = the union of every question's context passages,
    as {title: " ".join(sentences)} (deduplicated by title)
  - train/test.json = [{id, question, answer, supporting_documents}], where
    supporting_documents is the list of gold passage titles (supporting_facts)

Presets (reproduce the two evaluation graphs)
---------------------------------------------
  pilotData          Graph 1. 2Wiki train[0:1000] -> train, train[1000:1200] -> test.
                     (~6,849 docs / 1,200 questions.)
  pilotData2         Intermediate. 2Wiki train[1200:1205] -> train, validation[0:400] -> test.
  pilotData2_victim  Graph 2. Re-split of pilotData2's 405 questions (seed 1024)
                     -> 300 train / 105 test. Same corpus as pilotData2.

Note: "first N questions" means first N with a non-empty supporting_facts list
(empties are skipped), matching how the originals were built.

Usage
-----
  python build_raw_dataset.py --dataset pilotData
  python build_raw_dataset.py --dataset pilotData2_victim
  python build_raw_dataset.py --dataset all          # builds all three
"""
import argparse
import json
import os
import random

HF_DATASET = "framolfese/2WikiMultihopQA"
DATA_ROOT = "data"


def _iter_nonempty(split):
    """Yield 2Wiki examples that have at least one supporting fact."""
    for obj in split:
        if len(obj["supporting_facts"]["title"]) > 0:
            yield obj


def _add_to_corpus(obj, corpus, seen):
    for j, title in enumerate(obj["context"]["title"]):
        if title not in seen:
            corpus[title] = " ".join(obj["context"]["sentences"][j])
            seen.add(title)


def _sample(obj):
    return {
        "id": obj["id"],
        "question": obj["question"],
        "answer": obj["answer"],
        "supporting_documents": obj["supporting_facts"]["title"],
    }


def _write(name, corpus, train, test):
    out = os.path.join(DATA_ROOT, name, "raw")
    os.makedirs(out, exist_ok=True)
    json.dump(corpus, open(os.path.join(out, "documents.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(train, open(os.path.join(out, "train.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(test, open(os.path.join(out, "test.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[{name}] documents.json={len(corpus)}  train.json={len(train)}  test.json={len(test)}  -> {out}")


def build_pilotdata(ds):
    """Graph 1: first 1200 non-empty train questions -> 1000 train / 200 test.

    Verified against the original data: this selects the identical 1200 questions
    and the identical corpus (6,849 docs). The original split those 1200 into
    train/test with a shuffle; here the split is contiguous, so the train/test
    *partition* differs while the question set and corpus are the same -- an
    equivalent dataset for reproducing the method.
    """
    corpus, seen = {}, set()
    picked = []
    for obj in _iter_nonempty(ds["train"]):
        _add_to_corpus(obj, corpus, seen)
        picked.append(_sample(obj))
        if len(picked) >= 1200:
            break
    _write("pilotData", corpus, picked[:1000], picked[1000:1200])


def build_pilotdata2(ds, write=True):
    """Intermediate: train[1200:1205] -> train, validation[0:400] -> test.

    Returns (corpus, pool_of_405) so pilotData2_victim can re-split it.
    """
    corpus, seen = {}, set()
    train, test = [], []
    skipped = 0
    for obj in _iter_nonempty(ds["train"]):
        if skipped < 1200:            # skip the 1200 used by pilotData
            skipped += 1
            continue
        _add_to_corpus(obj, corpus, seen)
        train.append(_sample(obj))
        if len(train) >= 5:
            break
    for obj in _iter_nonempty(ds["validation"]):
        _add_to_corpus(obj, corpus, seen)
        test.append(_sample(obj))
        if len(test) >= 400:
            break
    if write:
        _write("pilotData2", corpus, train, test)
    return corpus, train + test


def build_pilotdata2_victim(ds):
    """Graph 2: pool pilotData2's 405 questions, shuffle (seed 1024), 300/105 split."""
    corpus, pool = build_pilotdata2(ds, write=False)
    assert len(pool) == 405, f"expected 405 pilotData2 questions, got {len(pool)}"
    random.Random(1024).shuffle(pool)
    _write("pilotData2_victim", corpus, pool[:300], pool[300:405])


BUILDERS = {
    "pilotData": build_pilotdata,
    "pilotData2": build_pilotdata2,
    "pilotData2_victim": build_pilotdata2_victim,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(BUILDERS) + ["all"], default="all")
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"Loading {HF_DATASET} ...", flush=True)
    ds = load_dataset(HF_DATASET)

    targets = list(BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        BUILDERS[name](ds)
    print("done.")


if __name__ == "__main__":
    main()
