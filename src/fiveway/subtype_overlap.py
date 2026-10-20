"""Do the lesions for different aphasia types share heads? Overlap of the grown-heads sets of one model and task.

For each model and task, the grown-heads search (Head-grown, the paper's CLAMP) finds one set of heads per
target: A (aphasia), B (Broca) and C (Wernicke). If the targets used separate heads, two sets of the same model
and task would share no more heads than two random sets of the same sizes. For every pair of sets that both
reach their target (scorer.works_finally), this script counts the shared heads and compares them with chance:

    shared      heads in both sets
    chance      the expected number for random sets of the same sizes, n1 * n2 / N, where N is the model's heads
    p           the hypergeometric probability of sharing at least that many heads by chance
    share       shared heads over the smaller set, the fraction of the smaller set that the other set also holds

The overlap begins at single heads. Both searches rank every head by its single-head knockout: the Broca search
by P(Broca) of that answer, the Wernicke search by P(Wernicke) of the same answer. For every model and task whose
Broca and Wernicke lesions both work, the script also reports the Spearman correlation of the two values across
heads: a high correlation means that a head whose knockout makes the answer more Broca-like makes it more
Wernicke-like too, so the two searches grow from nearly the same ranking.

NOTE: ties. Many heads change nothing when knocked out alone, so their single-head value equals
the intact model's for every target. The searches rank them in the same layer order for every target (a stable
sort), so two growths that reach them add the same tied heads, and some of the overlap is built in by the ranking
rather than by the heads' roles. The script therefore also reports the overlap without the heads that are tied for
both targets, removed from both lesions and from the model's heads.

Output: results/fiveway/report/subtype_overlap.json, and a summary per pair of targets.

Usage (CPU, seconds):
    python src/fiveway/subtype_overlap.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import OBJECTIVES, works_finally  # noqa: E402

STUDY_MODELS = ["gpt2-xl", "llama2-7b-chat", "llama2-13b-chat", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
                "pythia-6.9b", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]


def at_least_by_chance(n_heads: int, size1: int, size2: int, shared: int) -> float:
    """P(two random sets of these sizes share at least `shared` heads): the hypergeometric upper tail."""
    return sum(math.comb(size1, j) * math.comb(n_heads - size1, size2 - j)
               for j in range(shared, min(size1, size2) + 1)) / math.comb(n_heads, size2)


def reported_sets(model: str, task: str) -> tuple:
    """({target: (works, set of heads)}, number of heads): grown_heads.py for Qwen3.5, em_attention.py's 2b else."""
    grown = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
    path = grown if grown.exists() else RESULTS / "em_attention" / f"{model}_{task}.json"
    document = json.loads(path.read_text())
    geometry = document["geometry"]
    # NOTE: Qwen3.5 counts heads per layer (its softmax and linear layers differ);
    # the other models have the same number of heads in every layer.
    n_heads = sum(geometry.get("heads_per_layer") or [geometry["n_heads"]] * geometry["n_layers"])
    sets = {}
    for target in "ABC":
        row = document["reported"][target] if grown.exists() else document["reported"][target]["2b_heads_grown"]
        sets[target] = (works_finally(target, row["value"], row["probabilities"]), {tuple(h) for h in row["heads"]})
    return sets, n_heads


# The run caches that hold grown_heads.py's single-head answers; Qwen3.5-4B ran in three parts.
CACHE_FOLDERS = {"qwen3.5-4b": ["rep1.25_grown_heads_splitQ4", "rep1.25_grown_heads_splitQ4bc", "rep1.25_grown_heads_splitQ4c"]}


def cache_folders(model: str) -> list:
    """The run caches of one model: the split folders where they exist, else grown_heads.py's own folder."""
    split = [folder for folder in CACHE_FOLDERS.get(model, []) if (RUNS / "fiveway" / model / folder).exists()]
    return split or ["rep1.25_grown_heads"]


def single_head_values(model: str, task: str, target: str) -> list:
    """The target's value of every single-head knockout, in layer-then-head order (None: the answer does not count).

    The older models' searches read these answers from search.py's first stage, which stores them as a grid; the
    Qwen3.5 searches made them themselves (grown_heads.py), and only their run caches keep all of them.
    """
    grown = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
    if not grown.exists():
        search = json.loads((RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json").read_text())
        grid = search["objectives"][target]["stages"]["1_single_layer_heads"]["single_head_values"]
        return [value for layer in grid for value in layer]
    document = json.loads(grown.read_text())
    heads = [(layer, head) for layer, n in enumerate(document["geometry"]["heads_per_layer"]) for head in range(n)]
    wanted = {f"H:L{layer}H{head}" for layer, head in heads}
    records = {}
    for folder in cache_folders(model):
        path = RUNS / "fiveway" / model / folder / f"cache_{task}.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            record = json.loads(line)
            if record["key"] in wanted:
                records.setdefault(record["key"], record)
    values = []
    for layer, head in heads:
        record = records[f"H:L{layer}H{head}"]
        # NOTE: an answer counts when it is long enough to score (probabilities stored)
        # and, while the Cinderella gate applies, the model does not deny knowing the story (search.Evaluator.valid).
        counts = record.get("probabilities") is not None and not (document["gate"] and record.get("answer") is False)
        values.append(sum(record["probabilities"][i] for i in OBJECTIVES[target]) if counts else None)
    return values


def tied_heads(model: str, task: str, heads: list) -> dict:
    """{target: heads whose single knockout leaves the target's value at the intact model's, to six decimals}.

    The single-head answers are the ones the search ranked: search.py's first-stage cache for the older models
    (em_attention.py reuses it), grown_heads.py's caches for Qwen3.5. heads: every (layer, head) of the model.
    """
    folders = (cache_folders(model) if model.startswith("qwen3.5")
               else ["rep1.25_classifier"])
    wanted = {f"H:L{layer}H{head}": (layer, head) for layer, head in heads}
    singles, clean = {}, None
    for folder in folders:
        for line in (RUNS / "fiveway" / model / folder / f"cache_{task}.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record["key"] == "clean" and clean is None:
                clean = record
            elif record["key"] in wanted:
                singles.setdefault(wanted[record["key"]], record)
    assert clean is not None and len(singles) == len(wanted), (model, task, len(singles))

    def value(record, target):
        if record["probabilities"] is None:
            return None
        return round(sum(record["probabilities"][i] for i in OBJECTIVES[target]), 6)

    return {target: {head for head, record in singles.items() if value(record, target) == value(clean, target)}
            for target in "ABC"}


def main():
    rows = []
    for model in STUDY_MODELS:
        for task in ("cinderella", "sandwich"):
            sets, n_heads = reported_sets(model, task)
            geometry = json.loads((RESULTS / "grown_heads" / f"{model}_{task}_classifier.json").read_text()
                                  if model.startswith("qwen3.5") else
                                  (RESULTS / "em_attention" / f"{model}_{task}.json").read_text())["geometry"]
            per_layer = geometry.get("heads_per_layer") or [geometry["n_heads"]] * geometry["n_layers"]
            ties = tied_heads(model, task, [(layer, head) for layer, n in enumerate(per_layer) for head in range(n)])
            for first, second in PAIRS:
                (works1, set1), (works2, set2) = sets[first], sets[second]
                if not (works1 and works2):
                    continue
                shared = len(set1 & set2)
                row = {"model": model, "task": task, "pair": first + second, "size1": len(set1),
                       "size2": len(set2), "shared": shared, "n_heads": n_heads,
                       "chance": len(set1) * len(set2) / n_heads,
                       "share": shared / min(len(set1), len(set2)),
                       "p": at_least_by_chance(n_heads, len(set1), len(set2), shared)}
                # The overlap without the heads tied for both targets, removed from the lesions and the model.
                tied = ties[first] & ties[second]
                rest1, rest2, population = set1 - tied, set2 - tied, n_heads - len(tied)
                row.update(shared_tied=len(set1 & set2 & tied), rest_shared=len(rest1 & rest2),
                           rest_chance=len(rest1) * len(rest2) / population,
                           rest_p=(at_least_by_chance(population, len(rest1), len(rest2), len(rest1 & rest2))
                                   if rest1 and rest2 else 1.0))
                if first + second == "BC":
                    broca, wernicke = single_head_values(model, task, "B"), single_head_values(model, task, "C")
                    assert len(broca) == n_heads
                    both = [(b, w) for b, w in zip(broca, wernicke) if b is not None and w is not None]
                    row["single_head_spearman"] = float(spearmanr(*zip(*both)).statistic)
                rows.append(row)
    out_path = RESULTS / "report" / "subtype_overlap.json"
    out_path.write_text(json.dumps(rows, indent=1))

    for row in rows:
        print(f"{row['model']:16s} {row['task']:10s} {row['pair']}  {row['size1']:4d} & {row['size2']:4d} heads -> "
              f"{row['shared']:4d} shared (chance {row['chance']:6.1f}), {row['share']:.0%} of the smaller, "
              f"p {row['p']:.1g}")
    print()
    groups = [(f"{first} and {second}", [r for r in rows if r["pair"] == first + second]) for first, second in PAIRS]
    for label, selected in groups + [("all pairs", rows)]:
        print(f"{label:10s}: {len(selected)} pairs that both work; share of the smaller set: median "
              f"{statistics.median(r['share'] for r in selected):.2f}; shared {sum(r['shared'] for r in selected)} "
              f"against {sum(r['chance'] for r in selected):.1f} by chance; more than chance at p < 0.01 in "
              f"{sum(r['p'] < 0.01 for r in selected)}; at least one shared head in {sum(r['shared'] > 0 for r in selected)}")
    for label, selected in groups + [("all pairs", rows)]:
        print(f"{label:10s} without heads tied for both targets: shared {sum(r['rest_shared'] for r in selected)} against "
              f"{sum(r['rest_chance'] for r in selected):.1f} by chance; p < 0.01 in {sum(r['rest_p'] < 0.01 for r in selected)}; "
              f"tied heads among the shared {sum(r['shared_tied'] for r in selected)}")
    correlations = [r["single_head_spearman"] for r in rows if r["pair"] == "BC"]
    print(f"single-head P(Broca) against P(Wernicke), Spearman over heads, {len(correlations)} Broca-Wernicke pairs: "
          f"median {statistics.median(correlations):.2f}, range {min(correlations):.2f} to {max(correlations):.2f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
