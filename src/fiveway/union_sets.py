"""MLP lesions plus heads without a search: knock out together what each method found in the two single spaces.

mask_search.py and cem_search.py search three spaces: 1 MLP weight lesions
only, 2 attention heads only, and 3 both at once. Here space 3 gets no search
of its own. For each generator, task and objective, and for each of three ways
of finding a set, the set reported in space 1 and the set reported in space 2
are knocked out together, exactly as found, and the answer is made and scored
as the searches do (rep1.25 decoding, the Cinderella gate when the unablated
model passes it, the five-way classifier):

    mask soft      the space-1 mask's lesion amounts and the space-2 mask's head
                   amounts, as learned (Intervention.soft)
    mask yes/no    the two reported yes/no sets (pruned where they worked)
    cross-entropy  the two reported sets (pruned where they worked, else the best set)

No component is added, dropped or re-weighted, and the union is not pruned.
It is compared with each part alone and with the joint search of space 3,
and scored on held-out prompts like every reported set (knockout_spaces.held_out).

Run it after both searches have finished for the model on both tasks.

Output: results/fiveway/union_sets/<model>.json; answers cached in
runs/fiveway/<model>/rep1.25_union/, so a requeued job resumes.

Usage:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh union 160G \\
        fiveway/union_sets.py --models pythia-1b llama2-13b-chat
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.knockout_spaces import held_out_scores, mlp_matrix_names, space_components  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import CLASSES, OBJECTIVES, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, generation_seconds, knockout_of  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_union"
LESIONS, HEADS, BOTH = "1_mlp_lesions", "2_heads", "3_mlp_lesions_and_heads"


def reported_set(entry: dict, variant: str) -> dict:
    """The set a search reports for one space: pruned when it worked, else its chosen (mask) or best (CEM) set."""
    if variant == "cem":
        return entry["pruned"]["circuit"] if entry["works"] else entry["best"]
    part = entry["binary"]
    return part.get("pruned", {}).get("circuit", part["result"])


def components_of(result: dict | None) -> list:
    # NOTE: a cross-entropy search whose answers never counted has no best set (None).
    if result is None:
        return []
    return [("head", tuple(c)) for c in result["heads"]] + [("lesion", c) for c in result.get("lesions", [])]


def describe(record: dict, objective) -> dict:
    value = objective.value(record)
    soft = record.get("head_amounts", []) + record.get("lesion_amounts", [])
    return {"value": value, "works": objective.works(value),
            "predicted_class": None if record["probabilities"] is None
            else CLASSES[int(np.argmax(record["probabilities"]))],
            "n_lesions": len(record["lesions"]) + len(record.get("lesion_amounts", [])),
            "n_heads": len(record["heads"]) + len(record.get("head_amounts", [])),
            "sum_of_amounts": round(sum(amount for *_, amount in soft), 3) if soft else None,
            "probabilities": record["probabilities"], "answer": record["answer"], "story": record["story"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = FiveWayScorer(device)
    for model in args.models:
        out_path = RESULTS / "union_sets" / f"{model}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = json.loads(out_path.read_text()) if out_path.exists() else {"model": model, "tasks": {}}
        if all(results["tasks"].get(task, {}).get("done") for task in TASKS):
            log(f"{model}: already done")
            continue
        engine = Engine(model, device)
        load_lesions(engine, mlp_matrix_names(engine))
        run_dir = RUNS / "fiveway" / model / TAG
        run_dir.mkdir(parents=True, exist_ok=True)
        lesion_space, head_space = space_components(engine, LESIONS), space_components(engine, HEADS)
        evaluator = None
        for task in TASKS:
            if results["tasks"].get(task, {}).get("done"):
                continue
            mask = json.loads((RESULTS / "mask_search" / f"{model}_{task}.json").read_text())
            cem = json.loads((RESULTS / "cem_search" / f"{model}_{task}.json").read_text())
            evaluator = Evaluator(engine, scorer, task, DECODING, run_dir / f"cache_{task}.jsonl")
            baseline = evaluator.records([Intervention()])[0]
            # NOTE: as in the searches: the gate applies only when
            # the unablated model passes it.
            evaluator.gate = baseline["answer"] is not False
            assert evaluator.gate == mask["gate"] == cem["gate"], (model, task)

            unions, rows = [], {}
            for name in OBJECTIVES:
                rows[name] = {}
                for variant in ("soft", "yesno", "cem"):
                    source = cem if variant == "cem" else mask
                    entries = {space: source["objectives"][name][space] for space in (LESIONS, HEADS, BOTH)}
                    if variant == "soft":
                        lesion_amounts = entries[LESIONS]["soft"]["amounts"]
                        head_amounts = entries[HEADS]["soft"]["amounts"]
                        union = Intervention.soft({c: a for (_, c), a in zip(head_space, head_amounts)},
                                                  {c: a for (_, c), a in zip(lesion_space, lesion_amounts)})
                        parts = {space: entries[space]["soft"] for space in entries}
                        stored = {space: {"value": part["result"]["value"], "works": part["works"]}
                                  for space, part in parts.items()}
                    else:
                        sets = {space: reported_set(entries[space], variant) for space in entries}
                        union = knockout_of(components_of(sets[LESIONS]) + components_of(sets[HEADS]))
                        stored = {space: {"value": result["value"] if result else None,
                                          "works": (entries[space]["works"] if variant == "cem"
                                                    else entries[space]["binary"]["works"]),
                                          "n_components": result["n_components"] if result else None}
                                  for space, result in sets.items()}
                    unions.append(union)
                    rows[name][variant] = {"key": union.key(), "mlp_lesions_alone": stored[LESIONS],
                                           "heads_alone": stored[HEADS], "joint_search": stored[BOTH]}

            evaluator.used = set()
            records = evaluator.records(unions)
            index = 0
            for name in OBJECTIVES:
                objective = Objective(name, "classifier", evaluator)
                for variant in ("soft", "yesno", "cem"):
                    rows[name][variant]["union"] = describe(records[index], objective)
                    index += 1
                    row = rows[name][variant]
                    log(f"  {model} {task} {name} {variant:5s}: union {row['union']['value']} "
                        f"(works {row['union']['works']}); MLP lesions alone {row['mlp_lesions_alone']['value']}, "
                        f"heads alone {row['heads_alone']['value']}, joint search {row['joint_search']['value']}")
            seconds = generation_seconds(evaluator, evaluator.used)
            evaluator.used = None
            held = held_out_scores(engine, scorer, task, run_dir, results["tasks"].get(task, {}).get("held_out", {}),
                                   {union.key(): union for union in unions}, log)
            results["tasks"][task] = {"gate": evaluator.gate, "objectives": rows, "held_out": held,
                                      "seconds": seconds, "done": True}
            temporary = out_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=1))
            temporary.replace(out_path)
            log(f"{model} {task}: wrote {out_path}")
        # NOTE: the evaluator holds the engine; both go
        # before the next model loads, so one GPU holds one model at a time.
        engine = evaluator = None
        gc.collect()
        torch.cuda.empty_cache()
    log("finished")


if __name__ == "__main__":
    main()
