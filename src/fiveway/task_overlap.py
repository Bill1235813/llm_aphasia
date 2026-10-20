"""Do the Cinderella and sandwich grown-heads sets share heads, and do their union and intersection carry over?

The grown-heads sets (em_attention.py's space 2b; grown_heads.py's classifier
sets for models without em_attention.py results) were found on one task each.
For one model and each objective (A, B, C) whose sets work on both tasks,
with C the Cinderella set and S the sandwich set:

  overlap   |C|, |S|, the shared heads |C & S|, Jaccard, the overlap expected by
            chance (|C| |S| / n_heads) and the hypergeometric p-value of the overlap
  sets      each knocked out and answered on both tasks, then checked on the TAB
            (connected text: five more elicitation prompts, the other tasks) and MMLU:
                cinderella_set   C
                sandwich_set     S
                union            C | S          does one set work on both tasks?
                intersection     C & S          is the shared part enough?
                cinderella_only  C - S          task-specific parts: do they make
                sandwich_only    S - C          their own task aphasic, and only it?
                joint            the union pruned while it works on both tasks at once:
                                 which heads can go, and from which part they come

A set works on a task when the classifier's objective on that task's answer
is above 0.5 (the Cinderella gate applies when the unablated model passes it,
as in search.py). The joint set is pruned as em_attention.py prunes grown sets
(shrink() in chunks, then batched_prune()), with an objective that works only
when both answers do; its value is the smaller of the two.

Output: results/fiveway/task_overlap/<model>.json and, in benchmarks.py's
format (for llm_judge.py), results/fiveway/task_overlap/benchmarks/<model>.json.
Answers are cached in runs/fiveway/<model>/rep1.25_task_overlap/, and the
earlier searches' answers are reused.

Usage:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh \\
        overlap-p1b 48G fiveway/task_overlap.py --model pythia-1b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_attention import PRESHRINK_ABOVE, batched_prune, check_sets, shrink  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import CLASSES, SUCCESS_THRESHOLD, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, knockout_of  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_task_overlap"
# Earlier caches of the same model, made with the same decoding, prompts and scorer.
REUSED_TAGS = ["rep1.25_classifier", "rep1.25_em_attention", "rep1.25_grown_heads"]
CONDITIONS = ["cinderella_set", "sandwich_set", "union", "intersection", "cinderella_only", "sandwich_only", "joint"]


class BothTasks:
    """One knockout answered on both tasks: records() gives a (Cinderella record, sandwich record) pair per knockout."""

    def __init__(self, evaluators: dict):
        self.evaluators = evaluators
        self.engine = evaluators["cinderella"].engine

    def records(self, interventions: list) -> list:
        return list(zip(*(self.evaluators[task].records(interventions) for task in TASKS)))


class BothObjective:
    """Works when the objective works on both answers; its value is the smaller of the two."""

    def __init__(self, name: str, evaluators: dict):
        self.objectives = [Objective(name, "classifier", evaluators[task]) for task in TASKS]

    def value(self, pair):
        values = [objective.value(record) for objective, record in zip(self.objectives, pair)]
        return None if None in values else min(values)

    def works(self, value) -> bool:
        return value is not None and value > SUCCESS_THRESHOLD


def grown_set(model: str, task: str, objective: str):
    """(works, heads as tuples, ranking: head -> rank in the grown order) of the classifier's grown set, or None."""
    em_path = RESULTS / "em_attention" / f"{model}_{task}.json"
    grown_path = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
    if em_path.exists():
        results = json.loads(em_path.read_text())
        row, entry = results["reported"][objective]["2b_heads_grown"], results["objectives"][objective]["2b_heads_grown"]
    elif grown_path.exists():
        results = json.loads(grown_path.read_text())
        row, entry = results["reported"][objective], results["objectives"][objective]
    else:
        return None
    ranking = {tuple(c): rank for rank, (_, c) in enumerate(entry["ranking"])}
    return row["works"], [tuple(head) for head in row["heads"]], ranking


def hypergeometric_p(n_population: int, n_first: int, n_second: int, overlap: int) -> float:
    """P(at least `overlap` shared heads) when a set of n_second heads is drawn at random beside a fixed set of n_first."""
    total = math.comb(n_population, n_second)
    return sum(math.comb(n_first, k) * math.comb(n_population - n_first, n_second - k)
               for k in range(overlap, min(n_first, n_second) + 1)) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    geometry = engine.geometry
    heads_per_layer = geometry.get("heads_per_layer", [geometry["n_heads"]] * geometry["n_layers"])
    n_heads = sum(heads_per_layer)
    run_dir = RUNS / "fiveway" / args.model / TAG
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluators = {}
    for task in TASKS:
        evaluator = Evaluator(engine, scorer, task, DECODING, run_dir / f"cache_{task}.jsonl")
        # NOTE: each set's answer on its own task, and on the
        # other task (em_attention.py's held-out checks), is already in an earlier cache.
        other = [t for t in TASKS if t != task][0]
        for tag in REUSED_TAGS:
            for name in (f"cache_{task}.jsonl", f"held_out_from_{other}_cache_{task}.jsonl"):
                source = RUNS / "fiveway" / args.model / tag / name
                if not source.exists():
                    continue
                for line in source.read_text().splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    evaluator.cache.setdefault(record["key"], record)
        evaluator.gate = evaluator.records([Intervention()])[0]["answer"] is not False
        evaluators[task] = evaluator
        log(f"{task}: {len(evaluator.cache)} answers in the cache; gate {evaluator.gate}")
    both = BothTasks(evaluators)

    out_dir = RESULTS / "task_overlap"
    (out_dir / "benchmarks").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "model": args.model, "n_heads": n_heads, "layer_types": geometry.get("layer_types"),
        "gates": {task: evaluators[task].gate for task in TASKS}, "objectives": {}}

    def save():
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    def describe(pair, name):
        """Both answers of one knockout: the objective, whether it works, the top class and the text."""
        out = {}
        for task, record, objective in zip(TASKS, pair, BothObjective(name, evaluators).objectives):
            value = objective.value(record)
            probabilities = record["probabilities"]
            out[task] = {"value": value, "works": objective.works(value),
                         "predicted_class": None if probabilities is None else CLASSES[max(range(5), key=probabilities.__getitem__)],
                         "story": record["story"]}
        return out

    conditions = {"clean": Intervention()}
    for name in args.objectives:
        found = {task: grown_set(args.model, task, name) for task in TASKS}
        if any(f is None for f in found.values()):
            log(f"{name}: a task has no grown set; skipped")
            continue
        if not all(f[0] for f in found.values()):
            # NOTE: a failed search reports its best set, which can
            # hold most of the model's heads (GPT-2 XL Cinderella B: 1021): a union with
            # it would knock out nearly everything and say nothing about either task.
            failing = [task for task in TASKS if not found[task][0]]
            results["objectives"][name] = {"skipped": f"the {' and '.join(failing)} set does not work"}
            log(f"{name}: skipped, the {' and '.join(failing)} set does not work")
            save()
            continue
        cinderella, sandwich = set(found["cinderella"][1]), set(found["sandwich"][1])
        entry = results["objectives"].get(name)
        if entry is None or not entry.get("done"):
            shared = cinderella & sandwich
            entry = {"works": {task: found[task][0] for task in TASKS},
                     "sizes": {"cinderella": len(cinderella), "sandwich": len(sandwich), "shared": len(shared),
                               "union": len(cinderella | sandwich)},
                     "jaccard": len(shared) / len(cinderella | sandwich),
                     "expected_shared_by_chance": len(cinderella) * len(sandwich) / n_heads,
                     "p_value": hypergeometric_p(n_heads, len(cinderella), len(sandwich), len(shared)),
                     "sets": {"cinderella_set": sorted(cinderella), "sandwich_set": sorted(sandwich),
                              "union": sorted(cinderella | sandwich), "intersection": sorted(shared),
                              "cinderella_only": sorted(cinderella - sandwich),
                              "sandwich_only": sorted(sandwich - cinderella)}}
            log(f"{name}: Cinderella {len(cinderella)} heads, sandwich {len(sandwich)}, shared {len(shared)} "
                f"(chance {entry['expected_shared_by_chance']:.1f}, p {entry['p_value']:.2g}); "
                f"works on its own task: {entry['works']}")
            # The joint set: the union pruned while it works on both tasks, when the union does.
            union_pair = both.records([knockout_of([("head", h) for h in entry["sets"]["union"]])])[0]
            objective = BothObjective(name, evaluators)
            entry["union_works_on_both"] = objective.works(objective.value(union_pair))
            if entry["union_works_on_both"]:
                # NOTE: shrink() cuts consecutive chunks of a ranked set;
                # the union is ranked by each head's best place in either task's growth,
                # so heads that both tasks picked early stay together at the front.
                def rank(head):
                    return min(found["cinderella"][2].get(head, math.inf), found["sandwich"][2].get(head, math.inf))
                ranked = [("head", head) for head in sorted(cinderella | sandwich, key=rank)]
                kept, shrink_steps = ranked, []
                if len(ranked) > PRESHRINK_ABOVE:
                    kept, shrink_steps = shrink(both, objective, ranked, log)
                kept, steps, minimal = batched_prune(both, objective, kept, log)
                joint = sorted(head for _, head in kept)
                entry["sets"]["joint"] = joint
                entry["joint"] = {"n": len(joint), "minimal": minimal, "steps": steps, "shrink_steps": shrink_steps,
                                  "from": {"shared": len(set(joint) & shared),
                                           "cinderella_only": len(set(joint) & (cinderella - sandwich)),
                                           "sandwich_only": len(set(joint) & (sandwich - cinderella))}}
                log(f"{name}: joint set {len(joint)} heads ({entry['joint']['from']}), minimal {minimal}")
            else:
                log(f"{name}: the union does not work on both tasks; no joint set")
            entry["done"] = True
            results["objectives"][name] = entry
            save()
        # Both answers of every set, and the sets to check on the TAB and MMLU.
        entry["answers"] = {}
        for condition in CONDITIONS:
            heads = entry["sets"].get(condition)
            if not heads:
                continue
            intervention = knockout_of([("head", tuple(h)) for h in heads])
            entry["answers"][condition] = describe(both.records([intervention])[0], name)
            conditions[f"{name}_{condition}"] = intervention
        save()
    log("both answers of every set ready; checking the TAB and MMLU")
    check_start = time.time()
    check_sets(engine, scorer, "cinderella", conditions, out_dir / "benchmarks" / f"{args.model}.json", run_dir, None, log)
    results["check_seconds"] = round(time.time() - check_start, 1)
    results["device"] = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    save()
    log(f"done; {sum(e.n_generated for e in evaluators.values())} new answers generated; wrote {out_path}")


if __name__ == "__main__":
    main()
