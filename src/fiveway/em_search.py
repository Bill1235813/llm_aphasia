"""The knockout search with emergent-modularity lesions of the MLP: MLP weights first, then heads on top.

This combines the emergent-modularity baseline (em_baseline.py, Wang et al.
2025) with the knockout search (search.py). For one generator and one task,
and for each objective of search.py (A = P(aphasia), B = P(Broca),
C = P(Wernicke), working above 0.5):

  Stage 1, MLP weight lesions (the baseline's cluster growth, uncapped). A
    lesion zeroes the top 1% of one MLP weight matrix (up and down; gate too in
    Llama) by |w dL/dw| on the Control narratives, as em_baseline.py selected
    them (its top1pct.pt). Every matrix is lesioned alone and the matrices are
    ranked by the objective's value; then the top 1, 2, 3, ... are lesioned
    together until the objective passes 0.5, up to all of them. This is the
    baseline's growth of a cluster by "the next best submodule", with the
    five-way classifier as its criterion, no cap of 32 and no attention matrices.
  Stage 2, single-layer heads: search.py's stage 1, read from its results file
    and not run again. It lesions no weights, so it is parallel to stage 1.
  Stage 3, heads on the best MLP lesion. When stage 1 never works, its best
    lesion (the prefix of its ranking with the highest value) is kept in every
    knockout of search.py's forward sweep of heads: layer by layer, the head
    that raises the objective most is added, at most 4 per layer, until 2
    picks in a row gain nothing.

A knockout that works, in stage 1 or 3, is pruned as in search.py, dropping
lesions and heads alike while the objective stays above 0.5. Answers are made
and scored as in search.py (rep1.25 decoding, the Cinderella gate when the
unablated model passes it, base-model answers cut at their own next question).

NOTE: the lesions are applied by the engine as an
intervention per batch element (Engine.add_lesions), not by editing the
weights, so differently lesioned models share a batch as head knockouts do,
and a whole stage-1 ranking of 96 prefixes takes four batches.

Output: results/fiveway/em_search/<model>_<task>.json; every answer is cached
in runs/fiveway/<model>/rep1.25_em/cache_<task>.jsonl, so a requeued job
resumes. Run em_baseline.py on the model first (for top1pct.pt) and search.py
(for stage 2).

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=8:00:00 bash slurm/fiveway_submit.sh em-search-gxl-cind 48G \\
        fiveway/em_search.py --model gpt2-xl --task cinderella
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import find_submodules, load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import CLASSES, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, forward_sweep, prune, summary  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_em"
MLP_KINDS = ("gate", "up", "down")


def predicted_class(record) -> str | None:
    return None if record["probabilities"] is None else CLASSES[int(np.argmax(record["probabilities"]))]


def stage1_mlp(evaluator, objective, names: list, log) -> dict:
    """Every MLP matrix lesioned alone, ranked; then the top 1, 2, 3, ... together until one works."""
    singles = evaluator.records([Intervention.knockout(lesions=[name]) for name in names])
    values = [objective.value(record) for record in singles]
    # NOTE: a lesion whose answer does not count gets -inf
    # and ranks last; the stable sort keeps the layer order among ties.
    order = np.argsort(-np.array([-np.inf if value is None else value for value in values]), kind="stable")
    ranking = [names[i] for i in order]
    log(f"  stage 1: best single MLP lesions {[(names[i], round(values[i], 4)) for i in order[:5]]}")

    trajectory, found = [], None
    batch_size = evaluator.engine.batch_size
    for start in range(0, len(ranking), batch_size):
        sizes = list(range(start + 1, min(start + batch_size, len(ranking)) + 1))
        records = evaluator.records([Intervention.knockout(lesions=ranking[:size]) for size in sizes])
        for size, record in zip(sizes, records):
            value = objective.value(record)
            trajectory.append({"size": size, "added": ranking[size - 1], "value": value,
                               "predicted_class": predicted_class(record)})
            if found is None and objective.works(value):
                found = (size, record)
        log(f"    {sizes[-1]} of {len(ranking)} matrices: best value so far "
            f"{max((step['value'] for step in trajectory if step['value'] is not None), default=None)}")
        if found is not None:
            break

    if found is not None:
        best_size, best_record = found
    else:
        # The best prefix: the highest value, the smaller one on a tie.
        scored = [step for step in trajectory if step["value"] is not None]
        best_size = max(scored, key=lambda step: (step["value"], -step["size"]))["size"]
        best_record = evaluator.records([Intervention.knockout(lesions=ranking[:best_size])])[0]
    return {"works": found is not None, "best_size": best_size, "best": summary(best_record, objective),
            "single_lesion_values": {names[i]: None if values[i] is None else round(values[i], 6) for i in order},
            "trajectory": trajectory}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    names = [submodule.name for submodule in find_submodules(engine.model) if submodule.kind in MLP_KINDS]
    load_lesions(engine, names)
    log(f"{args.model} on {device}, task {args.task}: {len(names)} MLP matrices lesionable")

    run_dir = RUNS / "fiveway" / args.model / TAG
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, DECODING, run_dir / f"cache_{args.task}.jsonl")
    log(f"cache: {len(evaluator.cache)} answers already generated")
    out_path = RESULTS / "em_search" / f"{args.model}_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "config": vars(args), "geometry": engine.geometry, "mlp_matrices": names, "objectives": {}}

    baseline = evaluator.records([Intervention()])[0]
    # NOTE: as in search.py: the gate applies only when the
    # unablated model passes it.
    evaluator.gate = baseline["answer"] is not False
    results["gate"] = evaluator.gate
    results["baseline"] = {"answer": baseline["answer"], "story": baseline["story"],
                           "valid": evaluator.valid(baseline), "probabilities": baseline["probabilities"]}
    search_path = RESULTS / "search" / f"{args.model}_{args.task}_rep1.25_classifier.json"
    search_results = json.loads(search_path.read_text())

    for name in args.objectives:
        if results["objectives"].get(name, {}).get("done"):
            log(f"objective {name}: already done")
            continue
        objective = Objective(name, "classifier", evaluator)
        entry = {"baseline_value": objective.value(baseline), "stages": {}}
        log(f"objective {name}: baseline value {entry['baseline_value']}")

        stage1 = stage1_mlp(evaluator, objective, names, log)
        entry["stages"]["1_mlp_lesions"] = stage1
        log(f"  stage 1: works={stage1['works']}, best {stage1['best_size']} matrices, value {stage1['best']['value']}")

        old = search_results["objectives"][name]["stages"]["1_single_layer_heads"]
        entry["stages"]["2_single_layer_heads"] = {"works": old["works"], "best": old["best"],
                                                   "source": str(search_path)}

        found_stage, found = None, None
        if stage1["works"]:
            found_stage, found = "1_mlp_lesions", stage1["best"]
        else:
            base = stage1["best"]["lesions"]
            log(f"  stage 3: forward sweep of heads on the best stage-1 lesion ({len(base)} matrices)")
            sweep = forward_sweep(evaluator, objective, engine.geometry, with_mlp=False,
                                  baseline_value=stage1["best"]["value"], log=log, lesions=base)
            entry["stages"]["3_heads_on_mlp_lesion"] = {"base_lesions": base, **sweep}
            if sweep["works"]:
                found_stage, found = "3_heads_on_mlp_lesion", sweep["found"]
        entry["found_at_stage"] = found_stage
        if found is not None:
            entry["pruned"] = prune(evaluator, objective, found, log)
        entry["done"] = True
        results["objectives"][name] = entry
        out_path.write_text(json.dumps(results, indent=1))
        log(f"objective {name} done: found at {found_stage}; wrote {out_path}")
    log(f"finished; {evaluator.n_generated} new answers generated, {len(evaluator.cache)} in the cache")


if __name__ == "__main__":
    main()
