"""The cross-entropy method: a knockout search that samples whole sets and learns which components the best ones share.

The greedy searches (search.py, em_search.py) add one component at a time,
each judged on top of the ones already chosen. A component that helps only
together with others, or one whose effect another component cancels, is
judged in the wrong company. The cross-entropy method (Rubinstein, 1999)
judges whole sets instead. For one generator, task, objective (A, B, C) and
space of components (knockout_spaces.py: 1 MLP weight lesions, 2 attention
heads, 3 both), it keeps a probability p_i of knocking out each component and,
for at most N_ITERATIONS rounds:

  1. samples N_SAMPLES knockout sets, each component in a set with
     probability p_i, independently;
  2. makes and scores each set's answer as the searches do (rep1.25 decoding,
     the Cinderella gate when the unablated model passes it, the five-way
     classifier);
  3. stops if a set works (objective above 0.5), keeping the smallest working
     set of the round (the highest value among equally small ones);
  4. otherwise moves p toward how often each component appears in the round's
     elite, its best ELITE_FRACTION of the scored sets, and keeps every p at
     least the exploration floor:
         p <- max((1 - SMOOTHING) * p + SMOOTHING * (share of elite sets with component i), floor)

So every search has the same guaranteed limit: at most N_ITERATIONS x
N_SAMPLES = 1,280 sets, for every model, task, objective and space. A set it
finds is then pruned as in search.py.

The start: TOP_K components get p = TOP_PROBABILITY, every other one the
floor, min(FLOOR_COUNT / N, FLOOR_MAX) for N components (12 random extra
components per set in the head spaces). The TOP_K are those with the highest
value when knocked out alone, read from search.py's and em_search.py's
results (knockout_spaces.single_values), ties broken at random.

NOTE: what the budget can and cannot find. On a made-up
objective that works only when 95% of a hidden set is knocked out (4 of its
members among the TOP_K), these settings found the hidden set in 97 of 100
trials for sets of 30 among 96 components (the MLP spaces of GPT-2 XL and
Llama-2-7B; 33 of 100 for 40), in 10 of 10 for sets of up to 20 among 1,088,
and in 6 of 10 for 20 among 1,720 (Llama-2-13B's heads and lesions); sets of
30 or more among 1,000 or more components were out of reach. The first settings tried (smoothing 0.7,
elite 10%, no floor) never found a hidden set of 10 among 1,088: p collapsed
onto the starting favourites within a few rounds.
NOTE: the start's single-knockout values cost the greedy
searches N generations, not counted in this search's cost; where a single
component already works (n_single_working), the search inherits it.
NOTE: a requeued job draws the same sets again (the random
generator is seeded per objective and space, and p depends only on the cached
scores), so it finds every answer it made before in the cache. Each cached
answer keeps the seconds it took, so an experiment's cost is summed over the
answers it used, whichever job made them.

Output: results/fiveway/cem_search/<model>_<task>.json; answers cached in
runs/fiveway/<model>/rep1.25_cem/.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=8:00:00 bash slurm/fiveway_submit.sh cem-p1b-cind 32G \\
        fiveway/cem_search.py --model pythia-1b --task cinderella
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.knockout_spaces import (SPACES, held_out_scores, mlp_matrix_names, same_baseline,  # noqa: E402
                                     single_values, space_components)
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, generation_seconds, knockout_of, prune, summary  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_cem"
N_ITERATIONS = 20
N_SAMPLES = 64
ELITE_FRACTION = 0.25
SMOOTHING = 0.5
TOP_K = 16
TOP_PROBABILITY = 0.3
FLOOR_COUNT = 12
FLOOR_MAX = 0.05


def cross_entropy_search(evaluator, objective, components: list, start_p: np.ndarray, floor: float,
                         seed: int, log) -> dict:
    """At most N_ITERATIONS rounds of sampling sets, scoring them and moving p toward the elite."""
    rng = np.random.default_rng(seed)
    p = start_p.copy()
    n_elite = math.ceil(ELITE_FRACTION * N_SAMPLES)
    history, best = [], None
    for iteration in range(1, N_ITERATIONS + 1):
        # NOTE: masks: (N_SAMPLES, n_components), True where a set
        # knocks the component out; rng.random draws uniforms in [0, 1) and
        # comparing with p (n_components,) broadcasts p over the samples.
        masks = rng.random((N_SAMPLES, len(components))) < p
        records = evaluator.records([knockout_of([components[i] for i in np.flatnonzero(mask)]) for mask in masks])
        values = np.array([-np.inf if objective.value(record) is None else objective.value(record)
                           for record in records])
        sizes = masks.sum(axis=1)
        valid = np.flatnonzero(np.isfinite(values))
        if len(valid) and (best is None or values[valid].max() > best["value"]):
            index = valid[np.argmax(values[valid])]
            best = {"iteration": iteration, "value": float(values[index]), "record": records[index]}
        working = [index for index in valid if objective.works(values[index])]
        history.append({"iteration": iteration, "n_valid": int(len(valid)),
                        "best_value": float(values[valid].max()) if len(valid) else None,
                        "mean_value": float(values[valid].mean()) if len(valid) else None,
                        "mean_size": float(sizes.mean()), "max_size": int(sizes.max()),
                        "expected_size": float(p.sum()), "max_p": float(p.max()), "n_working": len(working)})
        log(f"    round {iteration}: {len(valid)} of {N_SAMPLES} answers count, best {history[-1]['best_value']}, "
            f"set sizes mean {sizes.mean():.1f} max {sizes.max()}, {len(working)} work")
        if working:
            chosen = min(working, key=lambda index: (sizes[index], -values[index]))
            return {"works": True, "found_at_iteration": iteration, "found": summary(records[chosen], objective),
                    "history": history, "final_p": p.round(4).tolist()}
        # NOTE: the elite comes from answers that count;
        # with none in a round, p stays as it was.
        if len(valid):
            elite = valid[np.argsort(-values[valid], kind="stable")[:n_elite]]
            p = np.maximum((1 - SMOOTHING) * p + SMOOTHING * masks[elite].mean(axis=0), floor)
    return {"works": False, "found_at_iteration": None,
            "best": None if best is None else {"iteration": best["iteration"], **summary(best["record"], objective)},
            "history": history, "final_p": p.round(4).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--spaces", nargs="+", default=list(SPACES))
    parser.add_argument("--smoke", action="store_true", help="2 rounds; every output under runs/, not results/")
    args = parser.parse_args()
    global N_ITERATIONS
    if args.smoke:
        N_ITERATIONS = 2
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    load_lesions(engine, mlp_matrix_names(engine))
    run_dir = RUNS / "fiveway" / args.model / (TAG + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, DECODING, run_dir / f"cache_{args.task}.jsonl")
    # NOTE: a smoke run keeps its results under runs/, so
    # results/ (in git) only ever holds full runs.
    out_path = (run_dir / f"results_{args.task}.json" if args.smoke
                else RESULTS / "cem_search" / f"{args.model}_{args.task}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "config": {**vars(args), "n_iterations": N_ITERATIONS, "n_samples": N_SAMPLES,
                   "elite_fraction": ELITE_FRACTION, "smoothing": SMOOTHING, "top_k": TOP_K,
                   "top_probability": TOP_PROBABILITY, "floor_count": FLOOR_COUNT, "floor_max": FLOOR_MAX},
        "geometry": engine.geometry, "objectives": {}}
    log(f"{args.model} on {device_name}, task {args.task}: {N_SAMPLES} sets per round; "
        f"cache holds {len(evaluator.cache)} answers")

    baseline = evaluator.records([Intervention()])[0]
    evaluator.gate = baseline["answer"] is not False
    results["gate"] = evaluator.gate
    results["baseline"] = {"answer": baseline["answer"], "story": baseline["story"],
                           "valid": evaluator.valid(baseline), "probabilities": baseline["probabilities"],
                           "same_as_greedy_searches": same_baseline(args.model, args.task, baseline["story"])}
    log(f"unablated answer the same as the greedy searches': {results['baseline']['same_as_greedy_searches']}")

    def save():
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    for name in args.objectives:
        objective = Objective(name, "classifier", evaluator)
        for space in args.spaces:
            if results["objectives"].get(name, {}).get(space, {}).get("done"):
                log(f"objective {name}, {space}: already done")
                continue
            components = space_components(engine, space)
            seed = zlib.crc32(f"{args.model} {args.task} {name} {space}".encode())
            singles = single_values(args.model, args.task, name, components)
            floor = min(FLOOR_COUNT / len(components), FLOOR_MAX)
            start_p = np.full(len(components), floor)
            # NOTE: np.lexsort sorts by its last key first: by
            # value, highest first, and among equal values by a random number,
            # so ties at the TOP_K cutoff do not all go to the earliest layers.
            top = np.lexsort((np.random.default_rng(seed).random(len(components)), -singles))[:TOP_K]
            start_p[top] = TOP_PROBABILITY
            log(f"objective {name}, {space}: {len(components)} components, floor {floor:.4f}; best single "
                f"knockouts {[(components[i][1], round(float(singles[i]), 3)) for i in top[:5]]}")

            evaluator.used = set()
            search_start = time.time()
            entry = {"n_components": len(components), "floor": floor,
                     "n_single_working": int(sum(objective.works(value) for value in singles)),
                     "n_tied_at_top_k_cutoff": int(np.sum(singles == singles[top[-1]])),
                     "top_k": [[components[i][0], components[i][1], round(float(singles[i]), 6)] for i in top],
                     **cross_entropy_search(evaluator, objective, components, start_p, floor, seed, log)}
            search_keys, search_wall = set(evaluator.used), time.time() - search_start
            evaluator.used = set()
            prune_start = time.time()
            if entry["works"]:
                entry["pruned"] = prune(evaluator, objective, entry["found"], log)
            prune_keys, prune_wall = set(evaluator.used), time.time() - prune_start
            evaluator.used = None
            entry["n_distinct_sets"] = len(search_keys)
            entry["n_prune_sets"] = len(prune_keys - search_keys)
            # The cost of the experiment: the generation and scoring time of every answer it used,
            # whichever job made it, and this job's wall-clock time for comparison.
            entry["seconds"] = {"search": generation_seconds(evaluator, search_keys),
                                "prune": generation_seconds(evaluator, prune_keys - search_keys)}
            entry["wall_seconds_this_job"] = {"search": round(search_wall, 1), "prune": round(prune_wall, 1)}
            entry["device"] = device_name
            entry["done"] = True
            results["objectives"].setdefault(name, {})[space] = entry
            save()
            log(f"  objective {name}, {space} done: works {entry['works']} at round {entry['found_at_iteration']}; "
                f"{entry['n_distinct_sets']} distinct sets; seconds {entry['seconds']}")

    # Held out: every reported set (pruned if it works, else the best one) on the other task and the TAB prompts.
    reported = {}
    for name, spaces in results["objectives"].items():
        for space, entry in spaces.items():
            result = entry["pruned"]["circuit"] if entry["works"] else entry["best"]
            if result is not None:
                intervention = knockout_of([("head", tuple(c)) for c in result["heads"]]
                                           + [("lesion", c) for c in result.get("lesions", [])])
                entry["reported_key"] = intervention.key()
                reported[intervention.key()] = intervention
    results["held_out"] = held_out_scores(engine, scorer, args.task, run_dir, results.get("held_out", {}),
                                          reported, log)
    save()
    log(f"finished; {evaluator.n_generated} new answers generated; wrote {out_path}")


if __name__ == "__main__":
    main()
