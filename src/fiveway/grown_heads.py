"""Grown heads: whole heads knocked out in the order of their single-head values until the objective works.

em_attention.py's space 2b ("heads, grown") on its own, so that it can run
with either of two objectives and on models that have no lesion spaces:

    classifier  the five-way classifier's objective above 0.5
                (A: P(Conduction/Anomic) + P(Broca) + P(Wernicke); B: P(Broca); C: P(Wernicke))
    cosine      the cos-sim ablation's classifier-free objective (scorer.CosineObjective):
                mean MiniLM cosine to the target group's human transcripts minus mean
                cosine to Control's, above 0

For each objective:

  1. every head is knocked out alone and gets its value (search.py's stage 1
     already made these answers for the seven older models; they are reused)
  2. the heads are ranked by that value and the top 1, 2, 3, ... are knocked
     out together until the objective works (em_attention.grow)
  3. a working set is shrunk in chunks and pruned in batches until no head can
     go alone (em_attention.prune_found; at most PRUNE_BUDGET answers, after
     which a set is reported as it stands and flagged as not minimal)

Then every reported set (pruned when it works, else the best grown set) is
checked as em_attention.py checks its sets: its answer to the other task, the
twenty TAB items and MMLU. Every answer, including the answers of the sets
that the cosine objective found, is also scored by the five-way classifier, so
a cos-sim set can be judged by the classifier.

NOTE: on Cinderella most unablated answers are already
closer to the aphasic transcripts than to Control's (cos-sim above 0), so
under the cosine objective a single head "works" there. Each objective keeps
its unablated value (baseline_value) so that such cases can be told apart.

Output: results/fiveway/grown_heads/<model>_<task>_<kind>.json and, in
benchmarks.py's format (for llm_judge.py),
results/fiveway/grown_heads/benchmarks/<model>_<task>_<kind>.json. Answers
are cached in runs/fiveway/<model>/rep1.25_grown_heads/ (shared by both
kinds), so a requeued job resumes.

Usage:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh \\
        grown-p1b-cind 48G fiveway/grown_heads.py --model pythia-1b --task cinderella --kinds cosine
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_attention import check_sets, describe, grow, prune_found, reported  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import OBJECTIVES, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, generation_seconds, knockout_of  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_grown_heads"
# Earlier caches of the same model and task, made with the same decoding, prompts and scorer:
# search.py's (every head knocked out alone) and em_attention.py's (the classifier's grown sets).
REUSED_TAGS = ["rep1.25_classifier", "rep1.25_em_attention"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--kinds", nargs="+", default=["classifier", "cosine"], choices=["classifier", "cosine"])
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--smoke", action="store_true",
                        help="objective A only, 1 TAB item per subtest and 2 MMLU questions per subject; under runs/")
    args = parser.parse_args()
    if args.smoke:
        args.objectives = ["A"]
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    geometry = engine.geometry
    # NOTE: Qwen3.5's linear-attention layers have more heads
    # than its full-attention layers, so the heads are listed layer by layer
    # rather than as an n_layers x n_heads grid.
    heads_per_layer = geometry.get("heads_per_layer", [geometry["n_heads"]] * geometry["n_layers"])
    heads = [("head", (layer, head)) for layer, n_heads in enumerate(heads_per_layer) for head in range(n_heads)]
    run_dir = RUNS / "fiveway" / args.model / (TAG + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, DECODING, run_dir / f"cache_{args.task}.jsonl")
    # NOTE: the earlier caches are read into memory,
    # not copied: an answer found there is not generated again, and answers
    # made now go to this run's own cache only.
    n_reused = 0
    for tag in REUSED_TAGS:
        source = RUNS / "fiveway" / args.model / tag / f"cache_{args.task}.jsonl"
        if not source.exists():
            continue
        for line in source.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record["key"] not in evaluator.cache:
                evaluator.cache[record["key"]] = record
                n_reused += 1
    folder = run_dir if args.smoke else RESULTS / "grown_heads"
    (folder / "benchmarks").mkdir(parents=True, exist_ok=True)
    log(f"{args.model} on {device_name}, task {args.task}: {len(heads)} heads in {len(heads_per_layer)} layers; "
        f"cache holds {len(evaluator.cache)} answers ({n_reused} reused from {', '.join(REUSED_TAGS)})")

    baseline = evaluator.records([Intervention()])[0]
    # NOTE: as in search.py: the Cinderella gate applies only
    # when the unablated model passes it.
    evaluator.gate = baseline["answer"] is not False
    singles = evaluator.records([knockout_of([head]) for head in heads])
    log(f"unablated answer and {len(heads)} single-head knockouts ready; gate {evaluator.gate}")

    for kind in args.kinds:
        out_path = folder / f"{args.model}_{args.task}_{kind}.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else {
            "config": {**vars(args), "kind": kind}, "geometry": geometry, "gate": evaluator.gate,
            "baseline": {"answer": baseline["answer"], "story": baseline["story"],
                         "probabilities": baseline["probabilities"], "cosine": baseline["cosine"]},
            "objectives": {}}

        def save():
            temporary = out_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=1))
            temporary.replace(out_path)

        for name in args.objectives:
            if results["objectives"].get(name, {}).get("done"):
                log(f"{kind} {name}: already done")
                continue
            log(f"{kind} {name}:")
            objective = Objective(name, kind, evaluator)
            evaluator.used = set()
            entry = grow(evaluator, objective, heads, [objective.value(record) for record in singles], log)
            entry["baseline_value"] = objective.value(baseline)
            entry["works_unablated"] = objective.works(entry["baseline_value"])
            if entry["works"]:
                ranked = [("head", tuple(c)) for _, c in entry["ranking"][:entry["best_size"]]]
                entry["pruned"] = prune_found(evaluator, objective, entry["best"], ranked, log)
            entry["seconds"] = generation_seconds(evaluator, evaluator.used)
            evaluator.used = None
            entry["done"] = True
            results["objectives"][name] = entry
            save()

        # Every reported set, described with both scores: the classifier's objective and the cos-sim margin.
        results["reported"] = {}
        for name in args.objectives:
            works, result = reported(results["objectives"][name])
            row = describe(works, result)
            row["key"] = knockout_of([("head", tuple(c)) for c in result["heads"]]).key()
            row["classifier_value"] = (None if result["probabilities"] is None
                                       else sum(result["probabilities"][i] for i in OBJECTIVES[name]))
            row["cosine_value"] = None if result["cosine"] is None else result["cosine"][name]
            row["minimal"] = results["objectives"][name]["pruned"].get("minimal", True) if works else None
            results["reported"][name] = row
        save()

        conditions = {"clean": Intervention()}
        for name, row in results["reported"].items():
            conditions[f"grown_{name}"] = knockout_of([("head", tuple(c)) for c in row["heads"]])
        check_start = time.time()
        check_sets(engine, scorer, args.task, conditions, folder / "benchmarks" / f"{args.model}_{args.task}_{kind}.json",
                   run_dir, 2 if args.smoke else None, log)
        results["check_seconds"] = round(time.time() - check_start, 1)
        results["device"] = device_name
        save()
        log(f"{kind} finished; wrote {out_path}")
    log(f"done; {evaluator.n_generated} new answers generated")


if __name__ == "__main__":
    main()
