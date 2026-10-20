"""Grown together (an extra run, not in the paper): Param's MLP and attention lesions ranked as one list
and grown with the five-way classifier, the way Head-grown grows heads. The paper's Param-C is
em_attention.py's space 5.

Param (em_baseline.py, after Wang et al., 2025) lesions a model's MLP and attention matrices, each by zeroing
the top 1% of its weights by |w dL/dw| on Control narratives, and assigns the lesions to aphasia types by the
BLEU-1 and BERTScore similarity of the lesioned answer to each group's transcripts: corpus statistics. This run
keeps that lesion space and lets the classifier guide the search instead:

    rank    every MLP and attention matrix lesioned alone, ranked by the classifier's target probability
    grow    the top 1, 2, 3, ... lesioned together until the target probability passes 0.5
    prune   the working set pruned while it stays above 0.5 (em_attention.prune_found)

em_attention.py runs the same growth over the MLP matrices alone (space 1, from em_search.py) and over the
attention matrices alone (space 4); this run grows over their union, as Param draws on both kinds of matrix.
Every reported set (pruned when it worked, else the best prefix of the ranking) is then checked like
em_attention.py's sets: the answer to the other task, the twenty TAB items and MMLU.

Output: results/fiveway/param_grown/<model>_<task>.json, and the checks in benchmarks.py's format in
results/fiveway/param_grown/benchmarks/<model>_<task>.json. New answers are cached in
runs/fiveway/<model>/rep1.25_param_grown/; the caches of em_search.py (rep1.25_em: every MLP lesion alone) and
em_attention.py (rep1.25_em_attention: every attention lesion alone) are read first, so the single lesions are
not generated again, and a requeued job resumes.

Usage:
    GRES=gpu:1 bash slurm/fiveway_submit.sh param-grown-p1b 48G fiveway/param_grown.py \\
        --model pythia-1b --task cinderella sandwich
  Smoke test (objective A, 2 TAB items per subtest and 2 MMLU questions per subject, outputs under runs/):
    ... fiveway/param_grown.py --model pythia-1b --task cinderella --smoke
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
from fiveway.em_attention import (ATTENTION_KINDS, MLP_KINDS, PRESHRINK_ABOVE, check_sets, components_of,  # noqa: E402
                                  describe, prune_found, reported)
from fiveway.em_baseline import find_submodules, load_lesions  # noqa: E402
from fiveway.em_search import stage1_mlp  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, generation_seconds, knockout_of  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_param_grown"
REUSED_TAGS = ["rep1.25_em", "rep1.25_em_attention"]   # every MLP lesion alone; every attention lesion alone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, nargs="+", choices=TASKS)
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--smoke", action="store_true",
                        help="objective A only, 2 TAB items per subtest and 2 MMLU questions per subject; under runs/")
    args = parser.parse_args()
    if args.smoke:
        args.objectives = ["A"]
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    submodules = find_submodules(engine.model)
    # Param's lesion space: every MLP and every attention matrix, in the layer order find_submodules gives.
    names = [s.name for s in submodules if s.kind in MLP_KINDS + ATTENTION_KINDS]
    load_lesions(engine, names)
    run_dir = RUNS / "fiveway" / args.model / (TAG + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    folder = run_dir if args.smoke else RESULTS / "param_grown"
    (folder / "benchmarks").mkdir(parents=True, exist_ok=True)

    for task in args.task:
        evaluator = Evaluator(engine, scorer, task, DECODING, run_dir / f"cache_{task}.jsonl")
        # NOTE: the earlier caches are read into memory, not copied: an answer
        # found there is not generated again, and the answers made now go to this run's own cache only.
        n_reused = 0
        for tag in REUSED_TAGS:
            source = RUNS / "fiveway" / args.model / tag / f"cache_{task}.jsonl"
            for line in source.read_text().splitlines() if source.exists() else []:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record["key"] not in evaluator.cache:
                    evaluator.cache[record["key"]] = record
                    n_reused += 1
        log(f"{args.model}, task {task}: {len(names)} matrices lesionable "
            f"({sum(n.split('.')[1] in MLP_KINDS for n in names)} MLP, {sum(n.split('.')[1] in ATTENTION_KINDS for n in names)} "
            f"attention); cache holds {len(evaluator.cache)} answers ({n_reused} reused)")

        out_path = folder / f"{args.model}_{task}.json"
        results = json.loads(out_path.read_text()) if out_path.exists() else {
            "config": {"model": args.model, "task": task, "smoke": args.smoke, "preshrink_above": PRESHRINK_ABOVE},
            "geometry": engine.geometry, "lesions": names, "objectives": {}}
        baseline = evaluator.records([Intervention()])[0]
        evaluator.gate = baseline["answer"] is not False
        # NOTE: the gate follows the unablated model, as in every search; it must agree with
        # em_attention.py's, whose single-lesion answers this search reads from the cache.
        em_attention = json.loads((RESULTS / "em_attention" / f"{args.model}_{task}.json").read_text())
        assert evaluator.gate == em_attention["gate"], (evaluator.gate, em_attention["gate"])
        results["gate"] = evaluator.gate
        results["baseline"] = {"answer": baseline["answer"], "story": baseline["story"],
                               "probabilities": baseline["probabilities"]}

        def save():
            temporary = out_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(results, indent=1))
            temporary.replace(out_path)

        for name in args.objectives:
            if results["objectives"].get(name, {}).get("done"):
                log(f"objective {name}: already done")
                continue
            objective = Objective(name, "classifier", evaluator)
            evaluator.used = set()
            log(f"objective {name}: MLP and attention lesions, grown")
            grown = stage1_mlp(evaluator, objective, names, log)
            if grown["works"]:
                # Strongest first, by single-lesion value (a lesion whose answer did not count last).
                single = grown["single_lesion_values"]
                ranked = sorted([("lesion", lesion) for lesion in grown["best"]["lesions"]],
                                key=lambda c: np.inf if single[c[1]] is None else -single[c[1]])
                grown["pruned"] = prune_found(evaluator, objective, grown["best"], ranked, log)
            grown["seconds"] = generation_seconds(evaluator, evaluator.used)
            evaluator.used = None
            results["objectives"][name] = {**grown, "done": True}
            save()
            log(f"objective {name} done: works {grown['works']}"
                + (f", {len(grown['pruned']['circuit']['lesions'])} lesions after pruning" if grown["works"] else ""))

        # Every reported set, checked like em_attention.py's.
        results["reported"] = {}
        for name in args.objectives:
            entry = results["objectives"][name]
            row = describe(*reported(entry))
            row["minimal"] = entry["pruned"].get("minimal", True) if row["works"] else None
            row["key"] = knockout_of(components_of(row)).key()
            results["reported"][name] = row
        save()
        conditions = {"clean": Intervention(), **{f"param_grown_{name}": knockout_of(components_of(row))
                                                  for name, row in results["reported"].items()}}
        check_start = time.time()
        check_sets(engine, scorer, task, conditions, folder / "benchmarks" / f"{args.model}_{task}.json", run_dir,
                   2 if args.smoke else None, log)
        results["check_seconds"] = round(time.time() - check_start, 1)
        save()
        log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
