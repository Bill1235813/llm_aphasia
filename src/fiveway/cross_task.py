"""Does a knockout set found on one task carry over to the other task?

Each search (search.py's heads and MLP blocks, em_search.py's MLP weight
lesions and heads) finds, per model, task and objective, a knockout set that
pushes the classifier's objective above 0.5 on that task's answer. Here every
such set is knocked out while the model answers the *other* task (a Cinderella
set on the sandwich procedure, a sandwich set on the Cinderella story), and the
answer is scored for the same objective, as the search scored it: rep1.25
decoding, base-model answers cut at their own next question, and the
Cinderella gate when the unablated model passes it.

A set transfers when the objective is above 0.5 on the other task too.

Output: results/fiveway/cross_task/<model>.json; answers are cached in
runs/fiveway/<model>/rep1.25_cross_task/cache_<task>.jsonl, so a requeued job
resumes.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=2:00:00 bash slurm/fiveway_submit.sh cross-task-gxl 48G \\
        fiveway/cross_task.py --model gpt2-xl
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
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.scorer import CLASSES, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective  # noqa: E402

DECODING = "rep1.25"
SOURCES = {"search": lambda model, task: RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json",
           "em_search": lambda model, task: RESULTS / "em_search" / f"{model}_{task}.json"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    # Every knockout set of both searches, both tasks and all three objectives.
    knockout_sets = []
    for source, path_of in SOURCES.items():
        for task in TASKS:
            path = path_of(args.model, task)
            for objective, entry in json.loads(path.read_text())["objectives"].items():
                if entry.get("found_at_stage") is None:
                    continue
                found = entry["pruned"]["circuit"]
                knockout_sets.append({"source": source, "task": task, "objective": objective,
                                      "found_at_stage": entry["found_at_stage"],
                                      "heads": found["heads"], "mlp_blocks": found["mlp_blocks"],
                                      "lesions": found.get("lesions", []), "own_task_value": found["value"]})
    log(f"{args.model}: {len(knockout_sets)} knockout sets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    lesion_names = sorted({name for knockout_set in knockout_sets for name in knockout_set["lesions"]})
    if lesion_names:
        load_lesions(engine, lesion_names)

    run_dir = RUNS / "fiveway" / args.model / "rep1.25_cross_task"
    run_dir.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "decoding": DECODING, "baselines": {}, "sets": []}
    for other_task in TASKS:
        evaluator = Evaluator(engine, scorer, other_task, DECODING, run_dir / f"cache_{other_task}.jsonl")
        baseline = evaluator.records([Intervention()])[0]
        # NOTE: as in search.py: the gate applies only when
        # the unablated model passes it.
        evaluator.gate = baseline["answer"] is not False
        results["baselines"][other_task] = {"gate": evaluator.gate, "story": baseline["story"],
                                            "probabilities": baseline["probabilities"]}
        # NOTE: the unablated answer must be the one the
        # search scored on this task, or the decoding here differs from the search's.
        search_baseline = json.loads(SOURCES["search"](args.model, other_task).read_text())["baseline"]
        log(f"{other_task}: unablated answer same as the search's: {baseline['story'] == search_baseline['story']}")

        moved = [knockout_set for knockout_set in knockout_sets if knockout_set["task"] != other_task]
        interventions = [Intervention.knockout(heads=knockout_set["heads"], mlp_blocks=knockout_set["mlp_blocks"],
                                               lesions=knockout_set["lesions"]) for knockout_set in moved]
        records = evaluator.records(interventions)
        for knockout_set, record in zip(moved, records):
            objective = Objective(knockout_set["objective"], "classifier", evaluator)
            value = objective.value(record)
            results["sets"].append({
                **{key: knockout_set[key] for key in ("source", "task", "objective", "found_at_stage",
                                                       "own_task_value")},
                "n_heads": len(knockout_set["heads"]), "n_mlp_blocks": len(knockout_set["mlp_blocks"]),
                "n_lesions": len(knockout_set["lesions"]),
                "other_task": other_task, "unablated_value": objective.value(baseline), "value": value,
                "transfers": objective.works(value),
                "predicted_class": None if record["probabilities"] is None
                else CLASSES[int(np.argmax(record["probabilities"]))],
                "probabilities": record["probabilities"], "answer": record["answer"], "story": record["story"]})
            log(f"  {knockout_set['source']:9s} {knockout_set['task']} {knockout_set['objective']} -> {other_task}: "
                f"{value if value is None else round(value, 3)} (own task {knockout_set['own_task_value']:.3f})")

    out_path = RESULTS / "cross_task" / f"{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    log(f"{sum(entry['transfers'] for entry in results['sets'])} of {len(results['sets'])} sets transfer; "
        f"wrote {out_path}")


if __name__ == "__main__":
    main()
