"""Make every reported knockout set's answer again from scratch and compare it with the stored one.

The learned-mask and cross-entropy searches (mask_search.py, cem_search.py)
read every answer from a cache once it is made, generated in batches next to
other knockouts. Here, for every model and task, the sets they report (the
soft mask, the yes/no set after pruning, and the cross-entropy method's set
after pruning) are generated again in a fresh process with an empty cache,
and the answer text and the classifier's objective are compared with the
results files. The engine's padding makes an answer independent of its batch
(test_engine.py, test_learned_searches.py), so every answer should match
exactly; a run on a different GPU type than the search may change rare last
digits of float32 sums and with them a greedy choice, which this check would
show.

With --source em_attention it does the same for the reported set of each of
em_attention.py's six spaces (attention-matrix lesions included), and with
--source grown_heads for grown_heads.py's sets, each judged by its own
objective (the classifier's, or the cos-sim margin).

Output: results/fiveway/report/learned_verification.json (or
em_attention_verification.json, grown_heads_verification.json) and a line per
model and task.

Usage:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh \\
        verify-learned 160G fiveway/verify_learned.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import find_submodules, load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.knockout_spaces import mlp_matrix_names, space_components  # noqa: E402
from fiveway.mask_search import soft_intervention  # noqa: E402
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, knockout_of  # noqa: E402


def set_intervention(result: dict) -> Intervention:
    return knockout_of([("head", tuple(c)) for c in result["heads"]] + [("lesion", c) for c in result.get("lesions", [])])


def learned_checks(engine, model: str, task: str) -> list:
    """(label, intervention, objective, stored result) for every set the learned-mask and cross-entropy searches report."""
    checks = []
    mask = json.loads((RESULTS / "mask_search" / f"{model}_{task}.json").read_text())
    cem = json.loads((RESULTS / "cem_search" / f"{model}_{task}.json").read_text())
    for objective, spaces in mask["objectives"].items():
        for space, entry in spaces.items():
            components = space_components(engine, space)
            checks.append((f"mask soft {objective} {space}", soft_intervention(components, entry["soft"]["amounts"]),
                           objective, entry["soft"]["result"]))
            binary = entry["binary"].get("pruned", {}).get("circuit", entry["binary"]["result"])
            checks.append((f"mask binary {objective} {space}", set_intervention(binary), objective, binary))
    for objective, spaces in cem["objectives"].items():
        for space, entry in spaces.items():
            result = entry["pruned"]["circuit"] if entry["works"] else entry["best"]
            if result is not None:
                checks.append((f"cem {objective} {space}", set_intervention(result), objective, result))
    return checks


def judged_again(name: str, stored: dict, record: dict, evaluator) -> tuple:
    """(value, same verdict as stored) of a regenerated answer, under the objective that found the set.

    NOTE: each set is judged by the objective that found it:
    the classifier's (above 0.5) or, for grown_heads.py's cos-sim sets, the
    cos-sim margin (above 0).
    NOTE: the Objective is made here, not in main(): kept
    in a variable of main() it would hold the evaluator, and through it the
    engine, past main()'s del, and the next model would load beside this one.
    """
    objective = Objective(name, stored.get("kind", "classifier"), evaluator)
    value = objective.value(record)
    return value, objective.works(value) == objective.works(stored["value"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[model for model in MODELS if not model.startswith("qwen3.5")],
                        help="the seven older models by default; name the Qwen3.5 models with --source grown_heads")
    parser.add_argument("--source", default="learned", choices=["learned", "em_attention", "grown_heads"])
    args = parser.parse_args()
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name()
    scorer = FiveWayScorer(device)
    out_path = RESULTS / "report" / f"{args.source}_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        engine = Engine(model, device)
        if args.source == "learned":
            load_lesions(engine, mlp_matrix_names(engine))
        elif args.source == "em_attention":
            load_lesions(engine, [s.name for s in find_submodules(engine.model)
                                  if s.kind in ("q", "k", "v", "o", "gate", "up", "down")])
        for task in TASKS:
            # (label, intervention, objective, stored summary of the set's answer)
            checks = []
            if args.source == "em_attention":
                results = json.loads((RESULTS / "em_attention" / f"{model}_{task}.json").read_text())
                gate = results["gate"]
                for objective, spaces in results["reported"].items():
                    for space, row in spaces.items():
                        checks.append((f"em {objective} {space}", set_intervention(row), objective, row))
            elif args.source == "grown_heads":
                for kind in ("classifier", "cosine"):
                    path = RESULTS / "grown_heads" / f"{model}_{task}_{kind}.json"
                    if not path.exists():
                        continue
                    results = json.loads(path.read_text())
                    gate = results["gate"]
                    for objective, row in results["reported"].items():
                        checks.append((f"grown {kind} {objective}", set_intervention(row), objective,
                                       {**row, "kind": kind}))
                if not checks:
                    continue
            else:
                checks = learned_checks(engine, model, task)
                gate = json.loads((RESULTS / "mask_search" / f"{model}_{task}.json").read_text())["gate"]
            with tempfile.TemporaryDirectory() as folder:
                evaluator = Evaluator(engine, scorer, task, "rep1.25", Path(folder) / "cache.jsonl")
                evaluator.gate = gate
                records = evaluator.records([intervention for _, intervention, _, _ in checks])
            rows = []
            for (label, intervention, name, stored), record in zip(checks, records):
                value, same_verdict = judged_again(name, stored, record, evaluator)
                rows.append({"label": label, "key": intervention.key(), "same_text": record["story"] == stored["story"],
                             "stored_value": stored["value"], "value": value, "same_verdict": same_verdict})
            entry = {"device": device_name, "n": len(rows), "n_same_text": sum(row["same_text"] for row in rows),
                     "n_same_verdict": sum(row["same_verdict"] for row in rows), "rows": rows}
            print(f"{model} {task}: {entry['n_same_text']} of {len(rows)} answers identical, "
                  f"{entry['n_same_verdict']} same verdict", flush=True)
            for row in rows:
                if not row["same_text"]:
                    print(f"   differs: {row['label']}: stored {row['stored_value']}, again {row['value']}", flush=True)
            # NOTE: the file is read again just before each write, so
            # jobs that verify different models one after another add to it rather than
            # overwrite each other's entries.
            report = json.loads(out_path.read_text()) if out_path.exists() else {}
            report[f"{model}_{task}"] = entry
            temporary = out_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=1))
            temporary.replace(out_path)
        # NOTE: the evaluator keeps a reference to the engine, so
        # both must go before the next model is loaded, or its memory is never freed.
        # NOTE: del alone is not enough: the engine's hooks are closures
        # over the engine, registered on the model's modules, so engine -> model ->
        # hook -> engine is a reference cycle, which only the garbage collector frees.
        del engine, evaluator
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
