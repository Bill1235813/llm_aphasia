"""Activation patching on a knockout circuit: which of its components carry the effect?

search.py finds, for one generator, one task and one objective (A = P(aphasia),
B = P(Broca), C = P(Wernicke) under the five-way classifier), a small set of
attention heads and 256-neuron MLP blocks whose joint knockout makes the
model's answer "work" (objective value > 0.5): the circuit, stored in
results/fiveway/search/<model>_<task>_rep1.25_classifier.json under
objectives[X].pruned.circuit. This script takes the circuit apart. Every
condition below is one Intervention, all of them are generated together in
batches, and every answer is made and scored as search.py makes and scores
it: the task's prompts, rep1.25 decoding, for Cinderella the yes/no question
first, base-model answers cut at the first question they write themselves,
the five-way scorer.

    clean               no intervention: the model's own answer
    corrupt             the whole circuit knocked out: the circuit's answer
    restore_<c>         corrupt, with component c's clean activations written
                        back at every prompt position (activation patching,
                        "denoising"): does c's clean signal undo the knockout?
    leave_one_out_<c>   the circuit minus c knocked out: is c necessary?
    alone_<c>           only c knocked out: is c enough on its own?
    restore_all         corrupt, with every component's clean activations written
                        back at the prompt positions: the circuit knocked out only
                        while the answer is written. Does the whole circuit act on
                        how the prompt is read? In a circuit of many components no
                        single restore_<c> can be expected to undo the knockout, so
                        this asks the question of the whole.

--source em_search takes em_search.py's circuits instead, and --source
grown_heads the grown-heads sets: em_attention.py's space 2b, or grown_heads.py's
classifier sets for the models em_attention.py did not run (Qwen3.5), as
grown_rescue.py reads them.

NOTE: restoring and leaving out ask different questions. In
restore_<c>, c computes nothing on the prompt: its output there is copied
from the clean run, whatever the knockouts upstream did to its inputs, and on
the generated positions c stays knocked out, because they have no clean
counterpart (engine.py). In leave_one_out_<c>, c runs normally everywhere but
reads a residual stream that the other knockouts have changed.

Components are named L<layer>H<head> (attention heads) and L<layer>MLP<block>
(256-neuron MLP blocks), and a condition's name ends in its component's name.

Output: RESULTS/patching/<model>_<task>_<objective>.json (em_search/patching/ or
grown_heads/patching/ for the other sources) with every condition
(its intervention, the gate's text and answer, the answer with and without
the cut, five-way probabilities, objective values), a per-component summary,
and whether clean and corrupt reproduce the search's own answers; the same as
a table on stdout. Generation takes a few batches, so the job is short: a
rerun whose file already lists the same conditions only prints the table.

Usage:
    bash slurm/fiveway_submit.sh patch-p1b-cind-A 32G fiveway/patching.py --model pythia-1b --task cinderella --objective A
  Every grown-heads set of a model in one process, the model loaded once:
    bash slurm/fiveway_submit.sh patch-grown-p1b 32G fiveway/patching.py --model pythia-1b \\
        --task cinderella sandwich --objective A B C --source grown_heads
  CPU smoke test with a circuit from the command line, writing outside results/:
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh patch-test 32G fiveway/patching.py --model pythia-1b \\
        --task cinderella --objective A --heads 0:1 1:2 --mlp-blocks 2:3 --output-dir runs/fiveway/patching/test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, ROOT  # noqa: E402
from fiveway.prompts import (STORY_TOKENS, YESNO_TOKENS, cut_at_new_question, long_enough,  # noqa: E402
                             parse_yes_no, task_prompt_ids)
from fiveway.scorer import CLASSES, OBJECTIVES, FiveWayScorer, objective_values, works_finally  # noqa: E402

DECODING = "rep1.25"
SEARCH_TAG = "rep1.25_classifier"   # the search results whose circuits are patched (search.py's default tag)


# --- the circuit and its components (benchmarks.py uses these too) ------------

def parse_pairs(pairs: list) -> list:
    """Command-line components "3:2" -> (3, 2), a (layer, head) or (layer, block) pair."""
    return [tuple(int(number) for number in pair.split(":")) for pair in pairs]


def read_circuit(model: str, task: str, objective: str, heads: list, mlp_blocks: list, source: str = "search"):
    """The circuit to take apart, or None when the search found none.

    ``source``: "search" (search.py's circuits), "em_search" (em_search.py's,
    which can hold weight lesions of MLP matrices) or "grown_heads" (the
    grown-heads sets, heads only). Components given on the command line
    (``--heads 0:1 1:2``) replace the search's circuit, for tests. The
    unablated answer and the circuit's answer that the search stored come
    along, for main()'s checks.
    """
    if heads or mlp_blocks:
        return {"heads": parse_pairs(heads), "mlp_blocks": parse_pairs(mlp_blocks), "lesions": [],
                "source": "command line"}
    if source == "grown_heads":
        # NOTE: the same set grown_rescue.py steers against: grown_heads.py's classifier set
        # where one exists (Qwen3.5), else em_attention.py's space 2b (the seven older models).
        grown_path = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
        found_path = grown_path if grown_path.exists() else RESULTS / "em_attention" / f"{model}_{task}.json"
        found = json.loads(found_path.read_text())
        row = found["reported"][objective] if grown_path.exists() else found["reported"][objective]["2b_heads_grown"]
        # NOTE: a set counts by the final-answer rule (scorer.works_finally), which also takes
        # a target that is the classifier's top class below 0.5; such a set stopped growing unpruned.
        if not works_finally(objective, row["value"], row["probabilities"]):
            return None
        return {"heads": [tuple(pair) for pair in row["heads"]], "mlp_blocks": [], "lesions": row["lesions"],
                "source": str(found_path), "found_at_stage": "2b_heads_grown", "search_value": row["value"],
                "baseline_story": found["baseline"]["story"], "circuit_story": row["story"]}
    if source == "em_search":
        search_path = RESULTS / "em_search" / f"{model}_{task}.json"
    else:
        search_path = RESULTS / "search" / f"{model}_{task}_{SEARCH_TAG}.json"
    search = json.loads(search_path.read_text())
    entry = search["objectives"][objective]
    if entry["found_at_stage"] is None:
        return None
    circuit = entry["pruned"]["circuit"]
    return {"heads": [tuple(pair) for pair in circuit["heads"]],
            "mlp_blocks": [tuple(pair) for pair in circuit["mlp_blocks"]], "lesions": circuit.get("lesions", []),
            "source": str(search_path), "found_at_stage": entry["found_at_stage"], "search_value": circuit["value"],
            "baseline_story": search["baseline"]["story"], "circuit_story": circuit["story"]}


def components_of(circuit: dict) -> list:
    """The circuit's components: ("head" | "block", (layer, index)) pairs, then ("lesion", matrix name)."""
    return ([("head", pair) for pair in circuit["heads"]] + [("block", pair) for pair in circuit["mlp_blocks"]]
            + [("lesion", name) for name in circuit.get("lesions", [])])


def component_name(kind: str, pair) -> str:
    """L3H5 (a head), L3MLP5 (a 256-neuron MLP block) or L3.up (a weight lesion of a matrix)."""
    if kind == "lesion":
        return pair
    layer, index = pair
    return f"L{layer}H{index}" if kind == "head" else f"L{layer}MLP{index}"


def restore(corrupt: Intervention, kind: str, pair: tuple) -> Intervention:
    """The corrupt knockout with one component's clean activations written back at the prompt positions.

    NOTE: Intervention is a frozen dataclass, so it cannot be
    changed in place; dataclasses.replace returns a copy with the given
    fields replaced.
    """
    if kind == "head":
        return dataclasses.replace(corrupt, patch_heads=(pair,))
    if kind == "lesion":
        return dataclasses.replace(corrupt, patch_lesions=(pair,))
    return dataclasses.replace(corrupt, patch_blocks=(pair,))


def leave_out(circuit: dict, kind: str, pair: tuple) -> Intervention:
    """The knockout of the whole circuit except one component."""
    heads = [head for head in circuit["heads"] if (kind, pair) != ("head", head)]
    blocks = [block for block in circuit["mlp_blocks"] if (kind, pair) != ("block", block)]
    lesions = [name for name in circuit.get("lesions", []) if (kind, pair) != ("lesion", name)]
    return Intervention.knockout(heads=heads, mlp_blocks=blocks, lesions=lesions)


def alone(kind: str, pair: tuple) -> Intervention:
    """The knockout of one component only."""
    if kind == "head":
        return Intervention.knockout(heads=[pair])
    if kind == "lesion":
        return Intervention.knockout(lesions=[pair])
    return Intervention.knockout(mlp_blocks=[pair])


# --- generation and scoring ---------------------------------------------------

def generate_and_score(engine, scorer, task: str, interventions: list) -> list:
    """The gate, answer and five-way scores of every intervention, made as search.py's Evaluator makes them.

    For Cinderella the yes/no question is asked first under the same
    intervention. Whether an answer counts is decided in main() (counts()),
    once the clean answer says whether the gate applies. Every answer long
    enough is scored, so a condition that makes the model say no can still
    be read; main() gives an answer that does not count no objective value,
    as the search does.
    """
    yesno_ids, story_ids = task_prompt_ids(engine, task)
    yesno_texts = [""] * len(interventions)
    answers = [None] * len(interventions)
    if yesno_ids is not None:
        yesno_texts = engine.generate(yesno_ids, interventions, YESNO_TOKENS, DECODING)
        answers = [parse_yes_no(text)[0] for text in yesno_texts]
    full_texts = engine.generate(story_ids, interventions, STORY_TOKENS, DECODING)
    # NOTE: the base models keep writing question and
    # answer turns of their own after answering; only their answer, the text
    # before the first new question, is scored, as in search.py.
    stories = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
    scored = [story for story in stories if long_enough(story)]
    # NOTE: probabilities: (n_scored, 5), columns in CLASSES order;
    # objectives: {"A" | "B" | "C": (n_scored,)}.
    probabilities = scorer.probabilities(scored)
    objectives = objective_values(probabilities)

    records, row = [], 0
    for yesno, answer, full_text, story in zip(yesno_texts, answers, full_texts, stories):
        record = {"yesno": yesno, "answer": answer, "story": story, "full_text": full_text,
                  "probabilities": None, "objectives": None}
        if long_enough(story):
            record["probabilities"] = probabilities[row].round(6).tolist()
            record["objectives"] = {name: round(float(values[row]), 6) for name, values in objectives.items()}
            row += 1
        records.append(record)
    return records


def difference(first, second):
    """first - second, or None when either value is missing (an answer that does not count)."""
    if first is None or second is None:
        return None
    return round(first - second, 6)


RECORD_FIELDS = ("yesno", "answer", "story", "full_text", "probabilities", "objectives")   # generate_and_score's


def counts(record: dict, gate: bool) -> bool:
    """Whether an answer counts, as search.Evaluator.valid decides: long enough (prompts.long_enough), and not a
    "no" to the yes/no question while the gate applies."""
    return long_enough(record["story"]) and (not gate or record["answer"] is not False)


def print_table(results: dict):
    objective = results["config"]["objective"]
    print(f"\n{results['config']['model']} {results['config']['task']}, objective {objective} "
          f"(value > 0.5 works), circuit {results['circuit']['heads']} heads, "
          f"{results['circuit']['mlp_blocks']} MLP blocks", flush=True)
    header = f"{'condition':<26}{'gate':>6}{'valid':>7}{objective:>8} " + "".join(f"{name[:10]:>11}" for name in CLASSES)
    print(header + "   answer", flush=True)
    for condition in results["conditions"]:
        gate = {True: "yes", False: "no", None: "-"}[condition["answer"]]
        value = "-" if condition["value"] is None else f"{condition['value']:.3f}"
        probabilities = condition["probabilities"] or [None] * len(CLASSES)
        cells = "".join("          -" if probability is None else f"{probability:>11.3f}" for probability in probabilities)
        print(f"{condition['name']:<26}{gate:>6}{str(condition['valid']):>7}{value:>8} {cells}   "
              f"{condition['story'][:70]!r}", flush=True)
    print(f"\n{'component':<14}{'restore':>9}{'leave out':>11}{'alone':>8}{'restore drop':>14}{'leave-out drop':>16}",
          flush=True)
    for component in results["components"]:
        cells = [component[key] for key in ("restore_value", "leave_one_out_value", "alone_value",
                                            "restore_drop", "leave_one_out_drop")]
        cells = ["-" if value is None else f"{value:.3f}" for value in cells]
        print(f"{component['name']:<14}{cells[0]:>9}{cells[1]:>11}{cells[2]:>8}{cells[3]:>14}{cells[4]:>16}", flush=True)
    print(f"\nchecks: {results['checks']}", flush=True)


def patch_circuit(args, task: str, objective: str, loaded: dict, log) -> None:
    """Take one circuit apart: generate and score every condition, write the results file, print the table.

    ``loaded`` holds the engine and the scorer once a circuit has needed them,
    so the circuits of several tasks and objectives share one model in memory.
    """
    circuit = read_circuit(args.model, task, objective, args.heads, args.mlp_blocks, args.source)
    if circuit is None:
        log(f"the search found no circuit for {args.model} {task} objective {objective}: nothing to patch")
        return
    components = components_of(circuit)
    names = [component_name(kind, pair) for kind, pair in components]
    corrupt = Intervention.knockout(heads=circuit["heads"], mlp_blocks=circuit["mlp_blocks"],
                                    lesions=circuit["lesions"])
    # The conditions in the order of the table: (name, kind, component name, intervention).
    conditions = [("clean", "clean", None, Intervention()), ("corrupt", "corrupt", None, corrupt)]
    conditions += [(f"restore_{name}", "restore", name, restore(corrupt, kind, pair))
                   for name, (kind, pair) in zip(names, components)]
    conditions += [(f"leave_one_out_{name}", "leave_one_out", name, leave_out(circuit, kind, pair))
                   for name, (kind, pair) in zip(names, components)]
    conditions += [(f"alone_{name}", "alone", name, alone(kind, pair))
                   for name, (kind, pair) in zip(names, components)]
    conditions.append(("restore_all", "restore_all", None,
                       dataclasses.replace(corrupt, patch_heads=tuple(circuit["heads"]),
                                           patch_blocks=tuple(circuit["mlp_blocks"]),
                                           patch_lesions=tuple(circuit["lesions"]))))
    keys = {name: intervention.key() for name, _, _, intervention in conditions}

    # NOTE: ROOT / path is path itself when path is absolute, so
    # --output-dir may be given relative to the repository or in full.
    out_path = ROOT / args.output_dir / f"{args.model}_{task}_{objective}.json"
    stored = json.loads(out_path.read_text()) if out_path.exists() else None
    if stored is not None and stored["keys"] == keys and "gate" in stored:
        log(f"{out_path} already holds these {len(conditions)} conditions")
        print_table(stored)
        return

    if stored is not None and stored["keys"] == keys:
        # NOTE: a file written before the gate followed
        # search.py holds the right answers; only whether each one counts is
        # decided again below, so nothing is generated.
        log(f"{out_path} holds these conditions from before the gate followed search.py; counting them again")
        config = stored["config"]
        records = [{field: condition[field] for field in RECORD_FIELDS} for condition in stored["conditions"]]
    else:
        if not loaded:
            loaded["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded["engine"] = Engine(args.model, loaded["device"])
            loaded["scorer"] = FiveWayScorer(loaded["device"])
        device, engine, scorer = loaded["device"], loaded["engine"], loaded["scorer"]
        if circuit["lesions"]:
            load_lesions(engine, circuit["lesions"])
        log(f"{args.model} on {device}; task {task}, objective {objective}; circuit from "
            f"{circuit['source']}: {names}; {len(conditions)} conditions")
        records = generate_and_score(engine, scorer, task, [intervention for *_, intervention in conditions])
        log(f"generated and scored {len(records)} answers")
        config = {"model": args.model, "task": task, "objective": objective, "decoding": DECODING,
                  "story_tokens": STORY_TOKENS, "yesno_tokens": YESNO_TOKENS, "device": str(device),
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}

    # NOTE: as in search.py, the gate applies only when the
    # unablated model passes it. Pythia-6.9B answers "no" to "Do you know the
    # story of Cinderella?" unablated; with the gate on, every condition of
    # its circuits would be rejected, the clean one included.
    gate = records[0]["answer"] is not False   # conditions[0] is the clean run
    results = {
        "config": config,
        "circuit": {**circuit, "heads": [list(pair) for pair in circuit["heads"]],
                    "mlp_blocks": [list(pair) for pair in circuit["mlp_blocks"]]},
        "keys": keys,
        "gate": gate,
        "conditions": [],
    }
    for (name, kind, component, intervention), record in zip(conditions, records):
        record["valid"] = counts(record, gate)
        # An answer that does not count has no objective value, as in the search.
        value = record["objectives"][objective] if record["valid"] else None
        results["conditions"].append({"name": name, "kind": kind, "component": component, "key": intervention.key(),
                                      "value": value, **record})

    by_name = {condition["name"]: condition for condition in results["conditions"]}
    corrupt_value = by_name["corrupt"]["value"]
    results["components"] = []
    for name, (kind, pair) in zip(names, components):
        # NOTE: a weight lesion's "index" is its matrix kind ("up", "down").
        layer, index = (int(pair[1:].split(".")[0]), pair.split(".")[1]) if kind == "lesion" else pair
        restore_value = by_name[f"restore_{name}"]["value"]
        leave_one_out_value = by_name[f"leave_one_out_{name}"]["value"]
        results["components"].append({
            "name": name, "kind": kind, "layer": layer, "index": index,
            "restore_value": restore_value, "leave_one_out_value": leave_one_out_value,
            "alone_value": by_name[f"alone_{name}"]["value"],
            # How far restoring c, or sparing c, brings the objective back down from the corrupt value.
            "restore_drop": difference(corrupt_value, restore_value),
            "leave_one_out_drop": difference(corrupt_value, leave_one_out_value),
        })

    # NOTE: the clean and corrupt conditions are the
    # search's baseline and pruned circuit, so their answers should come out
    # word for word as the search stored them. A difference means the two runs
    # computed differently (another GPU kind, or CPU against GPU), which can
    # move a near-tie in the logits; then the search's circuit is being patched
    # on answers it was not found on.
    results["checks"] = {}
    if circuit["source"] != "command line":
        results["checks"] = {
            "clean_answer_equals_search_baseline": by_name["clean"]["story"] == circuit["baseline_story"],
            "corrupt_answer_equals_search_circuit": by_name["corrupt"]["story"] == circuit["circuit_story"],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    log(f"wrote {out_path}")
    print_table(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, nargs="+", choices=TASKS,
                        help="one task or both: every task and objective given is patched in turn, the model "
                             "loaded once")
    parser.add_argument("--objective", required=True, nargs="+", choices=sorted(OBJECTIVES))
    parser.add_argument("--heads", nargs="*", default=[],
                        help="layer:head pairs that replace the search's circuit (for tests)")
    parser.add_argument("--mlp-blocks", nargs="*", default=[],
                        help="layer:block pairs of 256-neuron MLP blocks, likewise")
    parser.add_argument("--source", default="search", choices=["search", "em_search", "grown_heads"],
                        help="whose circuit: search.py's, em_search.py's or the grown-heads sets")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where the results file goes (default: results/fiveway/patching, or the source's "
                             "own patching/ folder); a test points it outside results/")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = RESULTS / "patching" if args.source == "search" else RESULTS / args.source / "patching"
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    loaded = {}
    for task in args.task:
        for objective in args.objective:
            patch_circuit(args, task, objective, loaded, log)


if __name__ == "__main__":
    main()
