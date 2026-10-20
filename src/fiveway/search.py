"""Greedy knockout search for components whose removal makes a model's answer aphasic.

For one generator and one task, and for each of three objectives scored by
the five-way classifier,

    A  P(aphasia) = P(Conduction/Anomic) + P(Broca) + P(Wernicke)
    B  P(Broca)
    C  P(Wernicke)

the search escalates through three stages and stops at the first one that
"works", i.e. finds a knockout whose answer has objective value > 0.5:

  Stage 1, single-layer heads (the earlier exp01). Knock out every head on
    its own; then, in each layer, the layer's top-k heads together
    (k in 1, 2, 3, 5, 8, 12), ranked by their single-head value.
  Stage 2, multi-layer heads, forward sweep (the earlier exp05 with heads
    only). Visit the layers from first to last; in each, greedily add the head
    that raises the objective most, on top of everything already knocked out,
    for at most 4 picks and until 2 picks in a row gain nothing.
  Stage 3, multi-layer heads and MLP, forward sweep (exp05). The same sweep,
    from scratch, where each layer's head picks are followed by greedy picks
    of 256-neuron MLP blocks.

A knockout counts only if the answer is at least 20 characters long and, for
Cinderella, the model does not say "no" to "Do you know the story of
Cinderella?" (the gate), unless the unablated model already says "no", in
which case the gate cannot tell a knockout that destroyed the model's knowledge
of the story from the intact model, and is not applied (results["gate"]).
A successful knockout is then pruned: components are dropped,
one per round and the one whose removal keeps the objective highest, while
the objective stays above 0.5. The pruned set is the reported circuit.

``--objective-kind cosine`` runs the same search with the classifier-free
objective of the cos-sim ablation (scorer.CosineObjective): mean MiniLM cosine
to the target group's human transcripts minus mean cosine to Control's,
"working" above 0.

Every generated answer is cached (``runs/fiveway/<model>/<tag>/``), so the
three objectives share generations, and a requeued job resumes where it
stopped. Results: ``results/fiveway/search/<model>_<task>_<tag>.json``.

Usage:
    bash slurm/fiveway_submit.sh search-l7b 96G fiveway/search.py --model llama2-7b-chat --task cinderella
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
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.prompts import (STORY_TOKENS, YESNO_TOKENS, cut_at_new_question, long_enough,  # noqa: E402
                             parse_yes_no, task_prompt_ids)
from fiveway.scorer import OBJECTIVES, SUCCESS_THRESHOLD, CosineObjective, FiveWayScorer  # noqa: E402

K_VALUES = [1, 2, 3, 5, 8, 12]   # stage 1: joint top-k sizes per layer
MAX_PICKS = 4                    # stages 2 and 3: picks per layer and per component type
PATIENCE = 2                     # ... stop after this many picks in a row without gain
MIN_GAIN = 1e-3                  # a pick "gains" when it raises the objective by more than this


class Evaluator:
    """Generates and scores knockouts, caching every answer on disk."""

    def __init__(self, engine, scorer, task, decoding, cache_path: Path):
        self.engine, self.scorer, self.task, self.decoding = engine, scorer, task, decoding
        self.cosine = CosineObjective(scorer)
        self.yesno_ids, self.story_ids = task_prompt_ids(engine, task)
        self.cache_path = cache_path
        self.gate = True   # main() turns the Cinderella yes/no gate off when the unablated model fails it
        self.used = None   # a set to collect the keys of every record asked for (mask_search.py, cem_search.py)
        self.cache = {}
        if cache_path.exists():
            # NOTE: a job preempted while appending can
            # leave a cut-off last line; it is skipped and regenerated.
            for line in cache_path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.cache[record["key"]] = record
        self.n_generated = 0

    def records(self, interventions: list) -> list:
        """The cached record of every intervention, generating the missing ones in batches."""
        missing, seen = [], set()
        if self.used is not None:
            self.used.update(intervention.key() for intervention in interventions)
        for intervention in interventions:
            key = intervention.key()
            if key not in self.cache and key not in seen:
                missing.append(intervention)
                seen.add(key)
        if missing:
            self._generate(missing)
        return [self.cache[intervention.key()] for intervention in interventions]

    def _generate(self, interventions: list):
        start = time.time()
        answers = [None] * len(interventions)
        yesno_texts = [""] * len(interventions)
        if self.yesno_ids is not None:
            yesno_texts = self.engine.generate(self.yesno_ids, interventions, YESNO_TOKENS, self.decoding)
            answers = [parse_yes_no(text)[0] for text in yesno_texts]
        full_texts = self.engine.generate(self.story_ids, interventions, STORY_TOKENS, self.decoding)
        # NOTE: the base models keep writing question and
        # answer turns of their own after answering; only their answer, the
        # text before the first new question, is scored.
        stories = full_texts if self.engine.spec.is_chat else [cut_at_new_question(t) for t in full_texts]
        # NOTE: every answer long enough to count is scored,
        # whatever its yes/no answer: whether the gate applies is decided per
        # run (valid() below), so the cache serves either choice.
        counted = [long_enough(story) for story in stories]
        scored = [story for story, ok in zip(stories, counted) if ok]
        probabilities = self.scorer.probabilities(scored)
        cosines = self.cosine.values(scored) if scored else {}
        # NOTE: each record keeps its share of the batch's
        # time (generation and scoring), so an experiment's cost can be summed
        # over the answers it used, including answers a preempted job made
        # before its requeue found them in the cache.
        seconds = round((time.time() - start) / len(interventions), 4)
        lines, row = [], 0
        for i, intervention in enumerate(interventions):
            record = {"key": intervention.key(), "heads": [list(c) for c in intervention.heads],
                      "mlp_blocks": [list(c) for c in intervention.mlp_blocks], "lesions": list(intervention.lesions),
                      **({"head_amounts": [list(c) for c in intervention.head_amounts],
                          "lesion_amounts": [list(c) for c in intervention.lesion_amounts]}
                         if intervention.head_amounts or intervention.lesion_amounts else {}),
                      "yesno": yesno_texts[i], "answer": answers[i], "story": stories[i],
                      "full_text": full_texts[i], "probabilities": None, "cosine": None, "seconds": seconds}
            if counted[i]:
                record["probabilities"] = probabilities[row].round(6).tolist()
                record["cosine"] = {name: round(float(values[row]), 6) for name, values in cosines.items()}
                row += 1
            self.cache[record["key"]] = record
            lines.append(json.dumps(record))
        with self.cache_path.open("a") as handle:
            handle.write("\n".join(lines) + "\n")
        self.n_generated += len(interventions)

    def valid(self, record) -> bool:
        """Whether a knockout's answer counts: long enough, and not a "no" to the gate question while the gate applies."""
        return long_enough(record["story"]) and (not self.gate or record["answer"] is not False)


class Objective:
    """Reads one objective's value off a cached record (None when the answer does not count)."""

    def __init__(self, name: str, kind: str, evaluator: Evaluator):
        self.name, self.kind, self.evaluator = name, kind, evaluator
        self.threshold = SUCCESS_THRESHOLD if kind == "classifier" else CosineObjective.SUCCESS_THRESHOLD

    def value(self, record):
        if not self.evaluator.valid(record):
            return None
        if self.kind == "classifier":
            return float(sum(record["probabilities"][column] for column in OBJECTIVES[self.name]))
        return record["cosine"][self.name]

    def works(self, value) -> bool:
        return value is not None and value > self.threshold


def summary(record, objective: Objective) -> dict:
    """What the results file keeps about one knockout: its components, scores and text."""
    lesions = record.get("lesions", [])   # weight lesions (em_search.py); caches from before have none
    amounts = {key: record[key] for key in ("head_amounts", "lesion_amounts") if key in record}   # mask_search.py
    return {"heads": record["heads"], "mlp_blocks": record["mlp_blocks"], **({"lesions": lesions} if lesions else {}),
            **amounts,
            "n_components": len(record["heads"]) + len(record["mlp_blocks"]) + len(lesions)
            + len(record.get("head_amounts", [])) + len(record.get("lesion_amounts", [])),
            "value": objective.value(record), "probabilities": record["probabilities"],
            "cosine": record["cosine"], "answer": record["answer"], "story": record["story"]}


def stage1(evaluator, objective, geometry) -> dict:
    """Single heads, then each layer's top-k single heads together."""
    n_layers, n_heads = geometry["n_layers"], geometry["n_heads"]
    singles = [Intervention.knockout(heads=[(layer, head)]) for layer in range(n_layers) for head in range(n_heads)]
    single_records = evaluator.records(singles)
    # NOTE: a head whose answer does not count gets -inf,
    # so it ranks last; -1 would not do, because a cos-sim margin can be negative.
    values = np.array([objective.value(r) if objective.value(r) is not None else -np.inf for r in single_records])
    values = values.reshape(n_layers, n_heads)
    k_values = [k for k in K_VALUES if k <= n_heads]
    top_k_sets = {}
    for layer in range(n_layers):
        # NOTE: the stable sort keeps head order among ties.
        ranked = list(np.argsort(-values[layer], kind="stable"))
        top_k_sets[layer] = [Intervention.knockout(heads=[(layer, int(h)) for h in ranked[:k]]) for k in k_values]
    top_k_records = evaluator.records([s for layer in range(n_layers) for s in top_k_sets[layer]])
    per_layer = {layer: top_k_records[layer * len(k_values):(layer + 1) * len(k_values)] for layer in range(n_layers)}

    candidates = [r for r in single_records + top_k_records if objective.value(r) is not None]
    working = [r for r in candidates if objective.works(objective.value(r))]
    # The reported knockout: the smallest one that works, the strongest among
    # equally small ones; when none works, the strongest overall.
    if working:
        best = min(working, key=lambda r: (len(r["heads"]), -objective.value(r)))
    else:
        best = max(candidates, key=lambda r: objective.value(r)) if candidates else None
    per_layer_best = []
    for layer in range(n_layers):
        scored = [r for r in per_layer[layer] if objective.value(r) is not None]
        per_layer_best.append(summary(max(scored, key=lambda r: objective.value(r)), objective) if scored else None)
    return {
        "works": bool(working),
        "best": None if best is None else summary(best, objective),
        "n_working_single_heads": int(sum(objective.works(objective.value(r)) for r in single_records)),
        "n_working_sets": len(working),
        "single_head_values": [[None if np.isinf(v) else round(float(v), 6) for v in row] for row in values],
        "per_layer_best": per_layer_best,
    }


def forward_sweep(evaluator, objective, geometry, with_mlp: bool, baseline_value: float, log, lesions=()) -> dict:
    """Stages 2 and 3: the layer-by-layer greedy sweep of the earlier exp05.

    ``lesions``: weight lesions (Intervention.lesions) in every knockout of
    the sweep, the base it adds heads and blocks to (em_search.py's stage 3).

    NOTE: as in exp05, every step commits its best
    candidate even when it gains nothing ("allow ties"), and a layer stops
    after PATIENCE such picks in a row; the gain is measured against the best
    value reached so far. Unlike exp05, the sweep stops as soon as a committed
    knockout works, since the stages escalate only until one works.
    """
    n_layers, n_heads, n_blocks = geometry["n_layers"], geometry["n_heads"], geometry["n_blocks"]
    heads, blocks, trajectory = [], [], []
    best_so_far = baseline_value if baseline_value is not None else 0.0
    for layer in range(n_layers):
        kinds = [("head", n_heads)] + ([("block", n_blocks)] if with_mlp else [])
        for kind, n_candidates in kinds:
            stale = 0
            for _ in range(MAX_PICKS):
                taken = {c for c in (heads if kind == "head" else blocks) if c[0] == layer}
                options = [(layer, i) for i in range(n_candidates) if (layer, i) not in taken]
                interventions = [Intervention.knockout(heads=heads + [option], mlp_blocks=blocks, lesions=lesions)
                                 if kind == "head"
                                 else Intervention.knockout(heads=heads, mlp_blocks=blocks + [option], lesions=lesions)
                                 for option in options]
                records = evaluator.records(interventions)
                scored = [(objective.value(r), option, r) for option, r in zip(options, records)
                          if objective.value(r) is not None]
                if not scored:
                    break
                value, option, record = max(scored, key=lambda item: item[0])
                gain = value - best_so_far
                if gain > MIN_GAIN:
                    best_so_far, stale = value, 0
                else:
                    stale += 1
                (heads if kind == "head" else blocks).append(option)
                trajectory.append({"layer": layer, "kind": kind, "added": list(option), "value": value,
                                   "gain": gain, "n_components": len(heads) + len(blocks)})
                if objective.works(value):
                    log(f"    works after L{layer} {kind} {option}: value {value:.3f}, "
                        f"{len(heads)} heads, {len(blocks)} blocks")
                    return {"works": True, "found": summary(record, objective), "trajectory": trajectory}
                if stale >= PATIENCE:
                    break
        log(f"    layer {layer:2d}: {len(heads)} heads, {len(blocks)} blocks, best value so far {best_so_far:.4f}")
    final = evaluator.records([Intervention.knockout(heads=heads, mlp_blocks=blocks, lesions=lesions)])[0]
    return {"works": False, "found": summary(final, objective), "trajectory": trajectory}


def knockout_of(components: list) -> Intervention:
    """The knockout of ("head", (layer, head)), ("block", (layer, block)) and ("lesion", name) components."""
    return Intervention.knockout(heads=[c for kind, c in components if kind == "head"],
                                 mlp_blocks=[c for kind, c in components if kind == "block"],
                                 lesions=[c for kind, c in components if kind == "lesion"])


def generation_seconds(evaluator, keys) -> float:
    """The summed generation and scoring time of the cached answers with these keys (0 for answers made before
    records kept their time)."""
    return round(sum(evaluator.cache[key].get("seconds", 0.0) for key in keys if key in evaluator.cache), 1)


def prune(evaluator, objective, found: dict, log) -> dict:
    """Drop components one at a time while the knockout still works; returns the minimal set."""
    components = ([("head", tuple(c)) for c in found["heads"]] + [("block", tuple(c)) for c in found["mlp_blocks"]]
                  + [("lesion", name) for name in found.get("lesions", [])])
    steps = []
    while len(components) > 1:
        candidates = [knockout_of(components[:i] + components[i + 1:]) for i in range(len(components))]
        records = evaluator.records(candidates)
        values = [objective.value(r) for r in records]
        keeping = [(value, i) for i, value in enumerate(values) if objective.works(value)]
        if not keeping:
            break
        value, i = max(keeping)
        removed = components[i][1]
        steps.append({"removed": [components[i][0], removed if isinstance(removed, str) else list(removed)],
                      "value": value})
        components = components[:i] + components[i + 1:]
    record = evaluator.records([knockout_of(components)])[0]
    log(f"    pruned to {len(components)} components: value {objective.value(record):.3f}")
    return {"circuit": summary(record, objective), "steps": steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=["cinderella", "sandwich"])
    parser.add_argument("--decoding", default="rep1.25", choices=["greedy", "rep1.25"])
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--objective-kind", default="classifier", choices=["classifier", "cosine"])
    parser.add_argument("--all-stages", action="store_true",
                        help="run every stage even after one works (for comparisons)")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    tag = args.tag or f"{args.decoding}_{args.objective_kind}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    geometry = engine.geometry
    log(f"{args.model} on {device}: {geometry}; task {args.task}, decoding {args.decoding}, "
        f"objectives {args.objectives} ({args.objective_kind})")

    run_dir = RUNS / "fiveway" / args.model / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, args.decoding, run_dir / f"cache_{args.task}.jsonl")
    log(f"cache: {len(evaluator.cache)} answers already generated")
    out_path = RESULTS / "search" / f"{args.model}_{args.task}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": vars(args), "geometry": geometry, "objectives": {}}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    baseline = evaluator.records([Intervention()])[0]
    # NOTE: the gate exists to reject knockouts that make the
    # model deny knowing the story. A model that denies it already when intact
    # (Pythia-6.9B answers "No." and goes on writing its own dialogue) would
    # have almost every knockout rejected, and the search would only look for
    # components that flip "No" to "Yes"; for such a model the gate is off.
    evaluator.gate = baseline["answer"] is not False
    results["gate"] = evaluator.gate
    results["baseline"] = {"answer": baseline["answer"], "yesno": baseline["yesno"], "story": baseline["story"],
                           "full_text": baseline["full_text"], "valid": evaluator.valid(baseline),
                           "probabilities": baseline["probabilities"], "cosine": baseline["cosine"]}
    log(f"baseline: gate {'on' if evaluator.gate else 'off (the unablated model answers no)'}, "
        f"valid {evaluator.valid(baseline)}, probabilities {baseline['probabilities']}\n    {baseline['story'][:300]!r}")

    for name in args.objectives:
        if results["objectives"].get(name, {}).get("done"):
            log(f"objective {name}: already done")
            continue
        objective = Objective(name, args.objective_kind, evaluator)
        baseline_value = objective.value(baseline)
        entry = {"baseline_value": baseline_value, "stages": {}}
        log(f"objective {name}: baseline value {baseline_value}")

        entry["stages"]["1_single_layer_heads"] = stage1(evaluator, objective, geometry)
        found_stage = "1_single_layer_heads" if entry["stages"]["1_single_layer_heads"]["works"] else None
        best = entry["stages"]["1_single_layer_heads"]["best"]
        log(f"  stage 1: works={found_stage is not None}, best {None if best is None else best['value']}")

        for stage, with_mlp in (("2_forward_heads", False), ("3_forward_heads_mlp", True)):
            if found_stage and not args.all_stages:
                break
            log(f"  {stage}:")
            entry["stages"][stage] = forward_sweep(evaluator, objective, geometry, with_mlp, baseline_value, log)
            if entry["stages"][stage]["works"] and not found_stage:
                found_stage = stage
            log(f"  {stage}: works={entry['stages'][stage]['works']}, value {entry['stages'][stage]['found']['value']}")

        entry["found_at_stage"] = found_stage
        if found_stage:
            found = (entry["stages"][found_stage]["best"] if found_stage.startswith("1")
                     else entry["stages"][found_stage]["found"])
            entry["pruned"] = prune(evaluator, objective, found, log)
        entry["done"] = True
        entry["n_generated_so_far"] = len(evaluator.cache)
        results["objectives"][name] = entry
        out_path.write_text(json.dumps(results, indent=1))
        log(f"objective {name} done: found at {found_stage}; wrote {out_path}")

    log(f"finished; {evaluator.n_generated} new answers generated, {len(evaluator.cache)} in the cache")


if __name__ == "__main__":
    main()
