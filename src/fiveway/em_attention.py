"""em_search with the attention matrices lesioned the same way as the MLP matrices: lesion type or location?

em_search.py found knockout sets in three spaces, by greedy search:

    1 MLP lesions only       stage 1: the top 1% of weights of an MLP matrix (by |w dL/dw| on
                             Control narratives, em_baseline.py) zeroed; matrices ranked by their
                             single-lesion value, then the top 1, 2, 3, ... lesioned together
                             until the objective passes 0.5
    2 heads only             stage 2: search.py's single-layer heads (whole heads zeroed)
    3 MLP lesions + heads    stage 3, when stage 1 fails: heads swept layer by layer on top of the
                             best stage-1 lesion

MLP lesions worked where heads did not (Llama-2, GPT-2 XL), but the two are not
removed alike: a head is removed whole, a lesion takes out the 1% of a
matrix's weights that matter most for fluent Control speech. This script
applies the same top-1% lesions to the attention matrices (q, k, v and o) and
searches them the way em_search.py searched the MLP's:

    4 attention lesions only           stage 1's growth over the attention matrices' lesions
                                       (compare with 1, the same procedure, and with 2)
    5 MLP lesions + attention lesions  stage 1 (reused from em_search.py); when it fails,
                                       attention lesions swept layer by layer on top of the best
                                       stage-1 lesion, as stage 3 sweeps heads (compare with 3)
    2b heads, grown                    stage 1's growth over whole heads, ranked by search.py's
                                       single-head values: space 2 with space 4's procedure, so
                                       that 4 against 2b compares only how attention is removed

A working set is pruned as in search.py: one component per round, the one
whose removal keeps the objective highest, while it stays above 0.5. Space 5
is pruned exactly so, as em_search.py pruned space 3. The grown sets of
spaces 4 and 2b can hold hundreds of components, too many for one removal per
round, so they are pruned in batches (batched_prune(), after shrink() for
sets of more than PRESHRINK_ABOVE); the result is again a set none of whose
components can go alone, unless its pruning reaches PRUNE_BUDGET answers (a
fragile set that loses a few components per round), when it is reported as it
stands and flagged as not minimal. Space 2 reports search.py's pruned set.
Answers are made and scored as in em_search.py: rep1.25 decoding, the
Cinderella gate when the unablated model passes it, the five-way classifier.

Then every reported set of all six spaces (a working set after pruning, else
the best set) is checked the same way, so that the spaces compare on equal
terms: its answer to the other task, the twenty TAB items (word and sentence
comprehension and repetition scored by rule, connected text by the
classifier) and MMLU, as benchmarks.py checks a set.

Output: results/fiveway/em_attention/<model>_<task>.json and, in benchmarks.py's
format (for llm_judge.py), results/fiveway/em_attention/benchmarks/<model>_<task>.json.
Answers are cached in runs/fiveway/<model>/rep1.25_em_attention/, so a requeued
job resumes. Run after em_baseline.py (top1pct.pt), search.py and em_search.py.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=12:00:00 bash slurm/fiveway_submit.sh em-attn-p1b-cind 48G \\
        fiveway/em_attention.py --model pythia-1b --task cinderella
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
from fiveway.benchmarks import (CONNECTED_TEXT_TOKENS, DECODING as TAB_DECODING, MMLU_SUBJECTS,  # noqa: E402
                                SHORT_ANSWER_TOKENS, TAB_ITEMS, answer_letter_ids, load_mmlu, mmlu_prompt_ids,
                                score_short_answer, summarize, tab_prompt)
from fiveway.em_baseline import find_submodules, load_lesions  # noqa: E402
from fiveway.em_search import stage1_mlp  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.prompts import cut_at_new_question, long_enough  # noqa: E402
from fiveway.scorer import CLASSES, FiveWayScorer, objective_values  # noqa: E402
from fiveway.search import (MAX_PICKS, MIN_GAIN, PATIENCE, Evaluator, Objective, generation_seconds,  # noqa: E402
                            knockout_of, prune, summary)

DECODING = "rep1.25"
TAG = "rep1.25_em_attention"
ATTENTION_KINDS = ("q", "k", "v", "o")
MLP_KINDS = ("gate", "up", "down")
PRESHRINK_ABOVE = 64
PRUNE_BUDGET = 15000    # answers batched_prune() may ask for; above what every set pruned before it existed needed
SPACES = ["1_mlp_lesions", "2_heads", "2b_heads_grown", "3_mlp_lesions_and_heads", "4_attention_lesions",
          "5_mlp_and_attention_lesions"]


def grow(evaluator, objective, components: list, values: list, log) -> dict:
    """em_search's stage 1 for components whose single values are already known: lesion the top 1, 2, 3, ... together.

    Used for heads, whose single-head values search.py's stage 1 measured; the
    attention lesions go through em_search.stage1_mlp itself, which first
    measures each lesion alone.
    """
    # NOTE: a component whose answer did not count has value
    # None, ranked last; the stable sort keeps the layer order among ties.
    order = np.argsort(-np.array([-np.inf if value is None else value for value in values]), kind="stable")
    ranking = [components[i] for i in order]
    trajectory, found = [], None
    batch_size = evaluator.engine.batch_size
    for start in range(0, len(ranking), batch_size):
        sizes = list(range(start + 1, min(start + batch_size, len(ranking)) + 1))
        records = evaluator.records([knockout_of(ranking[:size]) for size in sizes])
        for size, record in zip(sizes, records):
            value = objective.value(record)
            trajectory.append({"size": size, "value": value})
            if found is None and objective.works(value):
                found = (size, record)
        if found is not None:
            break
    if found is not None:
        best_size, best_record = found
    else:
        scored = [step for step in trajectory if step["value"] is not None]
        best_size = max(scored, key=lambda step: (step["value"], -step["size"]))["size"]
        best_record = evaluator.records([knockout_of(ranking[:best_size])])[0]
    log(f"    grown: works {found is not None}, best {best_size} components, value {objective.value(best_record)}")
    return {"works": found is not None, "best_size": best_size, "best": summary(best_record, objective),
            "ranking": [[kind, c if isinstance(c, str) else list(c)] for kind, c in ranking[:max(best_size, 32)]],
            "trajectory": trajectory}


def lesion_sweep(evaluator, objective, n_layers: int, base: list, baseline_value, log) -> dict:
    """search.py's forward sweep (em_search's stage 3), with attention-matrix lesions in place of heads.

    On top of the base lesions, visit the layers from first to last; in each,
    add the attention matrix (q, k, v or o) whose lesion raises the objective
    most, for at most MAX_PICKS picks and until PATIENCE picks in a row gain
    nothing, committing each pick as search.forward_sweep does; stop as soon as
    a knockout works.
    """
    picked, trajectory = [], []
    best_so_far = baseline_value if baseline_value is not None else 0.0
    for layer in range(n_layers):
        stale = 0
        for _ in range(MAX_PICKS):
            options = [f"L{layer}.{kind}" for kind in ATTENTION_KINDS if f"L{layer}.{kind}" not in picked]
            if not options:
                break
            records = evaluator.records([Intervention.knockout(lesions=base + picked + [option]) for option in options])
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
            picked.append(option)
            trajectory.append({"layer": layer, "added": option, "value": value, "gain": gain,
                               "n_attention_lesions": len(picked)})
            if objective.works(value):
                log(f"    works after {option}: value {value:.3f}, {len(base)} MLP + {len(picked)} attention lesions")
                return {"works": True, "found": summary(record, objective), "trajectory": trajectory}
            if stale >= PATIENCE:
                break
        log(f"    layer {layer:2d}: {len(picked)} attention lesions, best value so far {best_so_far:.4f}")
    final = evaluator.records([Intervention.knockout(lesions=base + picked)])[0]
    return {"works": False, "found": summary(final, objective), "trajectory": trajectory}


def components_of(result: dict) -> list:
    return [("head", tuple(c)) for c in result["heads"]] + [("lesion", c) for c in result.get("lesions", [])]


def shrink(evaluator, objective, ranked: list, log) -> tuple:
    """prune()'s rule applied to whole chunks of a large working set, in halving chunk sizes; returns (kept, steps).

    The set (ranked: strongest first) is cut into chunks of `size`
    consecutive components. Every chunk's removal is tried in one batch, and
    the removal that keeps the objective highest is taken while the set still
    works; then the rest is cut again. When no chunk can go, the size halves.
    It stops before chunks of one, where prune() itself takes over.

    NOTE: prune() alone tries every leave-one-out
    set in each round, about n^2 / 2 answers for n components: 215,000 for the
    656 heads of a grown set on Llama-2-13B. Chunks first cost a few hundred
    answers and leave prune() a much smaller set.
    NOTE: the weakest components cannot simply be cut from
    the end of a grown set: it is the shortest prefix of the ranking that
    works, so every shorter prefix already failed. A chunk from anywhere in the
    ranking may go.
    """
    kept, size, steps = list(ranked), len(ranked) // 2, []
    while size >= 2:
        chunks = [set(kept[start:start + size]) for start in range(0, len(kept), size)]
        candidates = [[component for component in kept if component not in chunk] for chunk in chunks]
        values = [objective.value(record) for record in evaluator.records([knockout_of(c) for c in candidates])]
        keeping = [(value, i) for i, value in enumerate(values) if objective.works(value)]
        if keeping:
            value, i = max(keeping)
            steps.append({"chunk_size": size, "removed": len(kept) - len(candidates[i]), "value": value})
            kept = candidates[i]
            size = min(size, len(kept) // 2)
        else:
            size //= 2
    log(f"    shrunk {len(ranked)} -> {len(kept)} components in chunks before pruning")
    return kept, steps


def batched_prune(evaluator, objective, components: list, log) -> tuple:
    """Prune a working set until no component can go alone, removing many per round; returns (kept, steps).

    Each round tries every leave-one-out set in one batch, as prune() does.
    The components whose removal alone keeps the set working are ranked by
    the value left after their removal, highest first. The round removes all
    of them, else the first half, a quarter, ... down to the single best one,
    whichever count first still works. It stops when no component can go
    alone, so the result is minimal in prune()'s sense.

    NOTE: prune() removes one component per round,
    at n answers a round: about 12,000 answers for 160 heads, four GPU hours
    on Llama-2-13B. Removing many per round takes a few rounds of n answers.
    NOTE: two components that can each go alone may not go
    together (one may stand in for the other), so a multiple removal is
    checked before it is kept; a single removal needs no check, since its
    leave-one-out set already worked.
    """
    kept, steps, asked = list(components), [], 0
    while len(kept) > 1:
        # NOTE: a fragile set, one that breaks when any two of
        # its components go together, loses only a few components per round. A
        # round is started only while it fits in PRUNE_BUDGET answers; past it, the
        # set is reported as it stands, working but not minimal.
        if asked + len(kept) > PRUNE_BUDGET:
            log(f"    pruning stopped at the budget of {PRUNE_BUDGET} answers with {len(kept)} components")
            return kept, steps, False
        candidates = [kept[:i] + kept[i + 1:] for i in range(len(kept))]
        asked += len(candidates)
        values = [objective.value(record) for record in evaluator.records([knockout_of(c) for c in candidates])]
        removable = sorted(((value, i) for i, value in enumerate(values) if objective.works(value)), reverse=True)
        if not removable:
            break
        count = len(removable)
        while True:
            dropped = {kept[i] for _, i in removable[:count]}
            candidate = [component for component in kept if component not in dropped]
            if count == 1:
                break
            asked += 1
            if objective.works(objective.value(evaluator.records([knockout_of(candidate)])[0])):
                break
            count //= 2
        kept = candidate
        steps.append({"removed": count, "left": len(kept)})
    log(f"    pruned in batches to {len(kept)} components ({len(steps)} rounds)")
    return kept, steps, True


def prune_found(evaluator, objective, found: dict, ranked: list, log) -> dict:
    """Prune a working grown set (ranked: strongest first): shrink() when larger than PRESHRINK_ABOVE, then batched_prune()."""
    kept, shrink_steps = list(ranked), []
    if len(ranked) > PRESHRINK_ABOVE:
        kept, shrink_steps = shrink(evaluator, objective, ranked, log)
    kept, steps, minimal = batched_prune(evaluator, objective, kept, log)
    return {"circuit": summary(evaluator.records([knockout_of(kept)])[0], objective), "steps": steps,
            "shrink_steps": shrink_steps, "minimal": minimal}


def reported(entry: dict) -> tuple:
    """(works, the set a space reports): the pruned set when it worked, else its best set."""
    if entry["works"]:
        return True, entry["pruned"]["circuit"]
    return False, entry["best"]


def em_search_spaces(em: dict, heads_search: dict, objective: str) -> dict:
    """Spaces 1-3 as em_search.py (and, for space 2, search.py) reported them: {space: (works, set)}."""
    entry = em["objectives"][objective]
    stages, found_at = entry["stages"], entry["found_at_stage"]
    stage1 = stages["1_mlp_lesions"]
    space1 = (True, entry["pruned"]["circuit"]) if found_at == "1_mlp_lesions" else (False, stage1["best"])
    # NOTE: space 2 is search.py's stage 1 (em_search.py's
    # stage 2 reuses it). When it worked, search.py stopped there and pruned
    # that set, so its pruned set is the one to report, as for every space.
    heads_entry = heads_search["objectives"][objective]
    if heads_entry["found_at_stage"] == "1_single_layer_heads":
        space2 = (True, heads_entry["pruned"]["circuit"])
    else:
        space2 = (False, stages["2_single_layer_heads"]["best"])
    if found_at is not None:
        space3 = (True, entry["pruned"]["circuit"])
    else:
        space3 = (False, stages["3_heads_on_mlp_lesion"]["found"])
    return {"1_mlp_lesions": space1, "2_heads": space2, "3_mlp_lesions_and_heads": space3}


def describe(works: bool, result: dict) -> dict:
    lesions = result.get("lesions", [])
    probabilities = result["probabilities"]
    return {"works": works, "value": result["value"],
            "n_mlp_lesions": sum(name.split(".")[1] in MLP_KINDS for name in lesions),
            "n_attention_lesions": sum(name.split(".")[1] in ATTENTION_KINDS for name in lesions),
            "n_heads": len(result["heads"]), "heads": result["heads"], "lesions": lesions,
            "predicted_class": None if probabilities is None else CLASSES[int(np.argmax(probabilities))],
            "probabilities": probabilities, "answer": result.get("answer"), "story": result["story"]}


def check_sets(engine, scorer, task: str, conditions: dict, out_path: Path, run_dir: Path, limit, log) -> dict:
    """The other task's answer, the twenty TAB items and MMLU for every condition (name -> Intervention).

    Writes out_path in benchmarks.py's format (conditions, tab, mmlu, summary)
    after every item; a rerun skips the items already there when the
    conditions are the same. Conditions with the same intervention share one
    generation.
    """
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    keys = {name: intervention.key() for name, intervention in conditions.items()}
    distinct = list(dict.fromkeys(keys.values()))
    by_key = {intervention.key(): intervention for intervention in conditions.values()}
    unique = [by_key[key] for key in distinct]
    results = {"config": {"task": task, "decoding": TAB_DECODING, "connected_text_tokens": CONNECTED_TEXT_TOKENS,
                          "short_answer_tokens": SHORT_ANSWER_TOKENS, "device_name": device_name},
               "conditions": keys, "other_task": {}, "tab": [], "mmlu": [], "summary": {}}
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous["conditions"] == keys:
            results.update(other_task=previous["other_task"], tab=previous["tab"], mmlu=previous["mmlu"])

    def save():
        results["summary"] = summarize(results)
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    # The other task's answer, scored as the searches score answers (its own gate rule).
    if not results["other_task"]:
        other_task = [name for name in TASKS if name != task][0]
        evaluator = Evaluator(engine, scorer, other_task, DECODING, run_dir / f"held_out_from_{task}_cache_{other_task}.jsonl")
        evaluator.gate = evaluator.records([Intervention()])[0]["answer"] is not False
        for key, record in zip(distinct, evaluator.records(unique)):
            results["other_task"][key] = {"task": other_task, "valid": evaluator.valid(record),
                                          "probabilities": record["probabilities"], "story": record["story"]}
        save()
        log(f"  other task ({other_task}): {len(distinct)} answers")

    done_tab = {item["item"] for item in results["tab"]}
    for item_id, item in enumerate(TAB_ITEMS):
        if item_id in done_tab or (limit is not None and item_id % 5 >= limit):
            continue
        prompt = tab_prompt(engine, item)
        is_connected = item["subtest"] == "connected_text"
        full_texts = engine.generate(engine.encode(prompt), unique,
                                     CONNECTED_TEXT_TOKENS if is_connected else SHORT_ANSWER_TOKENS,
                                     TAB_DECODING["connected_text" if is_connected else "short_answer"])
        responses = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
        records = {}
        if is_connected:
            scored = [index for index, response in enumerate(responses) if long_enough(response)]
            probabilities = scorer.probabilities([responses[index] for index in scored]) if scored else None
            for index, (key, full_text, response) in enumerate(zip(distinct, full_texts, responses)):
                record = {"response": response, "full_text": full_text, "probabilities": None, "objectives": None}
                if index in scored:
                    row = probabilities[scored.index(index)]
                    record["probabilities"] = row.round(6).tolist()
                    record["objectives"] = {name: round(float(values[0]), 6)
                                            for name, values in objective_values(row[None]).items()}
                records[key] = record
        else:
            for key, full_text, response in zip(distinct, full_texts, responses):
                records[key] = {"response": response, "full_text": full_text, **score_short_answer(item, response)}
        results["tab"].append({"item": item_id, **item, "rendered_prompt": prompt, "device_name": device_name,
                               "responses": {name: records[key] for name, key in keys.items()}})
        save()
    log(f"  TAB: {len(results['tab'])} items")

    # NOTE: as in benchmarks.py: no key-value cache for a
    # one-pass readout (it would hold tens of gigabytes for a long 5-shot
    # prompt); generation above needed it and runs before this.
    engine.model.config.use_cache = False
    letter_ids = answer_letter_ids(engine.tokenizer)
    done_mmlu = {(item["subject"], item["item"]) for item in results["mmlu"]}
    for subject in MMLU_SUBJECTS:
        questions, shots = load_mmlu(subject)
        for question_id, row in enumerate(questions if limit is None else questions[:limit]):
            if (subject, question_id) in done_mmlu:
                continue
            prompt_ids, n_shots = mmlu_prompt_ids(engine, subject, shots, row)
            # NOTE: logits: (n_distinct, vocab); log_softmax over the
            # vocabulary, then the four letter columns: (n_distinct, 4).
            logits = engine.next_token_logits(prompt_ids, unique, pad_batches=False)
            letter_log_probabilities = dict(zip(distinct, torch.log_softmax(logits, dim=-1)[:, letter_ids].tolist()))
            answer = int(row["answer"])
            entry = {"subject": subject, "item": question_id, "answer": answer, "n_shots": n_shots,
                     "prompt_tokens": len(prompt_ids), "device_name": device_name, "conditions": {}}
            for name, key in keys.items():
                values = letter_log_probabilities[key]
                prediction = int(np.argmax(values))
                entry["conditions"][name] = {"prediction": prediction, "correct": prediction == answer,
                                             "log_probabilities": [round(value, 5) for value in values]}
            results["mmlu"].append(entry)
            if (question_id + 1) % 25 == 0:
                save()
        save()
        log(f"  MMLU {subject}: done")
    engine.model.config.use_cache = True
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
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
    submodules = find_submodules(engine.model)
    attention_names = [s.name for s in submodules if s.kind in ATTENTION_KINDS]
    mlp_names = [s.name for s in submodules if s.kind in MLP_KINDS]
    load_lesions(engine, mlp_names + attention_names)
    geometry = engine.geometry
    run_dir = RUNS / "fiveway" / args.model / (TAG + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, DECODING, run_dir / f"cache_{args.task}.jsonl")
    # NOTE: a smoke run keeps its outputs under runs/, so
    # results/ (in git) only ever holds full runs.
    folder = run_dir if args.smoke else RESULTS / "em_attention"
    out_path = folder / f"{args.model}_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "config": {**vars(args), "preshrink_above": PRESHRINK_ABOVE}, "geometry": geometry,
        "attention_lesions": attention_names, "objectives": {}}
    log(f"{args.model} on {device_name}, task {args.task}: {len(attention_names)} attention and {len(mlp_names)} "
        f"MLP matrices lesionable; cache holds {len(evaluator.cache)} answers")

    em = json.loads((RESULTS / "em_search" / f"{args.model}_{args.task}.json").read_text())
    heads_search = json.loads((RESULTS / "search" / f"{args.model}_{args.task}_rep1.25_classifier.json").read_text())
    baseline = evaluator.records([Intervention()])[0]
    evaluator.gate = baseline["answer"] is not False
    assert evaluator.gate == em["gate"], (evaluator.gate, em["gate"])
    results["gate"] = evaluator.gate
    results["baseline"] = {"answer": baseline["answer"], "story": baseline["story"],
                           "probabilities": baseline["probabilities"],
                           "same_as_em_search": baseline["story"] == em["baseline"]["story"]}
    log(f"unablated answer the same as em_search's: {results['baseline']['same_as_em_search']}")

    def save():
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    head_components = [("head", (layer, head)) for layer in range(geometry["n_layers"]) for head in range(geometry["n_heads"])]
    for name in args.objectives:
        if results["objectives"].get(name, {}).get("done"):
            log(f"objective {name}: already done")
            continue
        objective = Objective(name, "classifier", evaluator)
        entry = {}

        log(f"objective {name}, 4_attention_lesions:")
        evaluator.used = set()
        space4 = stage1_mlp(evaluator, objective, attention_names, log)
        if space4["works"]:
            # Strongest first, by single-lesion value (a lesion whose answer did not count last).
            single = space4["single_lesion_values"]
            ranked = sorted([("lesion", lesion) for lesion in space4["best"]["lesions"]],
                            key=lambda c: np.inf if single[c[1]] is None else -single[c[1]])
            space4["pruned"] = prune_found(evaluator, objective, space4["best"], ranked, log)
        space4["seconds"] = generation_seconds(evaluator, evaluator.used)
        entry["4_attention_lesions"] = space4

        log(f"objective {name}, 2b_heads_grown:")
        evaluator.used = set()
        values = [value for row in heads_search["objectives"][name]["stages"]["1_single_layer_heads"]["single_head_values"]
                  for value in row]
        space2b = grow(evaluator, objective, head_components, values, log)
        if space2b["works"]:
            ranked = [("head", tuple(c)) for _, c in space2b["ranking"][:space2b["best_size"]]]
            space2b["pruned"] = prune_found(evaluator, objective, space2b["best"], ranked, log)
        space2b["seconds"] = generation_seconds(evaluator, evaluator.used)
        entry["2b_heads_grown"] = space2b

        log(f"objective {name}, 5_mlp_and_attention_lesions:")
        evaluator.used = set()
        em_entry = em["objectives"][name]
        stage1 = em_entry["stages"]["1_mlp_lesions"]
        if em_entry["found_at_stage"] == "1_mlp_lesions":
            # NOTE: as for space 3: when MLP lesions alone work,
            # the search stops at stage 1, and space 5 reports stage 1's set.
            space5 = {"from_em_stage_1": True, "works": True, "pruned": em_entry["pruned"]}
        else:
            base = stage1["best"]["lesions"]
            sweep = lesion_sweep(evaluator, objective, geometry["n_layers"], base, stage1["best"]["value"], log)
            space5 = {"from_em_stage_1": False, "base_lesions": base, **sweep}
            if sweep["works"]:
                # NOTE: pruned exactly as em_search.py pruned
                # space 3, one component per round, so the two compare alike.
                space5["pruned"] = prune(evaluator, objective, sweep["found"], log)
            else:
                space5["best"] = sweep["found"]
        space5["seconds"] = generation_seconds(evaluator, evaluator.used)
        entry["5_mlp_and_attention_lesions"] = space5
        evaluator.used = None

        entry["done"] = True
        results["objectives"][name] = entry
        save()
        log(f"objective {name} done: attention lesions {space4['works']}, heads grown {space2b['works']}, "
            f"MLP + attention lesions {space5['works']}")

    # Every reported set of the six spaces, checked alike.
    reported_sets = {}
    for name in args.objectives:
        spaces = em_search_spaces(em, heads_search, name)
        own = results["objectives"][name]
        spaces["2b_heads_grown"] = reported(own["2b_heads_grown"])
        spaces["4_attention_lesions"] = reported(own["4_attention_lesions"])
        space5 = own["5_mlp_and_attention_lesions"]
        spaces["5_mlp_and_attention_lesions"] = (True, space5["pruned"]["circuit"]) if space5["works"] else (False, space5["best"])
        reported_sets[name] = {space: describe(works, result) for space, (works, result) in spaces.items()}
        # NOTE: when MLP lesions alone work, the search stops
        # at stage 1 and spaces 3 and 5 both report stage 1's set: such a pair
        # agrees by construction, and comparisons of 5 against 3 must leave it out.
        reused = em["objectives"][name]["found_at_stage"] == "1_mlp_lesions"
        for space in ("3_mlp_lesions_and_heads", "5_mlp_and_attention_lesions"):
            reported_sets[name][space]["from_em_stage_1"] = reused
        # A working grown set whose pruning stopped at PRUNE_BUDGET is not minimal; every other pruned set is.
        for space in ("2b_heads_grown", "4_attention_lesions"):
            row = reported_sets[name][space]
            row["minimal"] = own[space]["pruned"].get("minimal", True) if row["works"] else None
    results["reported"] = reported_sets
    for name, spaces in reported_sets.items():
        for space, row in spaces.items():
            row["key"] = knockout_of(components_of(row)).key()
    save()

    conditions = {"clean": Intervention()}
    for name, spaces in reported_sets.items():
        for space in SPACES:
            row = spaces[space]
            conditions[f"{space}_{name}"] = knockout_of(components_of(row))
    benchmarks_folder = folder / "benchmarks"
    benchmarks_folder.mkdir(parents=True, exist_ok=True)
    check_start = time.time()
    check_sets(engine, scorer, args.task, conditions, benchmarks_folder / f"{args.model}_{args.task}.json", run_dir,
               2 if args.smoke else None, log)
    results["check_seconds"] = round(time.time() - check_start, 1)
    results["device"] = device_name
    save()
    log(f"finished; {evaluator.n_generated} new answers generated; wrote {out_path}")


if __name__ == "__main__":
    main()
