"""What the learned-mask and cross-entropy searches knock out, and how their knockout sets are checked on held-out prompts.

mask_search.py (learned masks) and cem_search.py (the cross-entropy method)
search three spaces of components, the three stages of the comparison:

    1_mlp_lesions             MLP weight lesions only: each MLP weight matrix (up and
                              down; gate too in Llama) with its top 1% of weights by
                              |w dL/dw| on Control narratives zeroed (em_baseline.py)
    2_heads                   attention heads only
    3_mlp_lesions_and_heads   both

A component is ("lesion", "L3.up") or ("head", (layer, head)), as in
search.py's prune(). held_out() scores a knockout set's answers to prompts
no search saw: the other task's story and the five TAB connected-text prompts.
"""

from __future__ import annotations

import json

import numpy as np

from fiveway.benchmarks import CONNECTED_TEXT_TOKENS, TAB_ITEMS, tab_prompt
from fiveway.engine import Intervention
from fiveway.em_baseline import find_submodules
from fiveway.models import TASKS
from fiveway.paths import RESULTS
from fiveway.prompts import cut_at_new_question, long_enough
from fiveway.scorer import objective_values
from fiveway.search import Evaluator

DECODING = "rep1.25"
MLP_KINDS = ("gate", "up", "down")
SPACES = {"1_mlp_lesions": ("lesion",), "2_heads": ("head",), "3_mlp_lesions_and_heads": ("lesion", "head")}


def mlp_matrix_names(engine) -> list:
    """The lesionable MLP weight matrices, layer by layer: "L0.up", "L0.down", ..."""
    return [submodule.name for submodule in find_submodules(engine.model) if submodule.kind in MLP_KINDS]


def space_components(engine, space: str) -> list:
    """The components of a space, lesions first, each in layer order."""
    components = []
    if "lesion" in SPACES[space]:
        components += [("lesion", name) for name in mlp_matrix_names(engine)]
    if "head" in SPACES[space]:
        n_layers, n_heads = engine.geometry["n_layers"], engine.geometry["n_heads"]
        components += [("head", (layer, head)) for layer in range(n_layers) for head in range(n_heads)]
    return components


def single_values(model: str, task: str, objective: str, components: list) -> np.ndarray:
    """Each component's objective value when knocked out alone, from the earlier searches' results files.

    NOTE: search.py's stage 1 knocked out every head alone
    and em_search.py's stage 1 every MLP matrix alone, both with the same
    decoding, gate and classifier, so these values cost no new generation.
    An answer that did not count is -inf, ranked last.
    """
    heads = json.loads((RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json").read_text())
    head_values = heads["objectives"][objective]["stages"]["1_single_layer_heads"]["single_head_values"]
    lesions = json.loads((RESULTS / "em_search" / f"{model}_{task}.json").read_text())
    lesion_values = lesions["objectives"][objective]["stages"]["1_mlp_lesions"]["single_lesion_values"]
    values = [head_values[c[0]][c[1]] if kind == "head" else lesion_values[c] for kind, c in components]
    return np.array([-np.inf if value is None else value for value in values])


def same_baseline(model: str, task: str, story: str) -> bool:
    """Whether the unablated answer is the one search.py and em_search.py scored (their single knockouts start cem_search)."""
    stored = [json.loads((RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json").read_text())["baseline"],
              json.loads((RESULTS / "em_search" / f"{model}_{task}.json").read_text())["baseline"]]
    return all(baseline["story"] == story for baseline in stored)


def held_out_scores(engine, scorer, task: str, run_dir, previous: dict, reported: dict, log) -> dict:
    """Held-out scores keyed by intervention key: the stored ones, plus every reported set not scored yet.

    ``reported``: {key: Intervention}. The unablated model is always included,
    under the key "clean". Keying by the knockout itself means a redone
    experiment's new set is scored afresh, and an old set's score never
    stands in for it.
    """
    scores = dict(previous)
    wanted = {"clean": Intervention(), **reported}
    missing = [key for key in wanted if key not in scores]
    if missing:
        scores.update(zip(missing, held_out(engine, scorer, task, run_dir, [wanted[key] for key in missing], log)))
    return scores


def held_out(engine, scorer, task: str, run_dir, interventions: list, log) -> list:
    """For each knockout: its answer to the other task's prompt and its mean scores on the TAB connected-text prompts.

    Returns per intervention {"other_task": {"probabilities", "valid", "story"},
    "tab": {"n_scored", "mean_probabilities", "mean_objectives"}}. The
    other task's answers are cached in run_dir (their own file, since another
    job searches that task); the TAB answers are few and made again on a rerun.
    """
    other_task = [name for name in TASKS if name != task][0]
    evaluator = Evaluator(engine, scorer, other_task, DECODING, run_dir / f"held_out_from_{task}_cache_{other_task}.jsonl")
    baseline = evaluator.records([Intervention()])[0]
    # NOTE: as in search.py, the other task's gate applies only
    # when the unablated model passes it.
    evaluator.gate = baseline["answer"] is not False
    other_records = evaluator.records(interventions)

    tab_probabilities = [[] for _ in interventions]
    for item in [item for item in TAB_ITEMS if item["subtest"] == "connected_text"]:
        full_texts = engine.generate(engine.encode(tab_prompt(engine, item)), interventions, CONNECTED_TEXT_TOKENS,
                                     DECODING)
        responses = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
        scored = [index for index, response in enumerate(responses) if long_enough(response)]
        if scored:
            probabilities = scorer.probabilities([responses[index] for index in scored])
            for row, index in enumerate(scored):
                tab_probabilities[index].append(probabilities[row])
        log(f"    held out: TAB {item['prompt']!r} done")

    results = []
    for record, probabilities in zip(other_records, tab_probabilities):
        # NOTE: probabilities: a list of (5,) rows, one per scored TAB
        # answer; stacked to (n_scored, 5) and averaged over the answers.
        mean = np.mean(probabilities, axis=0) if probabilities else None
        results.append({
            "other_task": {"task": other_task, "valid": evaluator.valid(record),
                           "probabilities": record["probabilities"], "story": record["story"]},
            "tab": {"n_scored": len(probabilities),
                    "mean_probabilities": None if mean is None else mean.round(6).tolist(),
                    "mean_objectives": None if mean is None else
                    {name: round(float(value[0]), 6) for name, value in objective_values(mean[None]).items()}}})
    return results
