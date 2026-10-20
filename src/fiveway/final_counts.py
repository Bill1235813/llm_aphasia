"""Every lesion method's successes counted again with a final-answer rule that also accepts the classifier's top class.

The searches stop when the target objective passes 0.5 (A = P(aphasia), B = P(Broca), C = P(Wernicke)), and
their reports count a setting as found by that same rule. A final answer can, however, have its target as the
classifier's most probable class without passing 0.5: P(Wernicke) = 0.42 with every other class lower. This
script keeps every search as it ran and judges each method's final answer in each setting by two rules:

    threshold    the answer counts and the objective is above 0.5 (the rule of the reports and the paper)
    top class    the answer counts and either the objective is above 0.5 or the target is the most probable of
                 the five classes; for A, any of the three aphasia classes (Conduction/Anomic, Broca, Wernicke)

An answer counts when it is long enough and passes the Cinderella gate (search.Evaluator.valid); the stored
records give an answer that does not count no objective value. A method's final answer in a setting is its
pruned set's answer when it worked, else the set it reports when it did not (each search's best set).

Reads the results of every search on the seven older models (42 settings: 2 tasks x 3 objectives) and the
grown-heads sets of the four Qwen3.5 models, checks that the threshold rule reproduces every stored "works"
flag, and prints the counts by target and model group under both rules, with every setting that the top-class
rule adds.

Usage (CPU, seconds):
    python src/fiveway/final_counts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import CLASSES, OBJECTIVES, SUCCESS_THRESHOLD, works_finally  # noqa: E402

OLDER = ["gpt2-xl", "llama2-7b-chat", "llama2-13b-chat", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b"]
QWEN = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
GROUP = {"gpt2-xl": "GPT-2 XL", "llama2-7b-chat": "Llama-2-chat", "llama2-13b-chat": "Llama-2-chat",
         **{model: "Pythia" for model in OLDER if model.startswith("pythia")}, **{model: "Qwen3.5" for model in QWEN}}


def judge(objective: str, value, probabilities) -> tuple:
    """(threshold rule, top-class rule) for one final answer; value is None when the answer does not count."""
    if value is None:
        return False, False
    return value > SUCCESS_THRESHOLD, works_finally(objective, value, probabilities)


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# --- each method's final answer in one setting: (stored works flag, objective value, probabilities, size) -----------

def em_attention_space(space: str):
    def final(model, task, objective):
        row = load(RESULTS / "em_attention" / f"{model}_{task}.json")["reported"][objective][space]
        size = len(row.get("heads", [])) + len(row.get("lesions", []))
        return row["works"], row["value"], row["probabilities"], size
    return final


def grown_heads(model, task, objective):
    """Head-grown: grown_heads.py's classifier sets for Qwen3.5, em_attention.py's space 2b for the older models."""
    if model in QWEN:
        row = load(RESULTS / "grown_heads" / f"{model}_{task}_classifier.json")["reported"][objective]
        return row["works"], row["value"], row["probabilities"], len(row["heads"])
    return em_attention_space("2b_heads_grown")(model, task, objective)


def grown_cosine(model, task, objective):
    """Head-grown-cos: the cos-sim set's answer, scored by the classifier (its own stopping rule was cos-sim)."""
    row = load(RESULTS / "grown_heads" / f"{model}_{task}_cosine.json")["reported"][objective]
    value = row["classifier_value"]
    return value is not None and value > SUCCESS_THRESHOLD, value, row["probabilities"], len(row["heads"])


def mask(space: str, variant: str):
    def final(model, task, objective):
        part = load(RESULTS / "mask_search" / f"{model}_{task}.json")["objectives"][objective][space][variant]
        record = part.get("pruned", {}).get("circuit", part["result"])
        size = len(record.get("heads", [])) + len(record.get("lesions", [])) + len(record.get("mlp_blocks", []))
        return part["works"], record["value"], record["probabilities"], size
    return final


def cem(space: str):
    def final(model, task, objective):
        entry = load(RESULTS / "cem_search" / f"{model}_{task}.json")["objectives"][objective][space]
        record = entry["pruned"]["circuit"] if entry["works"] else entry["best"]
        size = len(record.get("heads", [])) + len(record.get("lesions", [])) + len(record.get("mlp_blocks", []))
        return entry["works"], record["value"], record["probabilities"], size
    return final


def layer_search(model, task, objective):
    """Layer search, all stages (search.py): the pruned set when a stage worked, else the best-valued stage's set."""
    entry = load(RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json")["objectives"][objective]
    if entry["found_at_stage"] is not None:
        record = entry["pruned"]["circuit"]
    else:
        stages = entry["stages"]
        candidates = [stages["1_single_layer_heads"]["best"]] + [stages[name]["found"] for name in
                                                                  ("2_forward_heads", "3_forward_heads_mlp")
                                                                  if stages.get(name, {}).get("found")]
        # NOTE: a set whose answer does not count has no value and ranks last.
        record = max(candidates, key=lambda r: -1 if r["value"] is None else r["value"])
    size = len(record["heads"]) + len(record["mlp_blocks"])
    return entry["found_at_stage"] is not None, record["value"], record["probabilities"], size


def param(model, task, objective):
    """Param (em_baseline.py): A works when any of the four subtype clusters reaches it, B and C by their own cluster."""
    document = load(RESULTS / "em_baseline" / model / "clusters.json")
    if document is None:
        return None
    clusters = document["tasks"][task]["clusters"]
    names = {"A": ["Anomic", "Broca", "Conduction", "Wernicke"], "B": ["Broca"], "C": ["Wernicke"]}[objective]
    finals = []
    for name in names:
        # NOTE: a subtype that no single lesion was assigned to has no cluster (no steps).
        if not clusters[name]["steps"]:
            continue
        step = clusters[name]["steps"][-1]      # the cluster's final answer
        value = step["objectives"][objective] if step["valid"] else None
        # NOTE: a cluster whose final answer was too short to score stores works as None.
        stored_works = document["tasks"][task]["comparison"]["classifier"]["clusters"][name]["works"]
        stored = bool(stored_works and stored_works[objective])
        finals.append((stored, value, step["probabilities"], step["size"]))
    # NOTE: the setting works when any of its clusters works, so its "final answer" under either
    # rule is the best of them; report the first cluster that the top-class rule accepts, else the first one.
    if not finals:
        return False, None, None, 0
    accepted = [final for final in finals if judge(objective, final[1], final[2])[1]]
    chosen = accepted[0] if accepted else finals[0]
    return any(final[0] for final in finals), *chosen[1:]


def param_grown(model, task, objective):
    """param_grown.py (kept for later, not in the paper): Param's MLP and attention lesions in one ranking, grown
    and pruned as Head-grown grows heads."""
    document = load(RESULTS / "param_grown" / f"{model}_{task}.json")
    if document is None or objective not in document.get("reported", {}):
        return None
    row = document["reported"][objective]
    return row["works"], row["value"], row["probabilities"], len(row["lesions"])


METHODS = [  # (section, name in the paper, final answer of one setting, models)
    ("Heads", "Head-mask", mask("2_heads", "soft"), OLDER),
    ("Heads", "Head-mask-hard", mask("2_heads", "binary"), OLDER),
    ("Heads", "Head-grown-cos", grown_cosine, OLDER),
    ("Heads", "Layer search, first stage", em_attention_space("2_heads"), OLDER),
    ("Heads", "Head-CEM", cem("2_heads"), OLDER),
    ("Heads", "Head-grown", grown_heads, OLDER),
    ("Heads and MLP blocks", "Layer search, all stages", layer_search, OLDER),
    ("MLP lesions", "Param-mask", mask("1_mlp_lesions", "soft"), OLDER),
    ("MLP lesions", "Param-mask-hard", mask("1_mlp_lesions", "binary"), OLDER),
    ("MLP lesions", "Cross-entropy method", cem("1_mlp_lesions"), OLDER),
    ("MLP lesions", "Grown", em_attention_space("1_mlp_lesions"), OLDER),
    ("Attention lesions", "Grown", em_attention_space("4_attention_lesions"), OLDER),
    ("MLP and attention lesions", "Param", param, OLDER),
    # NOTE: the paper's Param-C is "Grown, then attention sweep": Param's lesions searched with
    # the classifier. "Grown together" is the later run that ranks both kinds of lesion as one (param_grown.py).
    ("MLP and attention lesions", "Grown, then attention sweep", em_attention_space("5_mlp_and_attention_lesions"), OLDER),
    ("MLP and attention lesions", "Grown together", param_grown, OLDER),
    ("MLP lesions and heads", "Learned mask, soft", mask("3_mlp_lesions_and_heads", "soft"), OLDER),
    ("MLP lesions and heads", "Learned mask, hard", mask("3_mlp_lesions_and_heads", "binary"), OLDER),
    ("MLP lesions and heads", "Cross-entropy method", cem("3_mlp_lesions_and_heads"), OLDER),
    ("MLP lesions and heads", "Grown, then head sweep", em_attention_space("3_mlp_lesions_and_heads"), OLDER),
    ("All 11 models", "Head-grown", grown_heads, OLDER + QWEN),
]


def main():
    groups = ["GPT-2 XL", "Llama-2-chat", "Pythia", "Qwen3.5"]
    print("Counts as threshold rule -> top-class rule (unchanged counts shown once).\n")
    print(f"{'section':<26}{'method':<30}{'n':>4}{'Aph.':>9}{'Broca':>9}{'Wern.':>9}{'All':>9}"
          + "".join(f"{group:>14}" for group in groups))
    added = []
    for section, name, final, models in METHODS:
        counts = {}   # key -> [threshold, top class]
        n = 0
        for model in models:
            for task in TASKS:
                for objective in OBJECTIVES:
                    result = final(model, task, objective)
                    if result is None:
                        continue
                    stored, value, probabilities, size = result
                    threshold, top_class = judge(objective, value, probabilities)
                    assert threshold == stored, f"{name} {model} {task} {objective}: stored works {stored}, threshold {threshold}"
                    n += 1
                    for key in (objective, "All", GROUP[model]):
                        counts.setdefault(key, [0, 0])
                        counts[key][0] += threshold
                        counts[key][1] += top_class
                    if top_class and not threshold:
                        # NOTE: argsort over the five classes; [-1] is the top class, [-2] the runner-up.
                        order = np.argsort(probabilities)
                        added.append(f"{section}, {name}: {model} {task} {objective}: objective {value:.3f}, top class "
                                     f"{CLASSES[order[-1]]} {probabilities[order[-1]]:.3f}, runner-up "
                                     f"{CLASSES[order[-2]]} {probabilities[order[-2]]:.3f}, {size} components")

        def cell(key):
            if key not in counts:
                return "–"
            before, after = counts[key]
            return str(before) if before == after else f"{before}->{after}"

        print(f"{section:<26}{name:<30}{n:>4}" + "".join(f"{cell(key):>9}" for key in ("A", "B", "C", "All"))
              + "".join(f"{cell(group):>14}" for group in groups))
    print(f"\nSettings the top-class rule adds ({len(added)}):")
    print("\n".join(f"  {line}" for line in added))


if __name__ == "__main__":
    main()
