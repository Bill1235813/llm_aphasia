"""The learned-mask and cross-entropy searches on one page: what works, in which space, on held-out prompts, and how long it took.

Reads results/fiveway/mask_search/*.json (mask_search.py) and
results/fiveway/cem_search/*.json (cem_search.py) and writes
results/fiveway/report/learned_searches.md. Three ways of finding a knockout
set are compared, in each of the three spaces (knockout_spaces.py: 1 MLP
weight lesions, 2 attention heads, 3 both):

    mask, soft     the learned amounts themselves
    mask, binary   the learned mask made yes/no (amount > 0.5), then pruned
    cross-entropy  the cross-entropy method's first working set, then pruned

A set works when the classifier's objective on the task answer is above 0.5.
Held out: the classifier's mean objective on the five TAB connected-text
answers, and its objective on the other task's answer, with the set knocked out.

Usage:
    python src/fiveway/learned_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.knockout_spaces import SPACES  # noqa: E402
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import OBJECTIVES, SUCCESS_THRESHOLD  # noqa: E402

NAMES = {"gpt2-xl": "GPT-2 XL", "llama2-7b-chat": "Llama-2-7B-chat", "llama2-13b-chat": "Llama-2-13B-chat",
         "pythia-1b": "Pythia-1B", "pythia-1.4b": "Pythia-1.4B", "pythia-2.8b": "Pythia-2.8B",
         "pythia-6.9b": "Pythia-6.9B"}
SPACE_NAMES = {"1_mlp_lesions": "1 MLP lesions", "2_heads": "2 heads", "3_mlp_lesions_and_heads": "3 MLP lesions + heads"}
METHODS = [("mask", "soft"), ("mask", "binary"), ("cem", None)]
METHOD_NAMES = {("mask", "soft"): "learned mask, soft", ("mask", "binary"): "learned mask, yes/no",
                ("cem", None): "cross-entropy"}
COLUMNS = [(task, objective) for task in TASKS for objective in OBJECTIVES]


def load(folder: str, model: str, task: str):
    path = RESULTS / folder / f"{model}_{task}.json"
    return json.loads(path.read_text()) if path.exists() else None


def objective_value(probabilities, objective: str):
    return None if probabilities is None else sum(probabilities[column] for column in OBJECTIVES[objective])


def number(value, digits=2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def size_of(result: dict) -> str:
    """'7m+2h': lesioned matrices and heads of a binary set."""
    lesions, heads = len(result.get("lesions", [])), len(result["heads"])
    return "+".join(([f"{lesions}m"] if lesions else []) + ([f"{heads}h"] if heads else [])) or "0"


def run_outcome(document: dict, method: tuple, objective: str, space: str) -> dict | None:
    """Whether a run works, its reported set's size, its value, and its held-out scores."""
    entry = document["objectives"].get(objective, {}).get(space)
    if entry is None or not entry.get("done"):
        return None
    kind, variant = method
    if kind == "mask":
        part = entry[variant]
        reported = part.get("pruned", {}).get("circuit", part["result"])
        works, value, key = part["works"], reported["value"], part.get("reported_key")
        if variant == "soft":
            amounts = [a for a in part["amounts"] if round(a, 3) > 0]
            size = f"c{part['checkpoint']}: {len(amounts)} (Σk {sum(amounts):.1f})"
        else:
            size = f"c{part['checkpoint']}: {size_of(part['result'])}" + (
                f" → {size_of(reported)}" if "pruned" in part else "")
        inherited = False
    else:
        works, key = entry["works"], entry.get("reported_key")
        found = entry["pruned"]["circuit"] if works else entry.get("best")
        value = None if found is None else found["value"]
        size = (f"r{entry['found_at_iteration']}: {size_of(entry['found'])} → {size_of(found)}" if works else
                f"{entry['n_distinct_sets']} sets")
        # NOTE: where a single component already works alone, the
        # search starts with it among its favourites and inherits the success.
        inherited = entry["n_single_working"] > 0
    held = document.get("held_out", {}).get(key)
    tab = None if held is None or held["tab"]["mean_objectives"] is None else held["tab"]["mean_objectives"][objective]
    other = None
    if held is not None and held["other_task"]["valid"]:
        other = objective_value(held["other_task"]["probabilities"], objective)
    return {"works": works, "value": value, "size": size, "minutes": sum(entry["seconds"].values()) / 60,
            "tab": tab, "other": other, "inherited": inherited, "device": entry.get("device")}


def greedy_outcomes() -> dict:
    """The earlier greedy searches in the same three spaces: {(space, model, task, objective): (works, held-out TAB)}.

    heads only: search.py found the set at stage 1 or 2 (stage 3 adds 256-neuron
    MLP blocks, not lesions); MLP lesions only: em_search.py's stage 1; both:
    em_search.py at stage 1 or 3 (heads swept on top of the best stage-1 lesion).
    Held out: benchmarks.py's mean objective on the TAB connected-text answers.
    """
    outcomes = {}
    for model in NAMES:
        for task in TASKS:
            heads = json.loads((RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json").read_text())
            lesions = json.loads((RESULTS / "em_search" / f"{model}_{task}.json").read_text())
            for objective in OBJECTIVES:
                found = {"2_heads": heads["objectives"][objective]["found_at_stage"] in ("1_single_layer_heads",
                                                                                          "2_forward_heads"),
                         "1_mlp_lesions": lesions["objectives"][objective]["found_at_stage"] == "1_mlp_lesions",
                         "3_mlp_lesions_and_heads": lesions["objectives"][objective]["found_at_stage"] is not None}
                for space, works in found.items():
                    folder = RESULTS if space == "2_heads" else RESULTS / "em_search"
                    path = folder / "benchmarks" / f"{model}_{task}_{objective}.json"
                    tab = None
                    if works and path.exists():
                        connected = json.loads(path.read_text())["summary"]["tab"]["corrupt"]["connected_text"]
                        tab = (connected.get("mean_objectives") or {}).get(objective)
                    outcomes[(space, model, task, objective)] = (works, tab)
    return outcomes


def union_section() -> list:
    """Section: the union of each method's MLP-lesion set and head set, knocked out together without a search (union_sets.py)."""
    variants = {"soft": "learned mask, soft", "yesno": "learned mask, yes/no", "cem": "cross-entropy"}
    rows = {variant: [] for variant in variants}
    for model in NAMES:
        path = RESULTS / "union_sets" / f"{model}.json"
        if not path.exists():
            continue
        for task, part in json.loads(path.read_text())["tasks"].items():
            for objective, by_variant in part["objectives"].items():
                for variant, row in by_variant.items():
                    held = part["held_out"].get(row["key"], {})
                    tab = (held.get("tab", {}).get("mean_objectives") or {}).get(objective)
                    rows[variant].append((model, task, objective, row, tab))
    finished = sorted({model for entries in rows.values() for model, *_ in entries}, key=list(MODELS).index)
    lines = [f"## {len(METHODS) + 3}. MLP lesions plus heads without a search (`union_sets.py`)", "",
             "For each method, the set it found with MLP lesions only and the set it found with heads only, knocked "
             "out together exactly as found: no component added, dropped or re-weighted, and no pruning. Models so far: "
             + (", ".join(NAMES[model] for model in finished) or "none") + ".", "",
             "| method | union works | both parts work | union works when both parts work | union works when "
             "neither part works | joint search works | union held out above 0.5 |",
             "|---|---|---|---|---|---|---|"]
    for variant, name in variants.items():
        entries = rows[variant]
        works = [entry for entry in entries if entry[3]["union"]["works"]]
        both = [entry for entry in entries if entry[3]["mlp_lesions_alone"]["works"] and entry[3]["heads_alone"]["works"]]
        neither = [entry for entry in entries
                   if not entry[3]["mlp_lesions_alone"]["works"] and not entry[3]["heads_alone"]["works"]]
        joint = [entry for entry in entries if entry[3]["joint_search"]["works"]]
        held = [entry for entry in works if (entry[4] or 0) > SUCCESS_THRESHOLD]
        lines.append(f"| {name} | {len(works)} / {len(entries)} | {len(both)} | "
                     f"{sum(entry[3]['union']['works'] for entry in both)} / {len(both)} | "
                     f"{sum(entry[3]['union']['works'] for entry in neither)} / {len(neither)} | {len(joint)} | {len(held)} |")
    lines += ["", "Every union: objective value of the union (bold when it works) and its predicted class, against the "
              "MLP-lesion set alone, the head set alone and the joint search of both spaces.", "",
              "| model | task | objective | method | union | MLP lesions alone | heads alone | joint search |",
              "|---|---|---|---|---|---|---|---|"]
    for variant, name in variants.items():
        for model, task, objective, row, _ in rows[variant]:
            union = row["union"]
            value = number(union["value"])
            lines.append(f"| {NAMES[model]} | {task} | {objective} | {name} | "
                         f"{'**' + value + '**' if union['works'] else value} ({union['predicted_class'] or 'unscored'}) | "
                         f"{number(row['mlp_lesions_alone']['value'])} | {number(row['heads_alone']['value'])} | "
                         f"{number(row['joint_search']['value'])} |")
    return lines + [""]


def cell(outcome: dict | None) -> str:
    if outcome is None:
        return "…"
    if not outcome["works"]:
        return f"– ({number(outcome['value'])})"
    held = f"TAB {number(outcome['tab'])}, other {number(outcome['other'])}"
    return f"**{number(outcome['value'])}**{'†' if outcome['inherited'] else ''} {outcome['size']}; {held}"


def main():
    documents = {(kind, model, task): load(f"{kind}_search", model, task)
                 for kind in ("mask", "cem") for model in NAMES for task in TASKS}
    outcomes = {}
    for (kind, model, task), document in documents.items():
        if document is None:
            continue
        for method in METHODS:
            if method[0] != kind:
                continue
            for objective in OBJECTIVES:
                for space in SPACES:
                    outcomes[(method, model, task, objective, space)] = run_outcome(document, method, objective,
                                                                                     space)

    lines = ["# Learned masks and the cross-entropy method: which spaces work", "",
             "Generated by `src/fiveway/learned_summary.py` from `results/fiveway/mask_search/` and "
             "`results/fiveway/cem_search/`.", "",
             "A set **works** when the five-way classifier's objective on the task answer is above 0.5 "
             "(A: P(aphasia), B: P(Broca), C: P(Wernicke); rep1.25 decoding, the Cinderella gate when the "
             "unablated model passes it). Spaces: 1 MLP weight lesions (top 1% of each MLP matrix), 2 attention "
             "heads, 3 both. Held out: the set's objective on the five TAB connected-text answers (mean) and on "
             "the other task's answer.", "",
             "## 1. How many of the model, task and objective combinations work", "",
             "| method | " + " | ".join(SPACE_NAMES.values()) + " | any space |",
             "|---|" + "---|" * (len(SPACES) + 1)]
    for method in METHODS:
        cells, any_space = [], set()
        for space in SPACES:
            done = [key for key, outcome in outcomes.items() if key[0] == method and key[4] == space and outcome]
            working = [key for key in done if outcomes[key]["works"]]
            held = [key for key in working if (outcomes[key]["tab"] or 0) > SUCCESS_THRESHOLD]
            any_space |= {key[1:4] for key in working}
            cells.append(f"{len(working)} / {len(done)} (held out above 0.5: {len(held)})")
        lines.append(f"| {METHOD_NAMES[method]} | " + " | ".join(cells) + f" | {len(any_space)} |")
    greedy = greedy_outcomes()
    cells, any_space = [], set()
    for space in SPACES:
        working = [key for key, (works, _) in greedy.items() if key[0] == space and works]
        held = [key for key in working if (greedy[key][1] or 0) > SUCCESS_THRESHOLD]
        any_space |= {key[1:] for key in working}
        cells.append(f"{len(working)} / 42 (held out above 0.5: {len(held)})")
    lines.append("| greedy (earlier searches) | " + " | ".join(cells) + f" | {len(any_space)} |")
    lines += ["", "Each cell: working combinations / finished ones (out of 42), and how many working sets also "
              "push the TAB connected-text mean above 0.5. Greedy, for comparison: heads only = `search.py` "
              "stages 1-2; MLP lesions only = `em_search.py` stage 1; both = `em_search.py` stages 1 and 3 "
              "(heads swept on top of the best stage-1 lesion, not a joint search).", ""]

    lines += ["**By model** (working combinations out of 6: two tasks, three objectives; spaces 1 / 2 / 3)", "",
              "| model | " + " | ".join(METHOD_NAMES[method] for method in METHODS) + " | greedy |",
              "|---|" + "---|" * (len(METHODS) + 1)]
    for model in NAMES:
        cells = []
        for method in METHODS:
            counts = []
            for space in SPACES:
                done = [outcomes.get((method, model, task, objective, space)) for task in TASKS for objective in OBJECTIVES]
                finished = [outcome for outcome in done if outcome]
                counts.append(f"{sum(outcome['works'] for outcome in finished)}" + ("" if len(finished) == 6 else "…"))
            cells.append(" / ".join(counts))
        cells.append(" / ".join(str(sum(greedy[(space, model, task, objective)][0] for task in TASKS
                                        for objective in OBJECTIVES)) for space in SPACES))
        lines.append(f"| {NAMES[model]} | " + " | ".join(cells) + " |")
    lines += ["", "… : not all six finished yet.", ""] if any("…" in line for line in lines[-len(MODELS):]) else [""]

    for number_of_section, method in enumerate(METHODS, start=2):
        lines += [f"## {number_of_section}. {METHOD_NAMES[method]}", ""]
        if method == ("mask", "soft"):
            lines += ["Cell: objective value (bold when it works), the checkpoint (c1-c6 along the lambda path), "
                      "the number of components with amount above 0.0005 and their summed amount Σk; held-out "
                      "scores. A combination that does not work shows its best value in parentheses.", ""]
        elif method == ("mask", "binary"):
            lines += ["Cell: objective value (of the pruned set when it works), checkpoint, the yes/no set "
                      "(amount > 0.5; m matrices, h heads) → after pruning; held-out scores of the reported set.", ""]
        else:
            lines += ["Cell: objective value of the pruned set, the round that found the set (r1-r20), its size → "
                      "after pruning; held-out scores of the pruned set. † a single component already works alone "
                      "(the greedy search's stage 1), so the search started with a working component among its "
                      "favourites. A combination that does not work shows its best value and the number of distinct "
                      "sets tried (at most 1,280).", ""]
        for space in SPACES:
            lines += [f"**{SPACE_NAMES[space]}**", "",
                      "| model | " + " | ".join(f"{task} {objective}" for task, objective in COLUMNS) + " |",
                      "|---|" + "---|" * len(COLUMNS)]
            for model in NAMES:
                lines.append(f"| {NAMES[model]} | " + " | ".join(
                    cell(outcomes.get((method, model, task, objective, space))) for task, objective in COLUMNS)
                             + " |")
            lines.append("")

    lines += [f"## {len(METHODS) + 2}. Time per experiment (minutes)", "",
              "One experiment is one model, task, objective and space. Learned mask: training (6 stages of 50 "
              "steps) + generating and scoring the answers of its 12 masks + pruning. Cross-entropy: generating and "
              "scoring the answers of up to 20 rounds of 64 sets + pruning (its start reuses the greedy searches' "
              "single knockouts, not counted). Generation time is summed over the answers each experiment used, "
              "so it does not depend on preemptions or on answers shared with other experiments. Held-out checks "
              "and model loading are per job and not included. GPUs differ between jobs (column GPU), and small "
              "models ran several searches on one GPU at once (Pythia-1B and Pythia-1.4B four; GPT-2 XL four, "
              "then one per GPU; Pythia-2.8B, Pythia-6.9B and Llama-2-7B two; Llama-2-13B one), so their times "
              "include that sharing.", ""]
    for kind in ("mask", "cem"):
        method = ("mask", "soft") if kind == "mask" else ("cem", None)
        lines += [f"**{'learned mask' if kind == 'mask' else 'cross-entropy'}**", "",
                  "| model | task | GPU | " + " | ".join(f"{objective} {space[0]}" for objective in OBJECTIVES
                                                          for space in SPACES) + " | total |",
                  "|---|---|---|" + "---|" * (len(OBJECTIVES) * len(SPACES) + 1)]
        for model in NAMES:
            for task in TASKS:
                minutes = [outcomes.get((method, model, task, objective, space))
                           for objective in OBJECTIVES for space in SPACES]
                if not any(minutes):
                    continue
                total = sum(outcome["minutes"] for outcome in minutes if outcome)
                devices = sorted({outcome["device"] for outcome in minutes if outcome and outcome["device"]})
                lines.append(f"| {NAMES[model]} | {task} | {', '.join(devices)} | " + " | ".join(
                    "…" if outcome is None else f"{outcome['minutes']:.1f}" for outcome in minutes)
                             + f" | {total:.0f} |")
        lines.append("")
    lines += ["Column labels: objective and space number (1 MLP lesions, 2 heads, 3 both).", ""]

    lines += union_section()
    out_path = RESULTS / "report" / "learned_searches.md"
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
