"""Tables and figures of the knockout search: which heads and MLP blocks, knocked out, make an answer aphasic?

search.py looks, for one generator, one task and one objective, for components
whose knockout makes the model's answer "work". The objectives are read off the
five-way classifier and work above 0.5:

    A  P(aphasia) = P(Conduction/Anomic) + P(Broca) + P(Wernicke)
    B  P(Broca)
    C  P(Wernicke)

The search escalates through three stages and stops at the first that works,

    1_single_layer_heads   every head alone, then each layer's top-k heads together
    2_forward_heads        a sweep from the first layer to the last that greedily adds heads
    3_forward_heads_mlp    the same sweep, adding heads and 256-neuron MLP blocks

and prunes the knockout that worked to a minimal circuit. Each search writes
results/fiveway/search/<model>_<task>_<tag>.json: tag rep1.25_classifier for the
main study, rep1.25_cosine for the cos-sim ablation (the same search with a
classifier-free objective, the mean cosine to the target group's human
transcripts minus the mean cosine to Control's, working above 0). This script
reads all of them and writes

    results/fiveway/report/search_summary.md
        one table per tag with a row per (model, task, objective): the baseline
        value, the stage that worked, the number of components before and after
        pruning, the pruned circuit (L0H25 L1H8 L1B30: H a head, B an MLP block),
        its value, the classifier's argmax class and the first 150 characters of
        its answer; a table of the unablated baselines; and, when both tags exist,
        the classifier search against the cos-sim search on the same objective
    results/fiveway/report/search_summary.json
        the same numbers
    results/fiveway/report/figures/
        <model>_<task>_<tag>_<objective>_stage1   stage 1's single-head values, layers x heads
        <model>_<task>_<tag>_<objective>_sweeps   value against components knocked out, stages 2 and 3
        overview_<tag>                            the best value each stage reached, every model and task
        each as a PDF and a PNG

It reads JSON and draws with matplotlib, so it takes a minute on a CPU. Every
run rebuilds every output from the search files: rerunning it after more
searches finish is how it picks up new results.

Usage (the shell is fine; nothing here needs a GPU):
    .venv/bin/python src/fiveway/report.py
  on other search files, e.g. the hand-made ones test_report.py writes:
    .venv/bin/python src/fiveway/report.py --results-dir runs/fiveway/report_test/synthetic/search \\
        --cache-dir runs/fiveway/report_test/synthetic/cache --out-dir runs/fiveway/report_test/synthetic/report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS, ROOT, RUNS  # noqa: E402

CLASSIFIER_TAG, COSINE_TAG = "rep1.25_classifier", "rep1.25_cosine"
OBJECTIVES = ["A", "B", "C"]
STAGES = ["1_single_layer_heads", "2_forward_heads", "3_forward_heads_mlp"]
STAGE_NAMES = {"1_single_layer_heads": "1 single-layer heads", "2_forward_heads": "2 forward sweep, heads",
               "3_forward_heads_mlp": "3 forward sweep, heads + MLP", None: "none"}
ANSWER_CHARACTERS = 150   # how much of an answer the tables quote

# NOTE: copied from scorer.py and prompts.py instead of
# imported: importing them loads torch, transformer_lens and the classifier
# package, which a script that only reads JSON does not need. test_report.py
# checks that the copies still match.
CLASSES = ["Control", "Conduction/Anomic", "Broca", "Wernicke", "Model_scrambles"]
OBJECTIVE_COLUMNS = {"A": [1, 2, 3], "B": [2], "C": [3]}   # scorer.OBJECTIVES
THRESHOLDS = {"classifier": 0.5, "cosine": 0.0}            # an objective works above this value
MIN_STORY_CHARS = 20                                       # prompts.MIN_STORY_CHARS: shorter answers do not count


def answer_length(text: str) -> int:
    """prompts.answer_length: characters, not counting invisible format characters (Unicode category Cf)."""
    return sum(unicodedata.category(character) != "Cf" for character in text)

MODEL_NAMES = {"gpt2-xl": "GPT-2 XL", "llama2-7b-chat": "Llama-2-7B-chat", "llama2-13b-chat": "Llama-2-13B-chat",
               "pythia-1b": "Pythia-1B", "pythia-1.4b": "Pythia-1.4B", "pythia-2.8b": "Pythia-2.8B",
               "pythia-6.9b": "Pythia-6.9B"}
TASK_NAMES = {"cinderella": "Cinderella", "sandwich": "Sandwich"}
TARGET_NAMES = {"A": "aphasia", "B": "Broca", "C": "Wernicke"}
VALUE_LABELS = {"classifier": {name: f"P({target})" for name, target in TARGET_NAMES.items()},
                "cosine": {name: f"cos({target}) − cos(Control)" for name, target in TARGET_NAMES.items()}}

# House style (src/interp/make_figures.py): serif type, large fonts, no titles,
# a PDF per figure plus a PNG to look at.
plt.rcParams.update({"font.family": "serif", "axes.labelsize": 24, "xtick.labelsize": 22, "ytick.labelsize": 22,
                     "legend.fontsize": 22, "axes.linewidth": 1.2})
PNG_DPI = 100   # the PNGs are for a quick look; the PDFs are vector graphics
# seaborn's "deep" palette in hex, so that seaborn need not be imported.
BLUE, ORANGE, GREEN, RED = "#4c72b0", "#dd8452", "#55a868", "#c44e52"
NULL_GREY = "#b0b0b0"   # cells with no value
# seaborn's cubehelix "ch:s=-.2,r=.6" in hex: make_figures.py's colormap for single-head heatmaps.
SINGLE_HEAD_COLORMAP = LinearSegmentedColormap.from_list(
    "single_head", ["#e3d9c1", "#d8b9a4", "#cc9692", "#b97688", "#9d5b81", "#774476", "#4f3261", "#27213f"])
# Blue below an objective's threshold, red above it; the darkest tenth at each end is left out
# so that the numbers written on the cells stay readable.
WORKS_COLORMAP = LinearSegmentedColormap.from_list("works", plt.cm.RdBu_r(np.linspace(0.1, 0.9, 256)))


# --- Reading the search files ---------------------------------------------------------

def compact(circuit: dict) -> str:
    """A circuit as text in layer order, "L0H25 L1H8 L1B30": H an attention head, B a 256-neuron MLP block."""
    # NOTE: sorting (layer, kind, index) with kind 0 for a
    # head and 1 for a block lists each layer's heads before its MLP blocks,
    # the order in which a transformer block computes them.
    in_order = sorted([(layer, 0, head) for layer, head in circuit["heads"]]
                      + [(layer, 1, block) for layer, block in circuit["mlp_blocks"]])
    return " ".join(f"L{layer}{'H' if kind == 0 else 'B'}{index}" for layer, kind, index in in_order)


def components(circuit: dict) -> set:
    """A circuit's components as a set of (kind, layer, index)."""
    return ({("head", layer, head) for layer, head in circuit["heads"]}
            | {("block", layer, block) for layer, block in circuit["mlp_blocks"]})


def jaccard(first: set, second: set) -> float:
    """Shared elements over all elements: 1 for identical sets, 0 for disjoint ones."""
    return len(first & second) / len(first | second)


def argmax_class(probabilities):
    """The five-way classifier's most probable class, or None for an answer that was not scored."""
    return None if probabilities is None else CLASSES[int(np.argmax(probabilities))]


def best_values(entry: dict) -> dict:
    """The highest objective value any knockout of each stage reached: {stage: value, None if it did not run}."""
    best = {}
    for stage in STAGES:
        if stage not in entry["stages"]:
            best[stage] = None   # an earlier stage worked, so the search stopped before this one
        elif stage == "1_single_layer_heads":
            # NOTE: per_layer_best holds each layer's strongest
            # top-k set, and its k = 1 set is the layer's strongest single head,
            # so these values cover every knockout of stage 1. A layer is None
            # when none of its answers counted.
            values = [knockout["value"] for knockout in entry["stages"][stage]["per_layer_best"] if knockout is not None]
            best[stage] = max(values, default=None)
        else:
            # NOTE: a sweep whose every candidate gave an
            # answer that did not count has an empty trajectory, hence None.
            best[stage] = max((step["value"] for step in entry["stages"][stage]["trajectory"]), default=None)
    return best


def gate_of(search: dict, task: str):
    """Whether a search applied the Cinderella gate: True or False, None for the sandwich task, which has no gate.

    NOTE: search.py turns the gate off for a model whose
    unablated answer is already "no" (results["gate"]), and a search without
    it also counts knockouts that make the model deny knowing the story, so
    its circuits are not comparable to gated ones without that flag. Files
    written before search.py had the switch have no "gate": those searches
    always applied it.
    """
    if task != "cinderella":   # only Cinderella asks "Do you know the story?" (prompts.py)
        return None
    return search.get("gate", True)


def summarize_search(entry: dict, threshold: float) -> dict:
    """One row of a search table: the baseline, how far each stage got, and the pruned circuit."""
    best_per_stage = best_values(entry)
    found_at = entry["found_at_stage"]
    row = {"baseline_value": entry["baseline_value"],
           "baseline_works": entry["baseline_value"] is not None and entry["baseline_value"] > threshold,
           "found_at_stage": found_at, "best_value_per_stage": best_per_stage,
           "best_value": max((value for value in best_per_stage.values() if value is not None), default=None),
           "n_components_found": None, "n_components_pruned": None, "circuit": None, "circuit_text": None,
           "value": None, "probabilities": None, "argmax": None, "answer_start": None}
    if found_at is None:
        return row
    # The knockout that worked, as search.py hands it to the pruning: stage 1's
    # smallest working set, or the sweep's knockout at the step that worked.
    stage = entry["stages"][found_at]
    found = stage["best"] if found_at == "1_single_layer_heads" else stage["found"]
    circuit = entry["pruned"]["circuit"]
    row.update({"n_components_found": found["n_components"], "n_components_pruned": circuit["n_components"],
                "circuit": {"heads": circuit["heads"], "mlp_blocks": circuit["mlp_blocks"]},
                "circuit_text": compact(circuit), "value": circuit["value"],
                "probabilities": circuit["probabilities"], "argmax": argmax_class(circuit["probabilities"]),
                "answer_start": circuit["story"][:ANSWER_CHARACTERS]})
    return row


def compare(classifier_row: dict, cosine_row: dict) -> dict:
    """The classifier search against the cos-sim search on the same model, task and objective."""
    both_found = classifier_row["circuit"] is not None and cosine_row["circuit"] is not None
    # The five-way classifier's view of the cos-sim circuit's answer: search.py
    # scores every answer with the classifier, whichever objective it searches.
    probabilities = cosine_row["probabilities"]
    value = None
    if probabilities is not None:
        value = round(sum(probabilities[column] for column in OBJECTIVE_COLUMNS[cosine_row["objective"]]), 6)
    # NOTE: when the unablated answer already works under
    # the cos-sim objective, every knockout that keeps it working "works", so
    # the cos-sim circuit is typically the single head that raises the margin
    # most, and its overlap with the classifier's circuit says little. The flag
    # travels with the row.
    return {"model": cosine_row["model"], "task": cosine_row["task"], "objective": cosine_row["objective"],
            "cosine_baseline_value": cosine_row["baseline_value"], "cosine_baseline_works": cosine_row["baseline_works"],
            "classifier_circuit": classifier_row["circuit_text"], "cosine_circuit": cosine_row["circuit_text"],
            "jaccard": (jaccard(components(classifier_row["circuit"]), components(cosine_row["circuit"]))
                        if both_found else None),
            "cosine_answer_classifier_value": value,
            "cosine_answer_works_under_classifier": None if value is None else value > THRESHOLDS["classifier"],
            "cosine_answer_argmax": argmax_class(probabilities), "cosine_answer_probabilities": probabilities}


def cached_single_heads(cache_path: Path) -> dict:
    """search.py's cached answer of every single-head knockout: {(layer, head): record}."""
    records = {}
    for line in cache_path.read_text().splitlines():
        # NOTE: a job preempted while appending can leave a
        # cut-off last line; search.py skips it too.
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if len(record["heads"]) == 1 and not record["mlp_blocks"]:
            records[tuple(record["heads"][0])] = record
    return records


def counts(record: dict, gate: bool) -> bool:
    """Whether search.py counts a cached answer (search.Evaluator.valid): at least 20 characters long (not counting
    invisible format characters), and not a "no" to "Do you know the story of Cinderella?" while that gate applies."""
    return answer_length(record["story"]) >= MIN_STORY_CHARS and (not gate or record["answer"] is not False)


def single_head_values(entry: dict, objective: str, single_heads: dict, gate: bool) -> np.ndarray:
    """Stage 1's value of each single-head knockout, NaN where the answer did not count.

    NOTE: search.py marks an answer that did not count with
    -1 and stores every negative value as null. Probabilities are never
    negative, but cos-sim margins are: in pythia-1b's sandwich search every
    single head's margin is below 0, and all 128 are null. So for the cosine
    objective the values come from search.py's answer cache (single_heads)
    when it is on this disk, counted under the search's own gate.

    NOTE: whether an answer counts is decided here, as
    search.py decides it, rather than read off the cache: caches written
    before search.py's gate switch carry a "valid" flag, newer ones do not,
    and in a search run without the gate a "no" to the Cinderella question
    counts. The rule gives the old flag back for the old caches, whose
    searches always applied the gate.
    """
    rows = entry["stages"]["1_single_layer_heads"]["single_head_values"]
    values = np.array([[np.nan if value is None else value for value in row] for row in rows])
    # NOTE: values: (n_layers, n_heads), one cell per single-head knockout.
    for (layer, head), record in single_heads.items():
        values[layer, head] = record["cosine"][objective] if counts(record, gate) else np.nan
    return values


# --- Markdown ---------------------------------------------------------------------------

def number(value, bold: bool = False) -> str:
    """A value with three decimals (bold when asked), or a dash when there is none."""
    if value is None:
        return "–"
    return f"**{value:.3f}**" if bold else f"{value:.3f}"


def top_class(probabilities) -> str:
    """The five-way classifier's top class with its probability, "Wernicke (0.81)", or a dash."""
    if probabilities is None:
        return "–"
    return f"{argmax_class(probabilities)} ({max(probabilities):.2f})"


def snippet(text) -> str:
    """The start of an answer, fit for a Markdown table cell."""
    if text is None:
        return "–"
    # NOTE: a "|" would end the table cell and "<" could open an
    # HTML tag, so both are escaped; line breaks would end the table row, so
    # they are shown as ⏎.
    return text[:ANSWER_CHARACTERS].replace("|", "\\|").replace("<", "&lt;").replace("\n", " ⏎ ")


def markdown_table(header: list, rows: list) -> list:
    """The lines of a GitHub-flavored Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def markdown_report(rows: dict, kinds: dict, not_finished: dict, no_file: dict, baselines: list,
                    comparison: list, results_dir) -> str:
    """search_summary.md: an explanation, then the search tables, the baselines and the comparison."""
    lines = ["# Knockout search: summary", "",
             f"Written by `src/fiveway/report.py` from search.py's `{results_dir}/<model>_<task>_<tag>.json`; "
             "figures in `figures/`.", "",
             "- **Objectives.** Under the five-way classifier (tag `*_classifier`): **A** P(aphasia) = "
             "P(Conduction/Anomic) + P(Broca) + P(Wernicke), **B** P(Broca), **C** P(Wernicke); a knockout works "
             "above 0.5. The cos-sim ablation (tag `*_cosine`) scores the mean MiniLM cosine to the target group's "
             "human transcripts minus that to Control's; it works above 0.",
             "- **Stages.** 1: single-layer heads (each head alone, then each layer's top-k heads); 2: a forward "
             "sweep adding heads; 3: a forward sweep adding heads and 256-neuron MLP blocks. The search stops at "
             "the first stage that works and prunes that knockout to a minimal knockout set (not a circuit: nothing "
             "shows that the set computes the behaviour).",
             "- **Columns.** *baseline*: the unablated model's value, in bold when it already works. *best*: the "
             "highest value any stage reached, before pruning. *found* and *pruned*: the number of components "
             "before and after pruning. *knockout set*: `L<layer>H<head>` an attention head, `L<layer>B<block>` a "
             "256-neuron MLP block. *value*, *argmax* (the five-way classifier's top class and its probability) "
             "and *answer* belong to the pruned knockout set; an answer's line breaks are shown as ⏎.", ""]
    for tag, tag_rows in rows.items():
        lines += [f"## Searches `{tag}`", ""]
        if not tag_rows:
            lines += ["No results file yet.", ""]
            continue
        threshold = THRESHOLDS[kinds[tag]]
        table = []
        for row in tag_rows:
            circuit_text = "–" if row["circuit_text"] is None else f"`{row['circuit_text']}`"
            table.append([row["model"], row["task"], row["objective"], number(row["baseline_value"], row["baseline_works"]),
                          STAGE_NAMES[row["found_at_stage"]], number(row["best_value"]),
                          row["n_components_found"] or "–", row["n_components_pruned"] or "–", circuit_text,
                          number(row["value"]), top_class(row["probabilities"]), snippet(row["answer_start"])])
        lines += markdown_table(["model", "task", "objective", "baseline", "worked at", "best", "found", "pruned",
                                 "knockout set", "value", "argmax", "answer (first 150 characters)"], table)
        lines += ["", f"A knockout works above {threshold:g} here."]
        ungated = list(dict.fromkeys(f"{row['model']} {row['task']}" for row in tag_rows if row["gate"] is False))
        if ungated:
            lines += [f"Searched without the Cinderella gate (the unablated model already says it does not know the "
                      f"story, so a knockout's \"no\" counts too): {', '.join(ungated)}."]
        if not_finished[tag]:
            lines += [f"Not finished yet: {', '.join(' '.join(item) for item in not_finished[tag])}."]
        if no_file[tag]:
            lines += [f"No results file yet: {', '.join(' '.join(item) for item in no_file[tag])}."]
        lines += [""]

    lines += ["## Unablated baselines", "",
              "The model's own answer, scored by the five-way classifier (the same answer under both tags).", ""]
    table = []
    for row in baselines:
        probabilities = row["probabilities"] or [None] * len(CLASSES)
        table.append([row["model"], row["task"], "yes" if row["valid"] else "no"]
                     + [number(probability) for probability in probabilities]
                     + [row["argmax"] or "–", snippet(row["answer_start"])])
    lines += markdown_table(["model", "task", "scored"] + [f"P({name})" for name in CLASSES]
                            + ["argmax", "answer (first 150 characters)"], table) + [""]

    if comparison:
        lines += [f"## Classifier search (`{CLASSIFIER_TAG}`) against cos-sim search (`{COSINE_TAG}`)", "",
                  "*cos-sim unablated*: the unablated answer's cos-sim margin, in bold when it already works; the "
                  "cos-sim search then works with any knockout that keeps it working, and its knockout set is little "
                  "more than the strongest single head. *Jaccard*: shared components over all components of the two "
                  "sets. The last two columns are the five-way classifier's view of the cos-sim set's "
                  "answer: its value on the same objective (bold when it works, above 0.5) and its top class with "
                  "that class's probability.", ""]
        table = []
        for row in comparison:
            classifier_circuit = "none" if row["classifier_circuit"] is None else f"`{row['classifier_circuit']}`"
            cosine_circuit = "none" if row["cosine_circuit"] is None else f"`{row['cosine_circuit']}`"
            table.append([row["model"], row["task"], row["objective"],
                          number(row["cosine_baseline_value"], row["cosine_baseline_works"]),
                          classifier_circuit, cosine_circuit, number(row["jaccard"]),
                          number(row["cosine_answer_classifier_value"], row["cosine_answer_works_under_classifier"]),
                          top_class(row["cosine_answer_probabilities"])])
        lines += markdown_table(["model", "task", "objective", "cos-sim unablated", "classifier knockout set",
                                 "cos-sim knockout set", "Jaccard", "classifier value of the cos-sim answer",
                                 "argmax of the cos-sim answer"], table)
        lines += [""]

    lines += ["## Figures", "",
              "- `figures/overview_<tag>`: per task and model, each objective's baseline and the best value each "
              "stage reached; red works, blue does not, a black frame marks the stage that worked, grey did not "
              "run (an earlier stage worked), – an unablated answer that did not count, ? not finished yet; "
              "\"(no gate)\" after a model's name: its Cinderella search ran without the gate. The cos-sim margin "
              "is cos(target) − cos(Control).",
              "- `figures/<model>_<task>_<tag>_<objective>_stage1`: stage 1's value of every single-head knockout "
              "(layers down, heads across; grey where the answer did not count, a star where it works; the black "
              "line on the colorbar is the unablated value). The colors span the 5th to 99th percentile of the "
              "values, and a pointed end of the colorbar marks cells beyond its range, painted in its end color.",
              "- `figures/<model>_<task>_<tag>_<objective>_sweeps`: stages 2 and 3, the value after each greedy "
              "pick against the number of components knocked out (only when stage 1 did not work).", ""]
    return "\n".join(lines)


def write_atomically(path: Path, text: str) -> None:
    """Write through a temporary file, so that a killed run never leaves half a file behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


# --- Figures ----------------------------------------------------------------------------

def save(fig, figure_dir: Path, name: str) -> None:
    """Save a figure as a PDF (vector, for papers and slides) and a PNG (to look at)."""
    # NOTE: the names contain dots (pythia-1.4b, rep1.25),
    # so the extension is appended to the name: Path.with_suffix would replace
    # everything after the last dot.
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.png", bbox_inches="tight", dpi=PNG_DPI)
    plt.close(fig)


def style_heatmap_grid(ax, n_rows: int, n_columns: int) -> None:
    """No spines, no tick marks, and thin white lines between the cells."""
    ax.set_xticks(np.arange(n_columns + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def attach_colorbar(image, ax, fig, label: str, extend: str = "neither"):
    """A colorbar exactly as tall as the heatmap next to it; extend ("min", "max", "both") draws pointed ends."""
    colorbar_ax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
    colorbar = fig.colorbar(image, cax=colorbar_ax, extend=extend)
    colorbar.set_label(label, fontsize=24)
    colorbar.ax.tick_params(labelsize=22)
    return colorbar


def tick_step(n: int) -> int:
    """Label every 1st, 2nd, 4th or 8th row or column, whichever keeps the labels at 12 or fewer."""
    return next(step for step in (1, 2, 4, 8) if n / step <= 12)


def plot_single_heads(values: np.ndarray, baseline_value, threshold: float, label: str, figure_dir: Path, name: str):
    """Stage 1: the value of each single head's knockout, layers down and heads across."""
    n_layers, n_heads = values.shape   # NOTE: (n_layers, n_heads)
    finite = values[np.isfinite(values)]
    # NOTE: most knockouts barely move the objective, so the
    # colors span the 5th to 99th percentile of the values (and the unablated
    # value) rather than the whole range: the heads that matter stand out.
    # Cells beyond that range take its end colors, so the strongest heads all
    # look alike: a star marks every head that works, and their exact values
    # are in the search file (single_head_values).
    low, high = np.percentile(finite, [5, 99]) if finite.size else (0.0, 1.0)
    if baseline_value is not None:
        low, high = min(low, baseline_value), max(high, baseline_value)
    # NOTE: a pointed end on the colorbar says that some cells
    # lie beyond that end of its range and are painted in its end color.
    below, above = bool(finite.size and finite.min() < low), bool(finite.size and finite.max() > high)
    extend = {(False, False): "neither", (True, False): "min", (False, True): "max", (True, True): "both"}[below, above]
    fig, ax = plt.subplots(figsize=(11, 10))
    # NOTE: imshow masks NaN cells and paints them in the
    # colormap's "bad" color, here grey: answers that did not count.
    # interpolation="nearest" keeps every cell one flat color; the default
    # smooths neighboring cells into each other when the image is resampled.
    image = ax.imshow(values, aspect="auto", cmap=SINGLE_HEAD_COLORMAP.with_extremes(bad=NULL_GREY),
                      vmin=low, vmax=high, interpolation="nearest")
    style_heatmap_grid(ax, n_layers, n_heads)
    ax.set_xticks(range(0, n_heads, tick_step(n_heads)))
    ax.set_yticks(range(0, n_layers, tick_step(n_layers)))
    ax.set_xlabel("attention head")
    ax.set_ylabel("layer")
    # NOTE: NaN > threshold is False, so heads whose answer did
    # not count are never starred.
    working_layers, working_heads = np.nonzero(values > threshold)
    if working_layers.size:
        ax.scatter(working_heads, working_layers, marker="*", s=320, color="white", edgecolors="black",
                   linewidths=1.2, zorder=3, label=f"works (above {threshold:g})")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    if baseline_value is None:
        attach_colorbar(image, ax, fig, label, extend)
    else:
        colorbar = attach_colorbar(image, ax, fig, f"{label} (unablated {baseline_value:.3f})", extend)
        colorbar.ax.axhline(baseline_value, color="black", linewidth=3)
    save(fig, figure_dir, name)


def plot_sweeps(entry: dict, kind: str, label: str, figure_dir: Path, name: str):
    """Stages 2 and 3: the objective value after each greedy pick, against the number of components knocked out."""
    threshold = THRESHOLDS[kind]
    baseline = np.nan if entry["baseline_value"] is None else entry["baseline_value"]
    fig, ax = plt.subplots(figsize=(13, 6))
    for stage, color, legend in (("2_forward_heads", BLUE, "stage 2: heads"),
                                 ("3_forward_heads_mlp", ORANGE, "stage 3: heads + MLP")):
        if stage not in entry["stages"]:
            continue
        trajectory = entry["stages"][stage]["trajectory"]
        # NOTE: every step of a sweep commits one more
        # component, so a step's n_components is its place on the x axis, and
        # x = 0 is the unablated model. Both sweeps start from nothing knocked
        # out. A step can lower the value: the sweep commits its best pick even
        # when that pick gains nothing.
        counts = [0] + [step["n_components"] for step in trajectory]
        values = [baseline] + [step["value"] for step in trajectory]
        ax.plot(counts, values, color=color, linewidth=3, marker="o", markersize=5, label=legend)
        block_steps = [step for step in trajectory if step["kind"] == "block"]
        if block_steps:
            ax.scatter([step["n_components"] for step in block_steps], [step["value"] for step in block_steps],
                       marker="s", s=70, color=color, edgecolors="black", linewidths=1, zorder=3,
                       label="MLP block pick")
    stage1_best = entry["stages"]["1_single_layer_heads"]["best"]
    if stage1_best is not None:
        ax.scatter([stage1_best["n_components"]], [stage1_best["value"]], marker="D", s=150, color=GREEN,
                   edgecolors="black", zorder=4, label="stage 1: best set")
    if "pruned" in entry:
        circuit = entry["pruned"]["circuit"]
        ax.scatter([circuit["n_components"]], [circuit["value"]], marker="*", s=600, color=RED,
                   edgecolors="black", zorder=5, label="pruned set")
    ax.axhline(threshold, linestyle="--", color=RED, alpha=0.55, linewidth=1.4)
    # NOTE: get_yaxis_transform() takes x in axes coordinates (0
    # to 1, left to right) and y in data coordinates, so the note sits at the
    # right edge, on the threshold line, whatever the x range.
    ax.text(0.99, threshold, f"works above {threshold:g}", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=18, color=RED)
    if kind == "classifier":
        ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("components knocked out")
    ax.set_ylabel(label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=20,
              columnspacing=1.2, handletextpad=0.4)
    save(fig, figure_dir, name)


def plot_overview(tag_rows: list, not_finished: list, kind: str, figure_dir: Path, name: str):
    """Every task and model (rows) and objective (column groups): the baseline and the best value each stage reached."""
    # Rows: the tasks one after the other, each with its models by family and size (the order of MODELS).
    present = [(task, model) for task in TASKS for model in MODELS
               if any(row["task"] == task and row["model"] == model for row in tag_rows)]
    columns = [(objective, column) for objective in OBJECTIVES for column in ["baseline"] + STAGES]
    # NOTE: grid: (len(present) task-model rows, 3 objectives x 4 columns); NaN where nothing ran.
    grid = np.full((len(present), len(columns)), np.nan)
    for row in tag_rows:
        i = present.index((row["task"], row["model"]))
        for j, (objective, column) in enumerate(columns):
            if objective == row["objective"]:
                value = row["baseline_value"] if column == "baseline" else row["best_value_per_stage"][column]
                grid[i, j] = np.nan if value is None else value

    # NOTE: the colors diverge at the threshold, so a cell
    # is red when that stage reached a knockout that works and blue when it did
    # not; the range is symmetric around the threshold: [0, 1] for
    # probabilities, the largest distance from 0 for cos-sim margins.
    threshold = THRESHOLDS[kind]
    spread = 0.5 if kind == "classifier" else float(np.nanmax(np.abs(grid - threshold)))
    low, high = threshold - spread, threshold + spread
    # NOTE: each task's name is written sideways next to its
    # rows, so a task with only one or two models done yet gets taller rows to
    # leave the name room.
    rows_per_task = [sum(row_task == task for row_task, _ in present) for task in TASKS]
    row_height = max(0.7, 2.4 / min(n for n in rows_per_task if n > 0))
    fig, ax = plt.subplots(figsize=(16, 1.5 + row_height * len(present)))
    image = ax.imshow(grid, aspect="auto", cmap=WORKS_COLORMAP.with_extremes(bad=NULL_GREY), vmin=low, vmax=high,
                      interpolation="nearest")
    style_heatmap_grid(ax, len(present), len(columns))
    for i in range(len(present)):
        for j in range(len(columns)):
            if np.isfinite(grid[i, j]):
                shade = (grid[i, j] - low) / (high - low)   # 0 at the blue end, 1 at the red end
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=17,
                        color="white" if abs(shade - 0.5) > 0.35 else "black")
    # A black frame around the stage that worked: values just above the
    # threshold are nearly white, so the color alone would not show it.
    for row in tag_rows:
        if row["found_at_stage"] is not None:
            i = present.index((row["task"], row["model"]))
            j = columns.index((row["objective"], row["found_at_stage"]))
            # NOTE: clip_on=False keeps the frame's outer half
            # visible on the first and last rows, where it lies outside the axes.
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=4, zorder=3,
                                   clip_on=False))
    # A dash where the unablated answer did not count (shorter than 20
    # characters, or, in files from before search.py's gate switch, "no" to
    # the Cinderella question), so it has no value.
    # NOTE: a stage that ran but never got an answer that
    # counted also has no value; it stays grey like a stage that did not run.
    for row in tag_rows:
        if row["baseline_value"] is None:
            ax.text(columns.index((row["objective"], "baseline")), present.index((row["task"], row["model"])), "–",
                    ha="center", va="center", fontsize=17)
    for model, task, objective in not_finished:
        if (task, model) in present:
            for j, (column_objective, _) in enumerate(columns):
                if column_objective == objective:
                    ax.text(j, present.index((task, model)), "?", ha="center", va="center", fontsize=17)

    # Thick lines between the objectives and between the tasks, thin ones between model families.
    n_columns = len(columns)
    ax.vlines([3.5, 7.5], -0.5, len(present) - 0.5, color="black", linewidth=3)
    for i in range(1, len(present)):
        (task, model), (previous_task, previous_model) = present[i], present[i - 1]
        if task != previous_task:
            ax.hlines(i - 0.5, -0.5, n_columns - 0.5, color="black", linewidth=3)
        elif MODELS[model].family != MODELS[previous_model].family:
            ax.hlines(i - 0.5, -0.5, n_columns - 0.5, color="black", linewidth=1)
    # Each objective's name above the middle of its group of four columns.
    for group, objective in enumerate(OBJECTIVES):
        ax.text(4 * group + 1.5, -0.65, f"{objective}: {TARGET_NAMES[objective]}", ha="center", va="bottom",
                fontsize=24)
    ax.set_xticks(range(n_columns), ["base", "1", "2", "3"] * len(OBJECTIVES))
    # A Cinderella search run without the gate (see gate_of) says so next to the model's name.
    ungated = {(row["task"], row["model"]) for row in tag_rows if row["gate"] is False}
    ax.set_yticks(range(len(present)), [MODEL_NAMES[model] + (" (no gate)" if (task, model) in ungated else "")
                                        for task, model in present])
    ax.set_xlabel("search stage (base: the unablated model)")
    # NOTE: a gutter left of the first column holds each
    # task's name, as in the house style's grouped heatmaps.
    ax.set_xlim(-1.5, n_columns - 0.5)
    for task in TASKS:
        task_rows = [i for i, (row_task, _) in enumerate(present) if row_task == task]
        if task_rows:
            ax.text(-1.0, (task_rows[0] + task_rows[-1]) / 2, TASK_NAMES[task], ha="center", va="center",
                    rotation=90, fontsize=24)
    attach_colorbar(image, ax, fig, "best P(target)" if kind == "classifier" else "best cos-sim margin")
    save(fig, figure_dir, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                        help="the models to report (default: all seven)")
    parser.add_argument("--tags", nargs="+", default=[CLASSIFIER_TAG, COSINE_TAG],
                        help="search tags to report; the comparison needs both default tags")
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "search",
                        help="where search.py's <model>_<task>_<tag>.json files are")
    parser.add_argument("--cache-dir", type=Path, default=RUNS / "fiveway",
                        help="where search.py's answer caches are (<model>/<tag>/cache_<task>.jsonl)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS / "report")
    args = parser.parse_args()
    # NOTE: ROOT / path is path itself when path is absolute, so
    # the directories may be given relative to the repository or in full.
    results_dir, cache_dir, out_dir = ROOT / args.results_dir, ROOT / args.cache_dir, ROOT / args.out_dir
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    # 1. Read every search file there is: {(tag, model, task): results}.
    searches = {}
    for tag in args.tags:
        for model in args.models:
            for task in TASKS:
                path = results_dir / f"{model}_{task}_{tag}.json"
                if path.exists():
                    searches[tag, model, task] = json.loads(path.read_text())
    log(f"read {len(searches)} search files from {results_dir}")

    # 2. A row per (tag, model, task, objective); and what is not there yet.
    rows = {tag: [] for tag in args.tags}
    kinds = {}                                  # tag -> "classifier" or "cosine"
    not_finished = {tag: [] for tag in args.tags}   # tag -> [(model, task, objective)]
    no_file = {tag: [] for tag in args.tags}        # tag -> [(model, task)]
    for tag in args.tags:
        for model in args.models:
            for task in TASKS:
                if (tag, model, task) not in searches:
                    no_file[tag].append((model, task))
                    continue
                search = searches[tag, model, task]
                kinds[tag] = search["config"]["objective_kind"]
                for objective in OBJECTIVES:
                    if objective in search["objectives"]:
                        row = {"model": model, "task": task, "objective": objective, "gate": gate_of(search, task)}
                        row.update(summarize_search(search["objectives"][objective], THRESHOLDS[kinds[tag]]))
                        rows[tag].append(row)
                    # NOTE: search.py writes an objective
                    # into the file only once it is done, so an objective it
                    # was asked for that is missing is still running.
                    elif objective in search["config"]["objectives"]:
                        not_finished[tag].append((model, task, objective))
        log(f"{tag}: {len(rows[tag])} objectives done, {len(not_finished[tag])} not finished, "
            f"{len(no_file[tag])} model-task pairs without a file")

    # 3. The unablated baselines, one per (model, task).
    # NOTE: every tag generates the same unablated answer
    # (same model, prompts and decoding) and scores it with the same
    # classifier, so the first tag with a file stands for all; their
    # probabilities differ by float rounding only (test_report.py checks it).
    baselines = []
    for model in args.models:
        for task in TASKS:
            tags_with_file = [tag for tag in args.tags if (tag, model, task) in searches]
            if tags_with_file:
                baseline = searches[tags_with_file[0], model, task]["baseline"]
                baselines.append({"model": model, "task": task, "valid": baseline["valid"],
                                  "probabilities": baseline["probabilities"],
                                  "argmax": argmax_class(baseline["probabilities"]),
                                  "answer_start": baseline["story"][:ANSWER_CHARACTERS]})

    # 4. The classifier search against the cos-sim search, objective by objective.
    comparison = []
    if CLASSIFIER_TAG in rows and COSINE_TAG in rows:
        cosine_rows = {(row["model"], row["task"], row["objective"]): row for row in rows[COSINE_TAG]}
        for row in rows[CLASSIFIER_TAG]:
            key = (row["model"], row["task"], row["objective"])
            if key in cosine_rows:
                comparison.append(compare(row, cosine_rows[key]))

    # 5. The tables, as Markdown and as JSON.
    summary = {"results_dir": os.path.relpath(results_dir, ROOT), "thresholds": THRESHOLDS, "classes": CLASSES,
               "searches": {tag: {"objective_kind": kinds.get(tag), "rows": rows[tag],
                                  "not_finished": not_finished[tag], "no_results_file": no_file[tag]}
                            for tag in args.tags},
               "baselines": baselines, "comparison": comparison}
    write_atomically(out_dir / "search_summary.json", json.dumps(summary, indent=1))
    markdown = markdown_report(rows, kinds, not_finished, no_file, baselines, comparison,
                               os.path.relpath(results_dir, ROOT))
    write_atomically(out_dir / "search_summary.md", markdown)
    print(markdown, flush=True)
    log(f"wrote {out_dir / 'search_summary.md'} and {out_dir / 'search_summary.json'}")

    # 6. The figures of every search and objective, then one overview per tag.
    n_figures = 0
    for (tag, model, task), search in searches.items():
        kind = search["config"]["objective_kind"]
        cache_path = cache_dir / model / tag / f"cache_{task}.jsonl"
        single_heads = cached_single_heads(cache_path) if kind == "cosine" and cache_path.exists() else {}
        if kind == "cosine" and not single_heads:
            log(f"  {model} {task} {tag}: no answer cache at {cache_path}; negative single-head values stay grey")
        # NOTE: a search file can be written anew (a rerun
        # after a change to search.py), and an objective that then works at
        # stage 1 draws no sweeps figure. Removing this search's figures before
        # drawing them again keeps a figure of the superseded run from sitting
        # next to tables of the new one.
        for objective in OBJECTIVES:
            for figure in ("stage1", "sweeps"):
                for extension in ("pdf", "png"):
                    (figure_dir / f"{model}_{task}_{tag}_{objective}_{figure}.{extension}").unlink(missing_ok=True)
        for objective, entry in search["objectives"].items():
            name = f"{model}_{task}_{tag}_{objective}"
            label = VALUE_LABELS[kind][objective]
            # The search's own gate, as search.py stored it (see gate_of: older files always applied it). The
            # sandwich task asks no yes/no question, so every cached yes/no answer is None and the gate changes nothing.
            values = single_head_values(entry, objective, single_heads, search.get("gate", True))
            plot_single_heads(values, entry["baseline_value"], THRESHOLDS[kind], label, figure_dir, f"{name}_stage1")
            n_figures += 1
            if "2_forward_heads" in entry["stages"]:
                plot_sweeps(entry, kind, label, figure_dir, f"{name}_sweeps")
                n_figures += 1
        log(f"  figures of {model} {task} {tag}")
    for tag in args.tags:
        if rows[tag]:
            plot_overview(rows[tag], not_finished[tag], kinds[tag], figure_dir, f"overview_{tag}")
            n_figures += 1
    log(f"wrote {n_figures} figures (PDF and PNG) to {figure_dir}")


if __name__ == "__main__":
    main()
