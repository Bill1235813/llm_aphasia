"""Where our lesioned and steered answers lie among the human transcripts: one t-SNE per task.

transcript_tsne.py maps each task's human transcripts and the classifier's model scrambles in the text space of the
classifier's encoder (frozen BGE-large). This script embeds the models' own answers to the task prompt the same way and
fits one t-SNE per task on everything together, so that answers unlike any human speech form islands of their own
instead of being pressed into the human map:

    intact     each model's answer without any intervention (11 per task)
    lesioned   the answer of each CLAMP lesion for the Broca and the Wernicke target, the sets of Figure 4
               (11 models x 2 targets per task)
    steered    the answer to the task prompt with the task's Broca or Wernicke vector, at the setting that the sweep
               chose for the human evaluation (steer_annotate.chosen_setting; 11 models x 2 targets per task)

The general aphasia target is left out, as in the paper's report of the types.

A lesion reaches its target by the final-answer rule (scorer.works_finally); a steering setting by the sweep's rule,
a target probability above 0.5, as in Section 5. Answers too short to score (prompts.long_enough) are left out.

The placement is measured in the full embedding too: the classes of each answer's 10 nearest human transcripts or
model scrambles (cosine), averaged per task, method and target (results/fiveway/tsne/outputs.json).

Figure: results/fiveway/figures/tsne_outputs.{pdf,png}, rows lesioning and steering, columns the two tasks.

Usage (CPU job; the human embeddings come from transcript_tsne.py's cache):
    GPUS=0 CPUS=32 bash slurm/fiveway_submit.sh tsne-outputs 32G fiveway/output_tsne.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import paper_style  # noqa: E402
from fiveway.paper_style import plt  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.prompts import long_enough  # noqa: E402
from fiveway.scorer import works_finally  # noqa: E402
from fiveway.steer_annotate import chosen_setting  # noqa: E402
from fiveway.steer_generate import load_vectors  # noqa: E402
from fiveway.transcript_tsne import N_NEIGHBORS, PLOT_CLASS, TASKS, embed, load_task, tsne  # noqa: E402

STUDY_MODELS = ["gpt2-xl", "llama2-7b-chat", "llama2-13b-chat", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
                "pythia-6.9b", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
REPORTED_TARGETS = ["B", "C"]   # Broca and Wernicke; the answers of the aphasia target stay out of the figure
TARGET_COLORS = {"B": paper_style.CLASS_COLORS["broca"], "C": paper_style.CLASS_COLORS["wernicke"]}
TARGET_NAMES = {"B": "Broca", "C": "Wernicke"}
NEIGHBOR_CLASSES = ["control", "conduction/anomic", "broca", "wernicke", "model_scrambles"]


def model_answers(model: str, task: str) -> list:
    """The intact, lesioned and steered answers of one model to one task, as dicts with method, target, works, text."""
    grown = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
    if grown.exists():
        document = json.loads(grown.read_text())
        reported = {target: document["reported"][target] for target in "ABC"}
    else:
        document = json.loads((RESULTS / "em_attention" / f"{model}_{task}.json").read_text())
        reported = {target: document["reported"][target]["2b_heads_grown"] for target in "ABC"}
    answers = [{"method": "intact", "target": "", "works": False, "text": document["baseline"]["story"]}]
    for target, row in reported.items():
        answers.append({"method": "lesioned", "target": target, "text": row["story"],
                        "works": works_finally(target, row["value"], row["probabilities"])})
    vectors = load_vectors(RESULTS / "steering" / model / f"vectors_{task}.npz")
    sweep = json.loads((RESULTS / "steering" / model / f"sweep_{task}.json").read_text())
    for target in "ABC":
        setting = chosen_setting(model, task, target, vectors)
        entry = next(entry for entry in sweep["entries"] if entry["vector"] == target
                     and entry["layer"] == setting["layer"] and entry["strength"] == setting["strength"])
        answers.append({"method": "steered", "target": target, "text": entry["story"],
                        "works": setting["works_on_own_task"]})
    for answer in answers:
        answer["model"] = model
    return answers


def load_answers(task: str) -> tuple:
    """(answers that are long enough to score, their embeddings, the number left out), cached in runs/."""
    path = RUNS / "fiveway" / "tsne" / f"answers_{task}.npz"
    answers = [answer for model in STUDY_MODELS for answer in model_answers(model, task)]
    scored = [answer for answer in answers if answer["text"] and long_enough(answer["text"])]
    if path.exists() and np.load(path)["texts"].tolist() == [answer["text"] for answer in scored]:
        embeddings = np.load(path)["embeddings"]
    else:
        embeddings = embed([answer["text"] for answer in scored])
        np.savez(path, embeddings=embeddings, texts=np.array([answer["text"] for answer in scored]))
    return scored, embeddings, len(answers) - len(scored)


def neighbor_shares(reference: np.ndarray, reference_classes: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """(n_queries, 5): the share of each class among a query's nearest reference texts."""
    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="cosine").fit(reference).kneighbors(
        queries, return_distance=False)
    # NOTE: reference_classes[neighbors] is n_queries x 10 class names; comparing it with each class name
    # and averaging over the 10 neighbors gives that class's share.
    return np.stack([(reference_classes[neighbors] == name).mean(axis=1) for name in NEIGHBOR_CLASSES], axis=1)


def main():
    summary, layouts = {}, {}
    for task in TASKS:
        data = load_task(task)
        answers, answer_embeddings, n_left_out = load_answers(task)
        # NOTE: the cache holds every target's answers; the figure keeps the intact answers and the
        # Broca and Wernicke ones.
        keep = [i for i, answer in enumerate(answers) if answer["method"] == "intact" or answer["target"] in REPORTED_TARGETS]
        answers, answer_embeddings = [answers[i] for i in keep], answer_embeddings[keep]
        reference_classes = np.array([PLOT_CLASS[group] for group in data["groups"]])
        shares = neighbor_shares(data["embeddings"], reference_classes, answer_embeddings)
        summary[task] = {"left_out_too_short": n_left_out, "groups": {}}
        print(f"{task}: {len(answers)} answers placed, {n_left_out} too short to score; neighbor shares "
              f"({', '.join(NEIGHBOR_CLASSES)})")
        for method in ("intact", "lesioned", "steered"):
            for target in ([""] if method == "intact" else REPORTED_TARGETS):
                rows = [i for i, answer in enumerate(answers) if answer["method"] == method and answer["target"] == target]
                works = [i for i in rows if answers[i]["works"]]
                name = method if method == "intact" else f"{method} {TARGET_NAMES[target]}"
                entry = {"n": len(rows), "n_reaching_target": len(works),
                         "neighbor_shares": dict(zip(NEIGHBOR_CLASSES, shares[rows].mean(axis=0).round(3).tolist())),
                         "neighbor_shares_reaching_target": (dict(zip(NEIGHBOR_CLASSES, shares[works].mean(axis=0).round(3).tolist()))
                                                             if works else None)}
                summary[task]["groups"][name] = entry
                print(f"  {name:20s} n={len(rows):2d} (reach {len(works):2d})  "
                      + "  ".join(f"{share:.2f}" for share in entry["neighbor_shares"].values()))
        points = tsne(np.concatenate([data["embeddings"], answer_embeddings]))
        layouts[task] = (points[:len(reference_classes)], reference_classes, points[len(reference_classes):], answers)
    out_path = RESULTS / "tsne" / "outputs.json"
    out_path.write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(2, 2, figsize=(paper_style.PAGE_WIDTH, 5.6))
    for column, task in enumerate(TASKS):
        reference_points, reference_classes, answer_points, answers = layouts[task]
        for row, method in enumerate(("lesioned", "steered")):
            ax = axes[row, column]
            order = np.random.default_rng(0).permutation(len(reference_points))
            ax.scatter(reference_points[order, 0], reference_points[order, 1], s=4, linewidths=0, alpha=0.35,
                       c=[paper_style.CLASS_COLORS[name] for name in reference_classes[order]])
            # NOTE: the intact answers are drawn in every panel as the reference that each
            # intervention moves away from; a hollow marker is an answer that does not reach its target.
            for index, answer in enumerate(answers):
                if answer["method"] == "intact":
                    ax.scatter(*answer_points[index], s=26, marker="o", facecolor="white", edgecolor=paper_style.INK,
                               linewidths=0.8, zorder=3)
                elif answer["method"] == method:
                    color = TARGET_COLORS[answer["target"]]
                    ax.scatter(*answer_points[index], s=34, marker="^" if method == "lesioned" else "D",
                               facecolor=color if answer["works"] else "white",
                               edgecolor=paper_style.INK if answer["works"] else color, linewidths=0.8, zorder=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(paper_style.GRID)
            ax.set_title(f"{TASKS[task]}, {method} answers")
    background = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, alpha=0.6,
                             color=paper_style.CLASS_COLORS[name]) for name in NEIGHBOR_CLASSES]
    foreground = [plt.Line2D([], [], marker="o", linestyle="", markersize=5.5, markerfacecolor="white",
                             markeredgecolor=paper_style.INK)]
    foreground += [plt.Line2D([], [], marker="s", linestyle="", markersize=5.5, markerfacecolor=TARGET_COLORS[target],
                              markeredgecolor=paper_style.INK) for target in REPORTED_TARGETS]
    # NOTE: short labels keep the legend within the panels' width; the caption says that the dots
    # are human transcripts and model scrambles, triangles lesioned and diamonds steered answers.
    labels = [paper_style.CLASS_NAMES[name] for name in NEIGHBOR_CLASSES] + ["Intact answer"] + [
        f"{TARGET_NAMES[target]} target" for target in REPORTED_TARGETS]
    fig.subplots_adjust(top=0.88, wspace=0.04, hspace=0.12)
    fig.legend(background + foreground, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.2, columnspacing=0.9, fontsize=8)
    path = RESULTS / "figures" / "tsne_outputs"
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    print(f"wrote {out_path} and {path}.pdf/.png")


if __name__ == "__main__":
    main()
