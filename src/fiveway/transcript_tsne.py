"""Where the human transcripts of each aphasia type lie: a t-SNE of their text embeddings, one panel per task.

Section 5 reports Broca's and Wernicke's aphasia and sets the conduction and anomic types aside. This script shows the
reason: in the text space of the classifier's encoder, conduction and anomic transcripts lie among the control
transcripts, while Broca's and Wernicke's transcripts are easier to tell apart from them.

Representation. Each transcript is embedded by BAAI/bge-large-en-v1.5, the encoder that the five-way classifier
fine-tunes, here without the classifier's LoRA adapters, so the space knows nothing of our labels. A transcript's
vector is the encoder's [CLS] vector over its first 384 tokens (the classifier's own limit), scaled to length 1 as
BGE prescribes.

Transcripts. The ones the steering vectors use (steer_collect.find_duplicates): the first copy of each text, without
the texts listed under two groups, so 1,139 Cinderella and 880 sandwich transcripts. The second figure adds the
classifier's model-scramble training answers to the same task: 169 for Cinderella and 90 for sandwich.

Three measures, computed in the 1,024-dimensional embedding rather than read off the t-SNE picture:
    control_neighbors   the share of control transcripts among a transcript's 10 nearest human transcripts (cosine),
                        averaged over the transcripts of a group
    auc_vs_control      a linear probe's AUC for one group against the control transcripts, set up as the probes of
                        steer_analyze.py (C = 0.01, standardized, five folds that keep a speaker's transcripts together)
    scramble_neighbors  with the model scrambles added, the share of scrambles among the 10 nearest texts, averaged
                        over the scrambles and over the human transcripts

Outputs:
    runs/fiveway/tsne/<task>.npz                                  embeddings and groups (from AphasiaBank, not in git)
    results/fiveway/tsne/overlap.json                             both measures, per task and group
    results/fiveway/figures/tsne_transcripts.{pdf,png}            human transcripts only
    results/fiveway/figures/tsne_transcripts_scrambles.{pdf,png}  with the model scrambles

Usage (a CPU job; the embeddings take some minutes, a rerun reads them from runs/):
    GPUS=0 CPUS=32 bash slurm/fiveway_submit.sh tsne 32G fiveway/transcript_tsne.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import paper_style  # noqa: E402
from fiveway.paper_style import plt  # noqa: E402
from fiveway.paths import CLASSIFIER_DIR, RESULTS, RUNS  # noqa: E402
from fiveway.steer_analyze import GROUPS, N_FOLDS, probe_auc, speaker_id  # noqa: E402
from fiveway.steer_collect import find_duplicates, load_transcripts  # noqa: E402

ENCODER = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 384
BATCH_SIZE = 16
N_NEIGHBORS = 10
TASKS = {"cinderella": "Cinderella", "sandwich": "Sandwich"}
EXPECTED_KEPT = {"cinderella": 1139, "sandwich": 880}   # the counts of Appendix H
# The figure's classes are the classifier's: AphasiaBank's anomic and conduction groups form its one class.
PLOT_CLASS = {"Control": "control", "Anomic": "conduction/anomic", "Conduction": "conduction/anomic",
              "Broca": "broca", "Wernicke": "wernicke", "Model_scrambles": "model_scrambles"}


@torch.no_grad()
def embed(texts: list) -> np.ndarray:
    """(n_texts, 1024) unit-length [CLS] vectors of BGE-large."""
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).eval()
    # NOTE: sorting the texts by length keeps each batch's padding small. On a GPU this
    # takes seconds; on CPU some minutes, which spares the group's GPUs for a one-off.
    order = np.argsort([len(text) for text in texts])
    vectors = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for start in range(0, len(texts), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        inputs = tokenizer([texts[i] for i in batch], padding=True, truncation=True, max_length=MAX_TOKENS,
                           return_tensors="pt")
        # NOTE: last_hidden_state is batch x tokens x 1024; BGE's sentence vector is position 0, [CLS].
        cls = model(**inputs).last_hidden_state[:, 0]
        vectors[batch] = torch.nn.functional.normalize(cls, dim=-1).numpy()
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))} of {len(texts)}", flush=True)
    return vectors


def load_task(task: str) -> dict:
    """Embeddings, groups and speakers of a task's kept transcripts followed by its model scrambles (cached)."""
    path = RUNS / "fiveway" / "tsne" / f"{task}.npz"
    if path.exists():
        data = np.load(path)
        return {key: data[key] for key in data.files}
    rows = load_transcripts(task, None)
    _, kept = find_duplicates(rows)
    rows = [row for row, keep in zip(rows, kept) if keep]
    assert len(rows) == EXPECTED_KEPT[task], (task, len(rows))
    scrambles = [row for row in json.loads((CLASSIFIER_DIR / "data" / "model_scrambles" / "transcripts.json").read_text())
                 if task in row["File_DB"].lower()]
    data = {"embeddings": embed([row["text_response"] for row in rows + scrambles]),
            "groups": np.array([row["Group"] for row in rows + scrambles]),
            # NOTE: a model scramble has no speaker; its id stands in, and the probes skip it.
            "speakers": np.array([speaker_id(row) for row in rows] + [row["File_DB"] for row in scrambles])}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def overlap(embeddings: np.ndarray, groups: np.ndarray, speakers: np.ndarray) -> dict:
    """Per human group: the control share among each transcript's neighbors, and a probe's AUC against the controls."""
    human = groups != "Model_scrambles"
    x, labels_of, speakers = embeddings[human], groups[human], speakers[human]
    # NOTE: neighbors: transcripts x 10 indices of the nearest other transcripts (column 0, the transcript
    # itself, is dropped).
    neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="cosine").fit(x).kneighbors(
        x, return_distance=False)[:, 1:]
    control_share = (labels_of[neighbors] == "Control").mean(axis=1)
    result = {"control_share_of_all": float((labels_of == "Control").mean())}
    # NOTE: all_neighbors: texts x 10 nearest other texts, human transcripts and model scrambles together.
    all_neighbors = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="cosine").fit(embeddings).kneighbors(
        embeddings, return_distance=False)[:, 1:]
    scramble_share = (groups[all_neighbors] == "Model_scrambles").mean(axis=1)
    result["scramble_neighbors"] = {"of_scrambles": float(scramble_share[~human].mean()),
                                    "of_human_transcripts": float(scramble_share[human].mean()),
                                    "scramble_share_of_all": float((~human).mean())}
    for group in GROUPS:
        entry = {"n": int((labels_of == group).sum()), "control_neighbors": float(control_share[labels_of == group].mean())}
        if group != "Control":
            pair = np.isin(labels_of, ["Control", group])
            labels = (labels_of[pair] == group).astype(int)
            splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0)
            folds = splitter.split(np.zeros(len(labels)), labels, speakers[pair])
            entry["auc_vs_control"] = float(np.mean([probe_auc(x[pair][train], labels[train], x[pair][test], labels[test])
                                                     for train, test in folds]))
        result[group] = entry
    return result


def tsne(embeddings: np.ndarray) -> np.ndarray:
    # NOTE: t-SNE keeps neighborhoods, not the distances between far-apart groups, so the picture shows
    # which transcripts lie among which; overlap() measures the same thing in the full embedding.
    return TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", random_state=0).fit_transform(embeddings)


def draw(panels: dict, path: Path, with_scrambles: bool):
    fig, axes = plt.subplots(1, 2, figsize=(paper_style.PAGE_WIDTH, 2.75))
    for ax, (task, (points, groups)) in zip(axes, panels.items()):
        classes = np.array([PLOT_CLASS[group] for group in groups])
        # NOTE: the points are drawn in a random order, so that no group systematically hides
        # another; the model scrambles go on top.
        order = np.random.default_rng(0).permutation(len(points))
        order = np.concatenate([order[classes[order] != "model_scrambles"], order[classes[order] == "model_scrambles"]])
        ax.scatter(points[order, 0], points[order, 1], s=5, linewidths=0, alpha=0.8,
                   c=[paper_style.CLASS_COLORS[name] for name in classes[order]])
        ax.set_title(TASKS[task])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(paper_style.GRID)
    names = ["control", "conduction/anomic", "broca", "wernicke"] + (["model_scrambles"] if with_scrambles else [])
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=paper_style.CLASS_COLORS[name])
               for name in names]
    fig.subplots_adjust(top=0.82)
    fig.legend(handles, [paper_style.CLASS_NAMES[name] for name in names], loc="upper center", ncol=len(names),
               bbox_to_anchor=(0.5, 1.0), handletextpad=0.2, columnspacing=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main():
    results, panels = {}, {False: {}, True: {}}
    for task in TASKS:
        data = load_task(task)
        human = data["groups"] != "Model_scrambles"
        results[task] = overlap(data["embeddings"], data["groups"], data["speakers"])
        panels[False][task] = (tsne(data["embeddings"][human]), data["groups"][human])
        panels[True][task] = (tsne(data["embeddings"]), data["groups"])
        print(f"{task}: control share of all transcripts {results[task]['control_share_of_all']:.2f}; scramble "
              f"neighbors of scrambles {results[task]['scramble_neighbors']['of_scrambles']:.2f}, of human transcripts "
              f"{results[task]['scramble_neighbors']['of_human_transcripts']:.3f}")
        for group in GROUPS:
            entry = results[task][group]
            print(f"  {group:10s} n={entry['n']:4d}  control neighbors {entry['control_neighbors']:.2f}"
                  + (f"  probe AUC vs control {entry['auc_vs_control']:.3f}" if "auc_vs_control" in entry else ""))
    out_path = RESULTS / "tsne" / "overlap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    draw(panels[False], RESULTS / "figures" / "tsne_transcripts", with_scrambles=False)
    draw(panels[True], RESULTS / "figures" / "tsne_transcripts_scrambles", with_scrambles=True)
    print(f"wrote {out_path} and the two figures in {RESULTS / 'figures'}")


if __name__ == "__main__":
    main()
