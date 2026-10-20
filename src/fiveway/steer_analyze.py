"""Is aphasia linearly represented in a model's residual stream? Probes, vector geometry, PCA and severity.

The CPU half of the steering study's activation side, with no model and no
generation. It reads what steer_collect.py stored: each human transcript's
mean residual stream after every block (runs/fiveway/steering/<model>/
acts_<task>.npz and meta_<task>.json) and the steering vectors
(results/fiveway/steering/<model>/vectors_<task>.npz). Then it asks, per task
and layer:

  1. Is the group linearly decodable? ROC AUC of a linear probe (L2 logistic
     regression on standardized features), 5-fold cross-validated with each
     speaker's transcripts in one fold, for
         A  aphasic (Anomic, Conduction, Broca, Wernicke) vs Control
         B  Broca vs every other transcript
         C  Wernicke vs every other transcript
  2. Is it the same direction in both tasks? The probe fitted on all of one
     task's transcripts, tested on the other task's.
  3. How do the directions relate? Cosine similarity between A, B, C and AC,
     and between a task's vector and the other task's at the same layer.
  4. Is aphasia a main axis of variation? PCA of the per-transcript means: the
     variance fraction of PC1 to PC3, and |cosine| of PC1 with A.
  5. Is it graded by severity? Spearman correlation between each aphasic
     transcript's projection on the unit A vector and its aphasia quotient.
     A lower AQ is more severe, so a direction that tracks severity gives a
     negative correlation.
  6. How large is the step from Control to aphasia? |A| / resid_norm.

Every statistic counts each text once (the transcripts steer_collect.py
marked "kept"), as src/steering/check_duplicates.py did. For the LayerNorm
models (GPT-2, Pythia) every residual mean is centred over d_model first, as
steer_collect.py centred them for the vectors (centre_for_layernorm there):
the component along (1, 1, ..., 1) is invisible to every layer of those models.

Output: results/fiveway/steering/<model>/analysis.json, numbers only (no
transcript text, no file names), and a table printed to the log. A rerun
keeps the sections already written if they came from the same inputs and the
same version of this script.

Usage (CPU only, a few minutes):
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-analyze-p1b 32G fiveway/steer_analyze.py --model pythia-1b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402

GROUPS = ["Control", "Anomic", "Conduction", "Broca", "Wernicke"]
N_FOLDS = 5
PROBE_C = 0.01      # inverse strength of the probes' L2 penalty, as in the earlier study
N_COMPONENTS = 3    # principal components whose variance fraction is reported

# The probes' positive class; every other transcript is the negative class.
PROBE_TARGETS = {
    "A": ["Anomic", "Conduction", "Broca", "Wernicke"],   # aphasic vs Control
    "B": ["Broca"],                                       # Broca vs everyone else
    "C": ["Wernicke"],                                    # Wernicke vs everyone else
}
VECTOR_PAIRS = [("A", "B"), ("A", "C"), ("A", "AC"), ("B", "C"), ("B", "AC"), ("C", "AC")]

# The families whose every block reads the residual stream through a LayerNorm (as in steer_collect.py).
LAYERNORM_FAMILIES = ("gpt2", "neox")


def speaker_id(row: dict) -> str:
    """An approximate speaker id: the file name without its session suffix, within its corpus.

    NOTE: as in src/steering/check_duplicates.py: AphasiaBank
    names a session <speaker><session>, as in <speaker>-<session> or <speaker><session letter>,
    and some corpora add a PWA- prefix. Stripping both puts a speaker's
    sessions, and the sandwich corpus's second listings, under one id.
    Two steps go beyond check_duplicates.py, which split a few speakers'
    sessions between folds. A handful of files carry a copy marker that hid
    their session suffix ("<speaker>-<session> 2.cha_control", "<speaker><letter> 2.gem.cex"), so it is
    dropped first. And one sub-corpus numbers its speakers as a related one does
    (two speakers are in both, with the same text), so a corpus name's "-2" (or "-bi")
    suffix is dropped too. Should two people end up under one id, they only
    share a fold, which costs nothing; one person under two ids leaks.
    """
    base = row["File_DB"].split(".")[0].replace("PWA-", "")
    base = re.sub(r" \d+$", "", base)
    base = re.sub(r"(-\d+|-LARC|[a-z])$", "", base)
    corpus = row["Corpus"].split("-")[0]
    return f"{corpus}|{base}"


def unit(vectors: np.ndarray) -> np.ndarray:
    """Each row scaled to length 1."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def cosine(first: np.ndarray, second: np.ndarray) -> list:
    """Cosine similarity of two (n_layers, d_model) arrays, layer by layer."""
    return (unit(first) * unit(second)).sum(axis=-1).tolist()


def probe_auc(train_x, train_y, test_x, test_y) -> float:
    """ROC AUC on the test set of a logistic regression fitted on the standardized training set."""
    # NOTE: standardize with the training statistics only,
    # so nothing about the test transcripts leaks into the probe.
    # NOTE: the 1e-6 keeps a dimension that is constant
    # over the training transcripts from dividing by zero.
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    probe = LogisticRegression(C=PROBE_C, max_iter=2000)
    probe.fit((train_x - mean) / std, train_y)
    return float(roc_auc_score(test_y, probe.decision_function((test_x - mean) / std)))


def load_task(runs_dir: Path, results_dir: Path, task: str, centre: bool) -> dict:
    """A task's kept transcripts (residual means in float32, groups, speakers, AQs) and its steering vectors.

    centre: whether the model reads its residual stream through LayerNorm; the
    means are then centred over d_model, as steer_collect.py centred them.
    """
    meta = json.loads((runs_dir / f"meta_{task}.json").read_text())
    kept = np.array([row["kept"] for row in meta])
    with np.load(runs_dir / f"acts_{task}.npz") as stored:
        resid = stored["resid"][kept].astype(np.float32)      # NOTE: (n_kept, n_layers, d_model)
    if centre:
        # NOTE: the mean over d_model, (n_kept, n_layers, 1), broadcasts back over d_model.
        resid = resid - resid.mean(axis=-1, keepdims=True)
    with np.load(results_dir / f"vectors_{task}.npz") as stored:
        vectors = {key: stored[key] for key in stored.files}
    rows = [row for row in meta if row["kept"]]
    groups = np.array([row["Group"] for row in rows])
    # The vectors must come from exactly these activations: the same transcripts, centred the same way.
    # (A vectors file from before steer_collect.py centred LayerNorm models fails here; rerun steer_collect.py.)
    # NOTE: recomputed_a and vectors["A"]: (n_layers, d_model); the norms over d_model compare them layer by layer.
    recomputed_a = resid[groups != "Control"].mean(axis=0) - resid[groups == "Control"].mean(axis=0)
    mismatch = np.linalg.norm(recomputed_a - vectors["A"], axis=-1) / np.linalg.norm(vectors["A"], axis=-1)
    assert mismatch.max() < 1e-5, f"vectors_{task}.npz does not match acts_{task}.npz; rerun steer_collect.py"
    return {"resid": resid, "groups": groups, "vectors": vectors, "n_listed": len(meta),
            "speakers": np.array([speaker_id(row) for row in rows]),
            "aq": np.array([np.nan if row["aq"] is None else row["aq"] for row in rows])}


def analyze_task(data: dict, log) -> dict:
    """Probes, vector cosines, PCA, severity and the size of A for one task, per layer."""
    resid, groups, speakers, aq, vectors = data["resid"], data["groups"], data["speakers"], data["aq"], data["vectors"]
    n_layers = resid.shape[1]
    unit_a = unit(vectors["A"])                              # NOTE: (n_layers, d_model)
    section = {"counts": {"listed": data["n_listed"], "counted": len(groups), "speakers": len(set(speakers)),
                          "per_group": {group: int((groups == group).sum()) for group in GROUPS}}}

    # 1. Linear probes, tested on speakers they were not fitted on.
    section["probe_auc"] = {}
    for target, members in PROBE_TARGETS.items():
        labels = np.isin(groups, members).astype(int)
        # NOTE: StratifiedGroupKFold is GroupKFold that also
        # balances the classes. Every speaker's transcripts stay in one fold, so
        # each probe is tested on speakers it has not seen, and every test fold
        # gets its share of the rare positives (a few dozen Wernicke
        # transcripts), which plain GroupKFold does not ensure. Only the labels
        # and the speakers decide the folds, so all layers share them.
        splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=0)
        folds = list(splitter.split(np.zeros(len(labels)), labels, speakers))
        # NOTE: resid[train, layer]: one layer's means of the training
        # transcripts, (n_train, d_model).
        section["probe_auc"][target] = [
            float(np.mean([probe_auc(resid[train, layer], labels[train], resid[test, layer], labels[test])
                           for train, test in folds]))
            for layer in range(n_layers)]
        log(f"  probe {target} ({labels.sum()} positives of {len(labels)}): AUC by layer "
            + " ".join(f"{value:.2f}" for value in section["probe_auc"][target]))

    # 3. How the directions relate within the task (the cross-task cosines are in analyze_cross_task).
    # NOTE: A, B, C and AC all subtract the same Control
    # mean, and A's aphasic mean contains the Broca, Wernicke, Anomic and
    # Conduction transcripts. So their cosines are positive even where the
    # groups do not differ at all: on synthetic data with no group signal,
    # cos(A, B) came out near 0.8 with 60 transcripts per group. Read these
    # cosines against the probes, not on their own.
    section["vector_cosine"] = {f"{first}~{second}": cosine(vectors[first], vectors[second])
                                for first, second in VECTOR_PAIRS}

    # 4. Principal components of the per-transcript means.
    variance_fraction, pc1_abs_cosine = [], []
    for layer in range(n_layers):
        centered = resid[:, layer] - resid[:, layer].mean(axis=0)
        # NOTE: the right singular vectors of the centered data
        # matrix are its principal components, and the squared singular values
        # are proportional to the variance along each.
        # NOTE: LAPACK's default SVD driver (gesdd) failed
        # to converge on one of the cluster's nodes in the earlier study;
        # gesvd is slower but does not have that problem.
        _, singular_values, components = scipy.linalg.svd(centered, full_matrices=False, lapack_driver="gesvd")
        variance = singular_values ** 2
        variance_fraction.append((variance[:N_COMPONENTS] / variance.sum()).tolist())
        # NOTE: components: (n_components, d_model); row 0 is PC1, a unit vector, as is unit_a[layer].
        # A principal component's sign is arbitrary, hence the absolute value.
        pc1_abs_cosine.append(abs(float(components[0] @ unit_a[layer])))
    section["pca"] = {"variance_fraction": variance_fraction, "pc1_abs_cosine_A": pc1_abs_cosine}

    # 5. Severity: the coordinate along A against the aphasia quotient.
    has_aq = (groups != "Control") & ~np.isnan(aq)
    # NOTE: einsum "nld,ld->nl" dots each aphasic transcript's
    # (n_layers, d_model) means with the unit A vector of the same layer:
    # (n_with_aq, n_layers) coordinates along A.
    projection = np.einsum("nld,ld->nl", resid[has_aq], unit_a)
    section["severity"] = {"n_with_aq": int(has_aq.sum()),
                           "spearman_aq": [float(spearmanr(projection[:, layer], aq[has_aq]).statistic)
                                           for layer in range(n_layers)]}

    # 6. The size of the step from Control to aphasia, relative to the residual stream.
    section["A_norm_over_resid_norm"] = (np.linalg.norm(vectors["A"], axis=-1) / vectors["resid_norm"]).tolist()
    return section


def analyze_cross_task(data: dict, log) -> dict:
    """Probe transfer from one task to the other, and cosines between the two tasks' vectors, per layer."""
    first, second = TASKS
    section = {"vector_cosine": {name: cosine(data[first]["vectors"][name], data[second]["vectors"][name])
                                 for name in ("A", "B", "C", "AC")},
               "transfer_auc": {}}
    for train_task, test_task in ((first, second), (second, first)):
        train, test = data[train_task], data[test_task]
        # NOTE: many AphasiaBank speakers told both
        # narratives, so a transfer probe meets speakers it was fitted on. What
        # it measures is whether the two tasks share the direction, not whether
        # the probe generalizes to new speakers.
        entry = {}
        for target, members in PROBE_TARGETS.items():
            train_labels = np.isin(train["groups"], members).astype(int)
            test_labels = np.isin(test["groups"], members).astype(int)
            entry[target] = [probe_auc(train["resid"][:, layer], train_labels, test["resid"][:, layer], test_labels)
                             for layer in range(train["resid"].shape[1])]
        section["transfer_auc"][f"{train_task}_to_{test_task}"] = entry
        log(f"  {train_task} -> {test_task}: transfer AUC of A by layer "
            + " ".join(f"{value:.2f}" for value in entry["A"]))
    return section


def print_table(report: dict) -> None:
    """One line per layer and task, with the headline number of every section."""
    first, second = TASKS
    for task, other in ((first, second), (second, first)):
        section, cross = report[task], report["cross_task"]
        counts, transfer = section["counts"], cross["transfer_auc"][f"{task}_to_{other}"]
        print(f"\n{task}: {counts['counted']} of {counts['listed']} transcripts counted, "
              f"{counts['speakers']} speakers, {counts['per_group']}, "
              f"{section['severity']['n_with_aq']} aphasic with an AQ\n"
              f"  auc_*: probe AUC (5 folds by speaker); to_*: probe fitted here, tested on {other}; "
              f"X~Y: cosine of the vectors; A~A': cosine with {other}'s A;\n"
              f"  pc1-3: variance fractions; pc1~A: |cosine| of PC1 with A; rho_AQ: Spearman of the "
              f"projection on A with AQ; |A|/|h|: |A| / resid_norm", flush=True)
        print("layer auc_A auc_B auc_C  to_A  to_B  to_C    A~B    A~C    B~C  A~A'    pc1   pc2   pc3  pc1~A "
              "rho_AQ |A|/|h|", flush=True)
        cosines, pca = section["vector_cosine"], section["pca"]
        for layer in range(len(section["probe_auc"]["A"])):
            print(f"{layer:5d} " + " ".join(f"{section['probe_auc'][target][layer]:.3f}" for target in "ABC")
                  + " " + " ".join(f"{transfer[target][layer]:.3f}" for target in "ABC")
                  + " " + " ".join(f"{cosines[pair][layer]:+.3f}" for pair in ("A~B", "A~C", "B~C"))
                  + f" {cross['vector_cosine']['A'][layer]:+.3f}  "
                  + " ".join(f"{value:.3f}" for value in pca["variance_fraction"][layer])
                  + f"  {pca['pc1_abs_cosine_A'][layer]:.3f} {section['severity']['spearman_aq'][layer]:+.3f}"
                  + f"  {section['A_norm_over_resid_norm'][layer]:.3f}", flush=True)


def write_json(path: Path, payload) -> None:
    """Write through a temporary file, so a job killed mid-write never leaves half a file behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--limit", type=int, help="analyze the smoke test of steer_collect.py --limit LIMIT")
    args = parser.parse_args()

    # The folders of steer_collect.py; a --limit smoke test keeps everything under runs/.
    name = args.model if args.limit is None else f"{args.model}-limit{args.limit}"
    runs_dir = RUNS / "fiveway" / "steering" / name
    results_dir = RESULTS / "steering" / name if args.limit is None else runs_dir / "results"
    out_path = results_dir / "analysis.json"
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    # NOTE: a requeued job keeps the sections already
    # written, but only sections computed from the same inputs by the same
    # code: the fingerprint hashes this script and both tasks' meta and
    # vectors files (the vectors change whenever the activations do). New
    # activations, rebuilt vectors or an edited analysis therefore start from
    # scratch instead of mixing old and new numbers in one file.
    digest = hashlib.sha256(Path(__file__).read_bytes())
    for task in TASKS:
        digest.update((runs_dir / f"meta_{task}.json").read_bytes())
        digest.update((results_dir / f"vectors_{task}.npz").read_bytes())
    fingerprint = digest.hexdigest()[:16]
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    if report.get("fingerprint") != fingerprint:
        report = {"model": args.model, "limit": args.limit, "fingerprint": fingerprint}
    centre = MODELS[args.model].family in LAYERNORM_FAMILIES
    data = {task: load_task(runs_dir, results_dir, task, centre) for task in TASKS}
    for task in TASKS:
        if task in report:
            log(f"{task}: already in {out_path}")
            continue
        log(f"{task}: {len(data[task]['groups'])} kept transcripts, {data[task]['resid'].shape[1]} layers")
        report[task] = analyze_task(data[task], log)
        write_json(out_path, report)
    if "cross_task" not in report:
        log("cross-task transfer and cosines")
        report["cross_task"] = analyze_cross_task(data, log)
        write_json(out_path, report)
    print_table(report)
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
