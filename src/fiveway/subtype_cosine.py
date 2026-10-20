"""How alike are the Broca and Wernicke steering directions? Their cosine per layer, against shuffled labels.

The steering vector of a subtype is the mean residual stream of its patients' transcripts minus that of the
control transcripts (steer_collect.py: B for Broca, C for Wernicke). If the two subtypes were separate in the
model, B and C would point in different directions; if they shared most of their deviation from control
speech, cos(B, C) would be close to 1. For each model, task and layer this script reports:

    observed    cos(B, C) with the true Broca and Wernicke labels
    null        cos(B, C) with the Broca and Wernicke labels shuffled among the patients of the two subtypes
                (N_PERMUTATIONS times, keeping the group sizes), the cosine of two random halves of one pool

The shuffled labels give the cosine that two groups drawn from the same patients would have. An observed
cosine below the null's 2.5th percentile means the subtypes differ more than random groups of patients do.
The same comparison is made for B and C with the shared aphasia direction A projected out of each
(B_specific, C_specific in steer_collect.py), the part of each direction that is specific to its subtype.

It first rebuilds the vectors from the stored residual means and checks them against the saved ones.

Output: results/fiveway/steering/subtype_cosine.json, and a printed summary per model and task.

Usage (CPU, ~1 hour with 8 cores; reads runs/fiveway/steering/<model>/acts_<task>.npz):
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh subtype-cosine 48G fiveway/subtype_cosine.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.steer_collect import LAYERNORM_FAMILIES, build_vectors, centre_for_layernorm  # noqa: E402

N_PERMUTATIONS = 200
STUDY_MODELS = ["gpt2-xl", "llama2-7b-chat", "llama2-13b-chat", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
                "pythia-6.9b", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]


def unit(vectors: np.ndarray) -> np.ndarray:
    # NOTE: vectors: (..., d_model); the norm over d_model, kept as (..., 1), broadcasts back.
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    results = {}
    for model in STUDY_MODELS:
        centre = MODELS[model].family in LAYERNORM_FAMILIES
        for task in ("cinderella", "sandwich"):
            with np.load(RUNS / "fiveway" / "steering" / model / f"acts_{task}.npz") as stored:
                resid = stored["resid"]                  # NOTE: (n_transcripts, n_layers, d_model)
            meta = json.loads((RUNS / "fiveway" / "steering" / model / f"meta_{task}.json").read_text())
            groups = np.array([row["Group"] for row in meta])
            kept = np.array([row["kept"] for row in meta])
            vectors = build_vectors(resid, groups, kept, centre)
            saved = np.load(RESULTS / "steering" / model / f"vectors_{task}.npz")
            for name in ("A", "B", "C", "B_specific", "C_specific"):
                assert np.allclose(vectors[name], saved[name], rtol=1e-5, atol=1e-6), (model, task, name)

            resid = resid[kept].astype(np.float32)
            if centre:
                resid = centre_for_layernorm(resid)
            groups = groups[kept]
            control_mean = resid[groups == "Control"].mean(axis=0)
            unit_a = unit(vectors["A"])

            def cosines(broca, wernicke):
                """cos(B, C) and cos(B_specific, C_specific) per layer, as (n_layers,) arrays."""
                # NOTE: (v * unit_a).sum(-1, keepdims=True) is each layer's coordinate along A,
                # (n_layers, 1); times unit_a (n_layers, d_model) it is the component along A.
                specific = lambda v: v - (v * unit_a).sum(-1, keepdims=True) * unit_a
                return ((unit(broca) * unit(wernicke)).sum(-1),
                        (unit(specific(broca)) * unit(specific(wernicke))).sum(-1))

            observed, observed_specific = cosines(vectors["B"], vectors["C"])

            # The null: Broca and Wernicke labels shuffled among the patients of the two subtypes.
            pool = resid[np.isin(groups, ["Broca", "Wernicke"])]       # NOTE: (n_pool, n_layers, d_model)
            n_broca, n_pool = int((groups == "Broca").sum()), len(pool)
            # NOTE: the pool is flattened once, and each shuffle sums only the rows
            # that become Broca; the Wernicke sum is the total minus it.
            flat = pool.reshape(n_pool, -1)
            total = flat.sum(axis=0)
            null, null_specific = [], []
            for _ in range(N_PERMUTATIONS):
                is_broca = np.zeros(n_pool, bool)
                is_broca[rng.choice(n_pool, n_broca, replace=False)] = True
                broca_sum = flat[is_broca].sum(axis=0)
                broca = (broca_sum / n_broca).reshape(pool.shape[1:]) - control_mean
                wernicke = ((total - broca_sum) / (n_pool - n_broca)).reshape(pool.shape[1:]) - control_mean
                plain, specific = cosines(broca, wernicke)
                null.append(plain)
                null_specific.append(specific)
            null, null_specific = np.array(null), np.array(null_specific)   # NOTE: (N_PERMUTATIONS, n_layers)

            row = {"n_broca": n_broca, "n_wernicke": n_pool - n_broca, "n_control": int((groups == "Control").sum()),
                   "observed": observed.tolist(), "observed_specific": observed_specific.tolist(),
                   "null_mean": null.mean(0).tolist(), "null_low": np.percentile(null, 2.5, axis=0).tolist(),
                   "null_specific_mean": null_specific.mean(0).tolist(),
                   "null_specific_low": np.percentile(null_specific, 2.5, axis=0).tolist()}
            results[f"{model}|{task}"] = row
            below = int((observed < np.array(row["null_low"])).sum())
            print(f"{model:16s} {task:10s} Broca {n_broca} Wernicke {n_pool - n_broca}: cos(B, C) median "
                  f"{np.median(observed):.2f} [{observed.min():.2f}, {observed.max():.2f}], shuffled "
                  f"{np.median(row['null_mean']):.2f}; below the shuffled 2.5% in {below}/{len(observed)} layers | "
                  f"specific parts {np.median(observed_specific):.2f}, shuffled {np.median(row['null_specific_mean']):.2f}",
                  flush=True)
    out_path = RESULTS / "steering" / "subtype_cosine.json"
    out_path.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
