"""Steering against the grown-heads knockouts: does pushing back toward Control undo them?

steer_generate.py's rescue mode asked this of search.py's objective-A sets.
Here the knockout sets are the grown-heads sets (em_attention.py's space 2b,
or grown_heads.py's classifier sets for models that have no em_attention.py
results), for every objective whose set works, each with its own direction
from steer_collect.py:

    A  aphasic transcripts minus Control's      B  Broca minus Control      C  Wernicke minus Control

For one objective O and its knockout set K, every answer is generated
greedily (rep1.25) and scored by the five-way classifier:

    references  the intact model; the intact model with O projected out after every
                block; K; K with O projected out; K with each random direction projected out
    steered     K with O, or a random direction, added after one of 8 layers spread over
                the depth, at negative strengths (RESCUE_STRENGTHS, toward Control)

A push undoes the knockout when the answer's objective O falls to 0.5 or
below. The random directions (as long as A at every layer, as in
steer_generate.py) are the controls: O undoes the knockout in a meaningful
sense only where they do not.

Outputs:
    RESULTS/steering_grown/<model>/rescue_<task>.json    every setting with its answer and scores
    RESULTS/steering_grown/<model>/summary_<task>.json   per objective: the references' values, whether
                                                         removing O or a random direction undoes K, and
                                                         the share of steered settings that undo it
A rerun generates only the settings its output file does not list yet.

Usage:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh \\
        grown-rescue-p1b-cind 48G fiveway/grown_rescue.py --model pythia-1b --task cinderella
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import SUCCESS_THRESHOLD, FiveWayScorer, works_finally  # noqa: E402
from fiveway.steer_collect import LAYERNORM_FAMILIES  # noqa: E402
from fiveway.steer_generate import (N_RANDOM_DIRECTIONS, RESCUE_STRENGTHS, Setting, fingerprint,  # noqa: E402
                                    generate_and_score, load_vectors, random_directions, setting_key,
                                    spread_layers, steering_setting, write_json)

DECODING = "rep1.25"


def grown_sets(model: str, task: str) -> dict:
    """{objective: {"heads", "value", "source"}} for every objective whose grown-heads set works."""
    grown_path = RESULTS / "grown_heads" / f"{model}_{task}_classifier.json"
    em_path = RESULTS / "em_attention" / f"{model}_{task}.json"
    if grown_path.exists():
        rows, source = json.loads(grown_path.read_text())["reported"], str(grown_path.relative_to(RESULTS))
    else:
        rows = {name: spaces["2b_heads_grown"] for name, spaces in json.loads(em_path.read_text())["reported"].items()}
        source = str(em_path.relative_to(RESULTS))
    # NOTE: a set works by the final-answer rule (scorer.works_finally), which also takes a target
    # that is the classifier's top class below 0.5.
    return {name: {"heads": row["heads"], "value": row["value"], "source": source}
            for name, row in rows.items() if works_finally(name, row["value"], row["probabilities"])}


def build_settings(sets: dict, vectors: dict, resid_norm: np.ndarray, layers: list, strengths: list) -> list:
    """The references and steered knockouts of every objective's set."""
    random_names = [f"random{seed}" for seed in range(N_RANDOM_DIRECTIONS)]
    settings = []
    for name, found in sets.items():
        knockout = Intervention.knockout(heads=[tuple(head) for head in found["heads"]])
        references = [("intact", Intervention()), (f"intact, {name} removed", Intervention(remove=(name,))),
                      ("knockout", knockout)]
        references += [(f"knockout, {direction} removed", dataclasses.replace(knockout, remove=(direction,)))
                       for direction in [name] + random_names]
        for label, intervention in references:
            settings.append(Setting({"set_objective": name, "intervention": label, "vector": None, "objective": name,
                                     "layer": None, "strength": None}, intervention))
        for direction in [name] + random_names:
            for layer in layers:
                for strength in strengths:
                    settings.append(steering_setting(vectors, resid_norm, direction, name, layer, strength, base=knockout,
                                                     intervention="knockout + steer", set_objective=name))
    return settings


def summarize(entries: list, sets: dict) -> dict:
    """Per objective: the references' values, removal against random removal, and the share of pushes that undo."""

    def value(entry):
        return None if entry["objectives"] is None else entry["objectives"][entry["objective"]]

    def undoes(entry):
        # NOTE: an unscored answer (too short) is no rescue.
        return value(entry) is not None and value(entry) <= SUCCESS_THRESHOLD

    summary = {"threshold": SUCCESS_THRESHOLD, "objectives": {}}
    for name, found in sets.items():
        own = [entry for entry in entries if entry["set_objective"] == name]
        references = {entry["intervention"]: value(entry) for entry in own if entry["layer"] is None}
        steered = {}
        for entry in own:
            if entry["intervention"] == "knockout + steer":
                steered.setdefault(entry["vector"], []).append(entry)
        random_names = sorted(vector for vector in steered if vector.startswith("random"))
        summary["objectives"][name] = {
            "n_heads": len(found["heads"]), "set_value": found["value"], "source": found["source"],
            "references": references,
            "removing_own_direction_undoes": None if references.get(f"knockout, {name} removed") is None
            else references[f"knockout, {name} removed"] <= SUCCESS_THRESHOLD,
            "random_removals_undoing": sum(references.get(f"knockout, {vector} removed") is not None
                                           and references[f"knockout, {vector} removed"] <= SUCCESS_THRESHOLD
                                           for vector in random_names),
            "share_of_pushes_undoing": {vector: float(np.mean([undoes(entry) for entry in vector_entries]))
                                        for vector, vector_entries in steered.items()},
            # NOTE: the strengths are negative, so max picks the
            # weakest push toward Control that undoes the knockout at that layer.
            "weakest_strength_undoing": {
                vector: {str(layer): max([entry["strength"] for entry in vector_entries
                                          if entry["layer"] == layer and undoes(entry)], default=None)
                         for layer in sorted({entry["layer"] for entry in vector_entries})}
                for vector, vector_entries in steered.items()}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--strengths", type=float, nargs="+", default=RESCUE_STRENGTHS)
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    out_dir = RESULTS / "steering_grown" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    task_vectors = load_vectors(RESULTS / "steering" / args.model / f"vectors_{args.task}.npz")
    resid_norm = task_vectors["resid_norm"]                       # NOTE: (n_layers,)
    vectors = {name: task_vectors[name] for name in ("A", "B", "C")}
    random = random_directions(task_vectors["A"], centre=MODELS[args.model].family in LAYERNORM_FAMILIES)
    vectors.update(random)
    layers = spread_layers(task_vectors["A"].shape[0])
    sets = grown_sets(args.model, args.task)
    inputs = {"vectors": fingerprint(task_vectors), "random_directions": fingerprint(random), "sets": sets,
              "layers": layers, "strengths": args.strengths}
    log(f"{args.model} {args.task}: working grown sets for {sorted(sets)} "
        f"({', '.join(f'{name}: {len(found['heads'])} heads' for name, found in sets.items())}); layers {layers}")

    path = out_dir / f"rescue_{args.task}.json"
    output = {"model": args.model, "task": args.task, "decoding": DECODING, "inputs": inputs, "entries": []}
    if path.exists():
        output = json.loads(path.read_text())
        old = output["inputs"]
        # NOTE: a set that has started to count since (scorer.works_finally) joins the sets the
        # file was made for: their entries are kept as they are, and only the new set's settings are generated.
        extends = ({key: value for key, value in old.items() if key != "sets"}
                   == {key: value for key, value in inputs.items() if key != "sets"}
                   and all(inputs["sets"].get(name) == found for name, found in old["sets"].items()))
        if old != inputs and not extends:
            raise SystemExit(f"{path} was made from other inputs; move it away to start over")
        output["inputs"] = inputs
    settings = build_settings(sets, vectors, resid_norm, layers, args.strengths)
    done = {entry["key"] for entry in output["entries"]}
    missing = [setting for setting in settings if setting_key(setting.fields) not in done]
    log(f"{len(missing)} of {len(settings)} settings to generate -> {path}")
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = Engine(args.model, device)
        scorer = FiveWayScorer(device)
        for name, vector in vectors.items():
            assert vector.shape == (engine.geometry["n_layers"], engine.geometry["d_model"]), (name, vector.shape)
            engine.vectors[name] = torch.tensor(vector, dtype=torch.float32, device=device)
        for begin in range(0, len(missing), engine.batch_size):
            batch = missing[begin:begin + engine.batch_size]
            records = generate_and_score(engine, scorer, args.task, [setting.intervention for setting in batch], None)
            for setting, record in zip(batch, records):
                output["entries"].append({"key": setting_key(setting.fields), **setting.fields, "alpha": setting.alpha,
                                          **record})
            write_json(path, output)
            log(f"  {len(output['entries'])} of {len(settings)} settings in {path.name}")
    summary_path = out_dir / f"summary_{args.task}.json"
    write_json(summary_path, {"model": args.model, "task": args.task, "inputs": inputs,
                              **summarize(output["entries"], sets)})
    log(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
