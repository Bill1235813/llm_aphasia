"""Steered answers to seven prompts, for annotation: does steering make the model sound aphasic?

For one generator, every answer of this grid:

    7 prompts   the Cinderella and sandwich story requests (prompts.py) and the
                five TAB connected-text prompts (benchmarks.TAB_ITEMS)
    2 vectors   the model's steering vectors from its Cinderella or its sandwich
                transcripts (steer_collect.py)
    3 targets   A (aphasia), B (Broca), C (Wernicke): that vector added to the
                residual stream after one block

plus the unsteered answer to each prompt, for reference. Every answer is
scored by the five-way classifier: P(aphasia) = P(Conduction/Anomic) +
P(Broca) + P(Wernicke) for A, P(Broca) for B, P(Wernicke) for C.

The layer and strength of each (vector source, target) come from the steering
sweep on the vector's own task (steer_generate.py, sweep_<task>.json): the
smallest strength at which the target's probability went above 0.5, at the
layer where that happened (the higher probability on a tie). When no setting
went above 0.5 (Wernicke in GPT-2 XL and Llama-2), the setting with the
highest probability. The same setting then steers all seven prompts.

NOTE: the setting is chosen on one prompt (the vector's own
task) by the classifier, and then applied unchanged to six prompts it never
saw. The classifier's verdicts on those six are therefore not tuned to it,
which matters because annotation_csv.py compares it with an LLM judge.

Answers are made as in the steering runs: rep1.25 decoding, a base model's
answer cut at the first question it writes itself; 180 new tokens for the
stories and 120 for connected text, as in search.py and benchmarks.py. The
Cinderella yes/no question is not asked.

Output: results/fiveway/annotation/generations_<model>.json, rewritten after
every prompt; a rerun skips the prompts it already holds.

Usage (CPU-only job: 7 batches of 7 answers per model):
    GPUS=0 CPUS=32 bash slurm/fiveway_submit.sh annotate-gen-p1b 32G fiveway/steer_annotate.py --model pythia-1b
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
from fiveway.benchmarks import CONNECTED_TEXT_TOKENS, TAB_ITEMS, tab_prompt  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.prompts import STORY_TOKENS, cut_at_new_question, long_enough, task_prompt_ids  # noqa: E402
from fiveway.scorer import CLASSES, SUCCESS_THRESHOLD, FiveWayScorer, objective_values  # noqa: E402
from fiveway.steer_generate import fingerprint, load_vectors  # noqa: E402
from interp.generate import LLAMA2_STORY_USER, SANDWICH_USER  # noqa: E402

DECODING = "rep1.25"
VECTOR_SOURCES = {"cind": "cinderella", "sand": "sandwich"}
TARGETS = {"A": "aphasia", "B": "broca", "C": "wernicke"}
CONNECTED_TEXT_NAMES = ["trip", "childhood", "job", "home", "meal"]   # TAB's five prompts, in order
MODEL_NAMES = {"gpt2-xl": "gpt2_xl", "llama2-7b-chat": "llama_7b", "llama2-13b-chat": "llama_13b",
               "pythia-1b": "pythia_1b", "pythia-1.4b": "pythia_1.4b", "pythia-2.8b": "pythia_2.8b",
               "pythia-6.9b": "pythia_6.9b"}


def chosen_setting(model: str, task: str, target: str, vectors: dict) -> dict:
    """The sweep's weakest setting that made vector ``target`` work on its own task (or its best one)."""
    sweep = json.loads((RESULTS / "steering" / model / f"sweep_{task}.json").read_text())
    # NOTE: the sweep must come from the vectors steered
    # with here; after a rebuild of the vectors its strengths would mean
    # something else.
    assert sweep["inputs"]["vectors"] == fingerprint(vectors), f"{model} {task}: sweep made from other vectors"
    entries = [entry for entry in sweep["entries"] if entry["vector"] == target and entry["strength"] > 0
               and entry["objectives"] is not None]
    working = [entry for entry in entries if entry["objectives"][target] > SUCCESS_THRESHOLD]
    if working:
        best = min(working, key=lambda entry: (entry["strength"], -entry["objectives"][target]))
    else:
        best = max(entries, key=lambda entry: entry["objectives"][target])
    return {"layer": best["layer"], "strength": best["strength"], "works_on_own_task": bool(working),
            "value_on_own_task": best["objectives"][target]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    # The steering settings: one per vector source and target.
    settings = []
    for source, task in VECTOR_SOURCES.items():
        vectors = load_vectors(RESULTS / "steering" / args.model / f"vectors_{task}.npz")
        for target, target_name in TARGETS.items():
            setting = chosen_setting(args.model, task, target, vectors)
            layer, strength = setting["layer"], setting["strength"]
            # NOTE: as in steer_generate.steering_setting: alpha
            # makes the added vector `strength` times as long as the mean
            # residual stream at that layer.
            norm = np.linalg.norm(vectors[target][layer].astype(np.float64))
            alpha = float(strength * vectors["resid_norm"][layer] / norm)
            settings.append({"source": source, "target": target, "target_name": target_name,
                             "vector": f"{source}_{target}", "alpha": alpha,
                             "vectors_fingerprint": fingerprint(vectors), **setting})
            log(f"{source} {target}: layer {layer}, strength {strength}, alpha {alpha:.4g}, "
                f"{'works' if setting['works_on_own_task'] else 'best'} on {task} ({setting['value_on_own_task']:.3f})")

    out_path = RESULTS / "annotation" / f"generations_{args.model}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = json.loads(out_path.read_text()) if out_path.exists() else {"model": args.model, "decoding": DECODING,
                                                                         "settings": settings, "prompts": {}}
    assert output["settings"] == settings, f"{out_path} was made with other settings; move it away to start over"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    for source, task in VECTOR_SOURCES.items():
        vectors = load_vectors(RESULTS / "steering" / args.model / f"vectors_{task}.npz")
        for target in TARGETS:
            engine.vectors[f"{source}_{target}"] = torch.tensor(vectors[target], dtype=torch.float32, device=device)
    interventions = [Intervention()] + [Intervention(steer=((setting["layer"], setting["vector"], setting["alpha"]),))
                                        for setting in settings]
    log(f"{args.model} on {device}: {len(interventions)} answers per prompt")

    # The seven prompts: (short name, the question as the speaker hears it, prompt ids, new tokens).
    prompts = [("cind", LLAMA2_STORY_USER, task_prompt_ids(engine, "cinderella")[1], STORY_TOKENS),
               ("sand", SANDWICH_USER, task_prompt_ids(engine, "sandwich")[1], STORY_TOKENS)]
    connected = [item for item in TAB_ITEMS if item["subtest"] == "connected_text"]
    prompts += [(name, item["prompt"], engine.encode(tab_prompt(engine, item)), CONNECTED_TEXT_TOKENS)
                for name, item in zip(CONNECTED_TEXT_NAMES, connected)]

    for name, question, prompt_ids, max_new_tokens in prompts:
        if name in output["prompts"]:
            log(f"{name}: already done")
            continue
        # NOTE: the 7 answers share one prompt and
        # are generated as one unpadded batch; padding every batch to the
        # model's fixed size, as the search does for bit-identical results
        # across batches, would multiply the CPU time several times over.
        full_texts = engine.generate(prompt_ids, interventions, max_new_tokens, DECODING, pad_batches=False)
        answers = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
        scored = [answer for answer in answers if long_enough(answer)]
        probabilities = scorer.probabilities(scored)   # NOTE: (n_scored, 5), columns in CLASSES order
        values = objective_values(probabilities)       # {"A" | "B" | "C": (n_scored,)}
        records, row = [], 0
        for index, (full_text, answer) in enumerate(zip(full_texts, answers)):
            setting = None if index == 0 else settings[index - 1]
            record = {"id": None if setting is None else
                      f"{MODEL_NAMES[args.model]}_{name}_{setting['source']}vec_to_{setting['target_name']}",
                      "setting": setting, "answer": answer, "full_text": full_text,
                      "probabilities": None, "objectives": None, "predicted_class": None}
            if long_enough(answer):
                record["probabilities"] = dict(zip(CLASSES, probabilities[row].round(6).tolist()))
                record["objectives"] = {objective: round(float(value[row]), 6) for objective, value in values.items()}
                record["predicted_class"] = CLASSES[int(probabilities[row].argmax())]
                row += 1
            records.append(record)
            target = "A" if setting is None else setting["target"]
            shown = "unscored" if record["objectives"] is None else f"{target}={record['objectives'][target]:.3f}"
            log(f"  {name} {'unsteered' if setting is None else setting['vector']:<8} {shown:<10} {answer[:80]!r}")
        output["prompts"][name] = {"question": question, "records": records}
        out_path.write_text(json.dumps(output, indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
