"""Steer the five-way study's generators along aphasia directions while they answer.

The knockout search (search.py) switches components off. Steering instead adds
a fixed vector to the residual stream after one block, at every position, while
the model answers the task prompt. The vectors are differences of group means
of the model's own residual stream on the human AphasiaBank transcripts
(steer_collect.py writes them to RESULTS/steering/<model>/vectors_<task>.npz),
one row per layer:

    A                       aphasic transcripts minus Control transcripts
    B, C                    Broca minus Control, Wernicke minus Control
    B_specific, C_specific  the parts of B and C specific to that subtype
    resid_norm              the typical size of the residual stream at each layer

Strengths are relative: strength r at layer L adds alpha * v[L] with
alpha = r * resid_norm[L] / |v[L]|, a vector r times as large as the typical
residual stream at that layer, whatever the length of v. Negative r pushes
toward Control.

Modes (--modes, all six by default), every one decoded greedily with a
repetition penalty of 1.25 ("rep1.25") from the task prompts of prompts.py:

  sweep    A, B and C at 8 layers spread over the depth and 11 strengths from
           -2 to 4, plus the unsteered answer: which layer and strength turn
           the model's own answer aphasic?
  random   3 random directions as long as A at each layer (centred over d_model,
           like A, for GPT-2 and Pythia), at the same layers and the positive
           strengths: does any direction of that size do it?
  cross    the other task's A on this task's prompt: is the direction shared?
  subtype  B_specific and C_specific: does a subtype's own direction give that subtype?
  prefix   real speakers: the answer starts with the first 25 words of a Control
           or an aphasic transcript, and the model continues it for 120 tokens,
           steered with A at one layer, r in -2..2
  rescue   the knockout circuit of objective A from search.py, alone, with A
           added at negative strengths, and with the A direction projected out
           after every block: does steering back toward Control undo the
           knockout? The random directions, added and removed the same way,
           are the controls.

Every answer is stored next to its five-way probabilities and objective values
(A = P(aphasia), B = P(Broca), C = P(Wernicke)). Base-model answers are cut at
the first question they write themselves before scoring, as in search.py, and
answers shorter than prompts.MIN_STORY_CHARS characters get null scores. For Cinderella
the yes/no answer is generated under the same intervention and stored, but it
drops nothing here (the prefix mode, whose answers begin with a human's words,
has none).

Outputs:
    RESULTS/steering/<model>/<mode>_<task>.json        every setting with its text and scores
    RUNS/fiveway/steering/<model>/prefix_<task>.json   the prefix mode in full: it quotes
                                                       AphasiaBank transcripts, which stay out of git
    RESULTS/steering/<model>/prefix_<task>.json        the prefix mode's means per group and strength
    RESULTS/steering/<model>/summary_<task>.json       per vector and layer, the smallest strength
                                                       that works; random directions; rescue
A rerun (for example a requeued job) generates only the settings that its
output files do not list yet.

Usage:
    bash slurm/fiveway_submit.sh steer-p1b-cind 48G fiveway/steer_generate.py --model pythia-1b --task cinderella
  a larger model (the vectors of both tasks and the search results must be in results/ first):
    bash slurm/fiveway_submit.sh steer-l13b-cind 96G fiveway/steer_generate.py \\
        --model llama2-13b-chat --task cinderella
  CPU smoke test on a tiny grid, written to a new directory of its own so that no
  real run ever resumes from it (--vectors-dir and --search-file can point at
  other inputs, such as synthetic vectors):
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-test 32G fiveway/steer_generate.py \\
        --model pythia-1b --task cinderella --output-dir runs/fiveway/steering/test/steer_generate/<new directory> \\
        --layers 4 --strengths -1 1 --prefix-strengths 0 1 --prefix-transcripts 1 --rescue-strengths -1
  test_steer_generate.py checks the settings' steering against hooks written from scratch.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import CLASSIFIER_DIR, RESULTS, ROOT, RUNS  # noqa: E402
from fiveway.prompts import (STORY_TOKENS, YESNO_TOKENS, cut_at_new_question, long_enough,  # noqa: E402
                             parse_yes_no, task_prompt_ids, task_prompts)
from fiveway.scorer import SUCCESS_THRESHOLD, FiveWayScorer, objective_values  # noqa: E402
from fiveway.steer_collect import LAYERNORM_FAMILIES, centre_for_layernorm  # noqa: E402

DECODING = "rep1.25"
MODES = ["sweep", "random", "cross", "subtype", "prefix", "rescue"]
N_SPREAD_LAYERS = 8              # steering layers, spread evenly over the depth
# NOTE: the strengths are written as floats because
# argparse turns strengths given on the command line into floats, and a
# setting is found again by its JSON text, in which -2 and -2.0 differ.
SWEEP_STRENGTHS = [-2.0, -1.0, -0.5, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
PREFIX_STRENGTHS = [-2.0, -1.0, 0.0, 1.0, 2.0]
RESCUE_STRENGTHS = [-0.5, -1.0, -2.0, -4.0]
N_RANDOM_DIRECTIONS = 3
N_PREFIX_TRANSCRIPTS = 10        # per group: 10 Control and 10 aphasic speakers
PREFIX_WORDS = 25
MIN_TRANSCRIPT_WORDS = 60        # as in the earlier study's prefix mode
CONTINUATION_TOKENS = 120
SUBTYPES = ["Anomic", "Broca", "Conduction", "Wernicke"]


@dataclass
class Setting:
    """One generation: what the output file records about it, and the intervention that makes it."""
    fields: dict                 # vector, layer, strength, ...: identifies the setting in the output file
    intervention: Intervention
    alpha: float | None = None   # the absolute coefficient of the added vector, when there is one
    prefix: str | None = None    # prefix mode: the human words the answer starts with


def setting_key(fields: dict) -> str:
    """The text that identifies a setting in its output file.

    NOTE: the key holds the strength as given, not alpha:
    alpha comes from float32 vector norms that can differ in their last digits
    between machines, and a job requeued on another node must still recognise
    the settings it has already generated.
    """
    return json.dumps(fields, sort_keys=True)


def load_vectors(path: Path) -> dict:
    """Every array of a vectors file: (n_layers, d_model) vectors and resid_norm (n_layers,)."""
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def fingerprint(arrays: dict) -> str:
    """A short hash of named arrays: a vectors file's, or the random directions.

    NOTE: it is stored in every output file, and a rerun
    resumes only from outputs made with the same vectors. Otherwise, after
    steer_collect.py is rerun with a fix, old answers steered with the old
    vectors would silently count as done.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()[:16]


def write_json(path: Path, payload) -> None:
    """Write through a temporary file, so that a job preempted mid-write never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


def spread_layers(n_layers: int) -> list:
    """N_SPREAD_LAYERS layers evenly spread from the first block: 0, 2, ..., 14 for 16 layers."""
    return [round(i * n_layers / N_SPREAD_LAYERS) for i in range(N_SPREAD_LAYERS)]


def random_directions(vector_a: np.ndarray, centre: bool) -> dict:
    """N_RANDOM_DIRECTIONS random vectors, each as long as A at every layer: {"random0": (n_layers, d_model), ...}.

    ``centre``: whether the model reads its residual stream through LayerNorm
    (GPT-2, Pythia), for which steer_collect.py centred A over d_model.

    NOTE: in d_model dimensions a random Gaussian direction is
    almost orthogonal to any fixed one (cosine of order 1 / sqrt(d_model),
    about 0.02 for 2048), so these are vectors of A's size that point nowhere
    in particular: if they also turn the answer aphasic, A is not special.
    NOTE: a LayerNorm cannot see the all-ones direction
    (steer_collect.centre_for_layernorm), so for GPT-2 and Pythia A has no
    component along it, and the random directions are centred the same way to
    live where A lives. Adding an uncentred one would change little, but
    removing one (rescue mode) would not: the raw residual stream of the
    Hugging Face model has a mean over d_model that LayerNorm ignores (up to
    about 11 in GPT-2 XL's last layers), and hidden @ unit picks it up through
    the direction's own small mean. "Removing" the direction would then also
    push the stream along it by an amount unrelated to it; in GPT-2 XL that
    push is as large as the removed component itself.
    """
    directions = {}
    for seed in range(N_RANDOM_DIRECTIONS):
        gaussian = np.random.default_rng(seed).standard_normal(vector_a.shape)
        if centre:
            gaussian = centre_for_layernorm(gaussian)
        # NOTE: both norms are (n_layers, 1), one per layer, and
        # broadcast over d_model: every layer's row gets A's length at that layer.
        gaussian *= np.linalg.norm(vector_a, axis=1, keepdims=True) / np.linalg.norm(gaussian, axis=1, keepdims=True)
        directions[f"random{seed}"] = gaussian.astype(np.float32)
    return directions


def speaker(row: dict) -> str:
    """An approximate speaker id: the corpus and the file name without its session suffix.

    NOTE: the rule of steering/check_duplicates.py:
    AphasiaBank names a session <speaker><session> (<speaker>-<session> or
    <speaker><session letter>), and some corpora add a PWA- prefix; stripping both maps
    every session of a speaker to one id.
    """
    base = row["File_DB"].split(".")[0].replace("PWA-", "")
    return row["Corpus"] + "|" + re.sub(r"(-\d+|-LARC|[a-z])$", "", base)


def prefix_transcripts(task: str, n_per_group: int) -> list:
    """The prefix mode's speakers: n Control and n aphasic transcripts, in the corpus's order.

    Each text is used once and each speaker once, and only transcripts of at
    least MIN_TRANSCRIPT_WORDS words, so that the 25-word prefix is the
    beginning of a narrative and not all of it. The aphasic speakers are taken
    from the four subtypes in turn: Anomic, Broca, Conduction, Wernicke, Anomic, ...

    NOTE: in corpus order alone, the first ten aphasic
    Cinderella transcripts are all mild Anomic speakers of one sub-corpus,
    several of them twice (two sessions of one speaker). Taking the subtypes in
    turn and one session per speaker spreads the ten over subtypes and severity.
    """
    rows = json.loads((CLASSIFIER_DIR / "data" / task / "transcripts.json").read_text())
    groups_of_text = {}
    for row in rows:
        groups_of_text.setdefault(row["text_response"], set()).add(row["Group"])
    seen_texts, seen_speakers, usable = set(), set(), []
    for row in rows:
        text = row["text_response"]
        # NOTE: the sandwich corpus lists about 557
        # transcripts twice under two file names (<file>.gem.cex and
        # PWA-<file>.gem.cex), and four Cinderella texts carry both the Control
        # and the Anomic label: a repeated text is skipped, a text with two
        # labels is never used.
        if text in seen_texts or len(groups_of_text[text]) > 1:
            continue
        seen_texts.add(text)
        if len(text.split()) >= MIN_TRANSCRIPT_WORDS and speaker(row) not in seen_speakers:
            seen_speakers.add(speaker(row))
            usable.append(row)
    control = [row for row in usable if row["Group"] == "Control"][:n_per_group]
    by_subtype = [[row for row in usable if row["Group"] == subtype] for subtype in SUBTYPES]
    in_turn = [rows_of_subtype[rank] for rank in range(n_per_group) for rows_of_subtype in by_subtype
               if rank < len(rows_of_subtype)]
    return control + in_turn[:n_per_group]


def steering_setting(vectors: dict, resid_norm: np.ndarray, name: str, objective: str, layer: int,
                     strength: float, base: Intervention = Intervention(), **fields) -> Setting:
    """``base`` plus vector ``name`` added after block ``layer``, at a strength relative to the residual stream.

    NOTE: alpha = r * resid_norm[L] / |v[L]| makes the added
    vector alpha * v[L] exactly r times as long as the typical residual stream
    at layer L, however long v[L] is. A strength therefore means the same for
    every vector, layer and model, and the random directions, being as long as
    A, get exactly A's alpha.
    """
    alpha = float(strength * resid_norm[layer] / np.linalg.norm(vectors[name][layer].astype(np.float64)))
    fields.update(vector=name, objective=objective, layer=layer, strength=strength)
    return Setting(fields, dataclasses.replace(base, steer=((layer, name, alpha),)), alpha)


def build_settings(mode: str, args, vectors: dict, resid_norm: np.ndarray, transcripts: list,
                   circuit: dict, other_task: str) -> list:
    """Every generation of one mode.

    ``objective`` names the objective a vector is meant to raise: A for A, the
    other task's A and the random directions, B for B and B_specific, C for C
    and C_specific. The summary checks each vector against its own objective.
    """
    positive_strengths = [strength for strength in args.strengths if strength > 0]
    settings = []
    if mode == "sweep":
        settings.append(Setting({"vector": None, "objective": None, "layer": None, "strength": 0.0}, Intervention()))
        for name in ("A", "B", "C"):
            for layer in args.layers:
                for strength in args.strengths:
                    settings.append(steering_setting(vectors, resid_norm, name, name, layer, strength))

    elif mode == "random":
        for seed in range(N_RANDOM_DIRECTIONS):
            for layer in args.layers:
                for strength in positive_strengths:
                    settings.append(steering_setting(vectors, resid_norm, f"random{seed}", "A", layer, strength))

    elif mode == "cross":
        # NOTE: the strength is relative to this task's
        # residual stream, the one being steered, so the other task's vector
        # is added at exactly the size this task's own A would be.
        for layer in args.layers:
            for strength in positive_strengths:
                settings.append(steering_setting(vectors, resid_norm, f"A_{other_task}", "A", layer, strength))

    elif mode == "subtype":
        for name, objective in (("B_specific", "B"), ("C_specific", "C")):
            for layer in args.layers:
                for strength in positive_strengths:
                    settings.append(steering_setting(vectors, resid_norm, name, objective, layer, strength))

    elif mode == "prefix":
        for row in transcripts:
            # NOTE: Custom_field holds the aphasia quotient
            # of aphasic speakers as a string, "." when it is unknown; for
            # Control speakers it means something else, so it is not kept.
            speaker_fields = {"File_DB": row["File_DB"],
                              "group": "Control" if row["Group"] == "Control" else "aphasic",
                              "aphasiabank_group": row["Group"],
                              "aq": None if row["Group"] == "Control" else row["Custom_field"]}
            for strength in args.prefix_strengths:
                setting = steering_setting(vectors, resid_norm, "A", "A", args.layer, strength, **speaker_fields)
                setting.prefix = " ".join(row["text_response"].split()[:PREFIX_WORDS])
                settings.append(setting)

    elif mode == "rescue":
        knockout = Intervention.knockout(heads=circuit["heads"], mlp_blocks=circuit["mlp_blocks"])
        # NOTE: the intact model and the intact model with
        # the direction removed are the references: the first is where a
        # rescue should land, the second shows what removing A does by itself.
        # The random directions are the controls. Under greedy decoding any
        # push of this size can move the answer off the knockout's particular
        # path (in Pythia-1B's Cinderella run, removing one random direction
        # took the knockout from P(aphasia) 0.93 to 0.44, and pushes along the
        # random directions undid it about as often as pushes along A), so A
        # undoes the knockout only in a meaningful sense where they do not.
        random_names = [f"random{seed}" for seed in range(N_RANDOM_DIRECTIONS)]
        references = [("intact", Intervention()), ("intact, A removed", Intervention(remove=("A",))),
                      ("knockout", knockout)]
        references += [(f"knockout, {name} removed", dataclasses.replace(knockout, remove=(name,)))
                       for name in ["A"] + random_names]
        for name, intervention in references:
            settings.append(Setting({"intervention": name, "vector": None, "objective": "A", "layer": None,
                                     "strength": None}, intervention))
        for name in ["A"] + random_names:
            for layer in args.layers:
                for strength in args.rescue_strengths:
                    settings.append(steering_setting(vectors, resid_norm, name, "A", layer, strength, base=knockout,
                                                     intervention="knockout + steer"))
    return settings


def generate_and_score(engine, scorer, task: str, interventions: list, prefix: str | None) -> list:
    """Every intervention's answer, with its yes/no answer (Cinderella) and its five-way scores."""
    yesno_ids, story_ids = task_prompt_ids(engine, task)
    max_new_tokens = STORY_TOKENS
    if prefix is not None:
        # NOTE: the human words are written after the
        # prompt as the beginning of the model's own answer, and only what
        # the model adds is scored. The joined text is tokenized as a whole,
        # so the first word gets the same token as if the model had produced
        # it ("Answer: Once" and "[/INST] Once" both give the word-initial "Once").
        # NOTE: the repetition penalty counts every token
        # already in the context, the human prefix included, so the model is
        # also discouraged from repeating the speaker's own words.
        _, story_prompt = task_prompts(engine.tokenizer, engine.spec.is_chat, task)
        story_ids = engine.encode(story_prompt + " " + prefix)
        yesno_ids, max_new_tokens = None, CONTINUATION_TOKENS
    yesno_texts = [None] * len(interventions)
    if yesno_ids is not None:
        yesno_texts = engine.generate(yesno_ids, interventions, YESNO_TOKENS, DECODING)
    full_texts = engine.generate(story_ids, interventions, max_new_tokens, DECODING)
    # NOTE: as in search.py: a base model's answer ends
    # where it starts writing a question of its own; chat answers stay whole.
    stories = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
    scored = [story for story in stories if long_enough(story)]
    # NOTE: probabilities: (n_scored, 5), columns Control,
    # Conduction/Anomic, Broca, Wernicke, Model_scrambles; every objective
    # in values is (n_scored,).
    probabilities = scorer.probabilities(scored)
    values = objective_values(probabilities)
    records, row = [], 0
    for yesno, full_text, story in zip(yesno_texts, full_texts, stories):
        record = {"yesno": yesno, "answer": None if yesno is None else parse_yes_no(yesno)[0],
                  "full_text": full_text, "story": story, "probabilities": None, "objectives": None}
        # NOTE: a steered model can stop at once or emit
        # only whitespace; such answers are kept with null scores.
        if long_enough(story):
            record["probabilities"] = probabilities[row].round(6).tolist()
            record["objectives"] = {name: round(float(value[row]), 6) for name, value in values.items()}
            row += 1
        records.append(record)
    return records


def prefix_aggregate(entries: list) -> list:
    """One row per steering layer, speaker group and strength: counts and mean objective values, no text."""
    cells = {}
    for entry in entries:
        cells.setdefault((entry["layer"], entry["group"], entry["strength"]), []).append(entry["objectives"])
    rows = []
    for (layer, group, strength), objectives in sorted(cells.items()):
        scored = [values for values in objectives if values is not None]
        rows.append({"layer": layer, "group": group, "strength": strength,
                     "n": len(objectives), "n_scored": len(scored),
                     "mean": {name: float(np.mean([values[name] for values in scored])) if scored else None
                              for name in ("A", "B", "C")},
                     "share_above_threshold": {name: float(np.mean([values[name] > SUCCESS_THRESHOLD
                                                                    for values in scored])) if scored else None
                                               for name in ("A", "B", "C")}})
    return rows


def summarize(out_dir: Path, task: str, fingerprints: dict) -> dict:
    """The main numbers of one model and task, read back from the mode files written so far.

    ``fingerprints`` holds the current vectors' (under the task's name) and
    random directions' ("random") fingerprints.
    """
    outputs = {}
    for mode in ("sweep", "random", "cross", "subtype", "rescue"):
        path = out_dir / f"{mode}_{task}.json"
        if path.exists():
            outputs[mode] = json.loads(path.read_text())
            # NOTE: main() checks a mode file against the
            # current inputs only when it runs that mode. A file left over from
            # a run with other vectors or other random directions (say, after
            # --modes sweep alone with new vectors) must not mix into the summary.
            assert outputs[mode]["inputs"]["vectors"] == fingerprints[task], f"{path} was made from other vectors"
            if mode in ("random", "rescue"):
                assert outputs[mode]["inputs"].get("random_directions") == fingerprints["random"], \
                    f"{path} was made with other random directions"

    def entries(mode):
        return outputs[mode]["entries"] if mode in outputs else []

    def above(entry, objective):
        # NOTE: an unscored answer (shorter than
        # prompts.MIN_STORY_CHARS) never counts as above the threshold, as in search.py.
        return entry["objectives"] is not None and entry["objectives"][objective] > SUCCESS_THRESHOLD

    def share_above(selected, objective):
        return float(np.mean([above(entry, objective) for entry in selected])) if selected else None

    summary = {"threshold": SUCCESS_THRESHOLD, "modes": sorted(outputs),
               "inputs": {mode: outputs[mode]["inputs"] for mode in sorted(outputs)},
               "n_unscored": {mode: sum(entry["objectives"] is None for entry in entries(mode)) for mode in outputs}}
    unsteered = [entry for entry in entries("sweep") if entry["vector"] is None]
    summary["unsteered"] = unsteered[0]["objectives"] if unsteered else None

    # For every vector and layer, the strengths at which the answer is above
    # the threshold on the vector's own objective; the summary keeps the smallest.
    working = {}
    for mode in ("sweep", "cross", "subtype", "random"):
        for entry in entries(mode):
            if entry["vector"] is None or entry["strength"] <= 0:
                continue
            strengths = working.setdefault(entry["vector"], {}).setdefault(str(entry["layer"]), [])
            if above(entry, entry["objective"]):
                strengths.append(entry["strength"])
    summary["smallest_strength_above_threshold"] = {
        vector: {layer: min(strengths) if strengths else None for layer, strengths in by_layer.items()}
        for vector, by_layer in working.items()}

    # Random directions against A itself, at the same layers and strengths.
    random_entries = entries("random")
    random_layers = {entry["layer"] for entry in random_entries}
    a_entries = [entry for entry in entries("sweep") if entry["vector"] == "A" and entry["layer"] in random_layers]
    summary["random"] = {
        "n_settings": len(random_entries),
        "share_above_threshold": share_above(random_entries, "A"),
        "share_above_threshold_by_strength": {
            str(strength): {
                "random": share_above([entry for entry in random_entries if entry["strength"] == strength], "A"),
                "A": share_above([entry for entry in a_entries if entry["strength"] == strength], "A")}
            for strength in sorted({entry["strength"] for entry in random_entries})}}

    if "rescue" in outputs:
        rescue_entries = entries("rescue")

        def undoes(entry):
            # NOTE: an unscored answer is no rescue.
            return entry["objectives"] is not None and entry["objectives"]["A"] <= SUCCESS_THRESHOLD

        # For A and every random direction: per layer, the strengths whose
        # push brings the knockout's P(aphasia) to the threshold or below.
        undoing, steered = {}, {}
        for entry in rescue_entries:
            if entry["intervention"] == "knockout + steer":
                steered.setdefault(entry["vector"], []).append(entry)
                strengths = undoing.setdefault(entry["vector"], {}).setdefault(str(entry["layer"]), [])
                if undoes(entry):
                    strengths.append(entry["strength"])
        summary["rescue"] = {
            "circuit": outputs["rescue"]["inputs"]["circuit"],
            "value_A": {entry["intervention"]: None if entry["objectives"] is None else entry["objectives"]["A"]
                        for entry in rescue_entries if entry["layer"] is None},
            # NOTE: these strengths are negative, so max
            # picks the one closest to zero: the weakest push back toward
            # Control that brings P(aphasia) to the threshold or below.
            "weakest_strength_undoing_knockout": {
                vector: {layer: max(strengths) if strengths else None for layer, strengths in by_layer.items()}
                for vector, by_layer in undoing.items()},
            "share_undoing_knockout": {vector: float(np.mean([undoes(entry) for entry in vector_entries]))
                                       for vector, vector_entries in steered.items()}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="steering layers of every mode but prefix (default: 8 spread over the depth)")
    parser.add_argument("--strengths", type=float, nargs="+", default=SWEEP_STRENGTHS,
                        help="the sweep's strengths; random, cross and subtype use the positive ones")
    parser.add_argument("--layer", type=int, default=None, help="prefix mode's layer (default: a quarter of the depth)")
    parser.add_argument("--prefix-strengths", type=float, nargs="+", default=PREFIX_STRENGTHS)
    parser.add_argument("--prefix-transcripts", type=int, default=N_PREFIX_TRANSCRIPTS,
                        help="transcripts per group (Control, aphasic) in prefix mode")
    parser.add_argument("--rescue-strengths", type=float, nargs="+", default=RESCUE_STRENGTHS)
    parser.add_argument("--vectors-dir", default=None,
                        help="directory of vectors_<task>.npz, relative to the repository "
                             "(default RESULTS/steering/<model>)")
    parser.add_argument("--search-file", default=None,
                        help="the knockout search's results "
                             "(default RESULTS/search/<model>_<task>_rep1.25_classifier.json)")
    parser.add_argument("--output-dir", default=None,
                        help="for tests: write everything under this directory, in results/ and runs/, "
                             "instead of RESULTS/steering/<model> and RUNS/fiveway/steering/<model>")
    args = parser.parse_args()
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    # NOTE: ROOT / path is path itself when path is absolute,
    # so the options take paths relative to the repository or absolute ones.
    other_task = [task for task in TASKS if task != args.task][0]
    vectors_dir = ROOT / args.vectors_dir if args.vectors_dir else RESULTS / "steering" / args.model
    search_file = (ROOT / args.search_file if args.search_file
                   else RESULTS / "search" / f"{args.model}_{args.task}_{DECODING}_classifier.json")
    out_dir = ROOT / args.output_dir / "results" if args.output_dir else RESULTS / "steering" / args.model
    runs_dir = ROOT / args.output_dir / "runs" if args.output_dir else RUNS / "fiveway" / "steering" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # The steering vectors: name -> (n_layers, d_model) array.
    task_vectors = load_vectors(vectors_dir / f"vectors_{args.task}.npz")
    fingerprints = {args.task: fingerprint(task_vectors)}
    resid_norm = task_vectors["resid_norm"]                       # NOTE: (n_layers,)
    vectors = {name: task_vectors[name] for name in ("A", "B", "C", "B_specific", "C_specific")}
    random = random_directions(task_vectors["A"], centre=MODELS[args.model].family in LAYERNORM_FAMILIES)
    # NOTE: the random directions are made here, not read
    # from a file, so their own fingerprint goes into the outputs of the modes
    # that use them: answers made with other random directions are never resumed.
    fingerprints["random"] = fingerprint(random)
    vectors.update(random)
    if "cross" in args.modes:
        other_vectors = load_vectors(vectors_dir / f"vectors_{other_task}.npz")
        fingerprints[other_task] = fingerprint(other_vectors)
        vectors[f"A_{other_task}"] = other_vectors["A"]
    n_layers = task_vectors["A"].shape[0]
    args.layers = args.layers or spread_layers(n_layers)
    args.layer = round(n_layers / 4) if args.layer is None else args.layer

    circuit = None
    if "rescue" in args.modes and search_file.exists():
        objective_a = json.loads(search_file.read_text())["objectives"].get("A", {})
        if objective_a.get("found_at_stage"):
            found = objective_a["pruned"]["circuit"]
            circuit = {"heads": found["heads"], "mlp_blocks": found["mlp_blocks"], "value": found["value"]}
    transcripts = prefix_transcripts(args.task, args.prefix_transcripts) if "prefix" in args.modes else []
    log(f"{args.model} {args.task}: modes {args.modes}, layers {args.layers}, strengths {args.strengths}, "
        f"prefix layer {args.layer} and strengths {args.prefix_strengths}, "
        f"rescue strengths {args.rescue_strengths}; vectors {fingerprints}; circuit {circuit}")

    engine = scorer = None
    for mode in args.modes:
        if mode == "rescue" and circuit is None:
            reason = ("the knockout search found no circuit for objective A" if search_file.exists()
                      else "no knockout search results yet (rerun once the search is done)")
            log(f"rescue: skipped, {reason}: {search_file}")
            continue
        settings = build_settings(mode, args, vectors, resid_norm, transcripts, circuit, other_task)
        path = (runs_dir if mode == "prefix" else out_dir) / f"{mode}_{args.task}.json"
        inputs = {"vectors": fingerprints[args.task]}
        if mode == "cross":
            inputs["other_task_vectors"] = fingerprints[other_task]
        if mode in ("random", "rescue"):
            inputs["random_directions"] = fingerprints["random"]
        if mode == "rescue":
            inputs["circuit"] = circuit
        output = {"model": args.model, "task": args.task, "mode": mode, "decoding": DECODING, "inputs": inputs,
                  "entries": []}
        if path.exists():
            output = json.loads(path.read_text())
            if output["inputs"] != inputs:
                raise SystemExit(f"{path} was made from other inputs ({output['inputs']}, now {inputs}); "
                                 "move it away to start this mode over")
            # NOTE: answers made before invisible format
            # characters stopped counting toward an answer's length
            # (prompts.long_enough) were scored by len(); they lose their scores,
            # as generate_and_score would give them now.
            outdated = [entry for entry in output["entries"]
                        if entry["objectives"] is not None and not long_enough(entry["story"])]
            for entry in outdated:
                entry["probabilities"] = entry["objectives"] = None
            if outdated:
                write_json(path, output)
                log(f"{mode}: {len(outdated)} answers of invisible characters unscored in {path.name}")
        done = {entry["key"] for entry in output["entries"]}
        missing = [setting for setting in settings if setting_key(setting.fields) not in done]
        if not missing:
            log(f"{mode}: all {len(settings)} settings already in {path}")
            continue

        # NOTE: the models are loaded only once something
        # is missing, so a rerun after everything is done only rewrites the
        # summary.
        if engine is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            engine = Engine(args.model, device)
            scorer = FiveWayScorer(device)
            for name, vector in vectors.items():
                assert vector.shape == (engine.geometry["n_layers"], engine.geometry["d_model"]), (name, vector.shape)
                engine.vectors[name] = torch.tensor(vector, dtype=torch.float32, device=device)
            log(f"loaded {args.model} and the classifier on {device}; batches of {engine.batch_size}")
        log(f"{mode}: {len(missing)} of {len(settings)} settings to generate -> {path}")

        # NOTE: every batch shares one prompt, so
        # the prefix mode, whose prompt changes with the speaker, fills only
        # len(prefix_strengths) of each padded batch. Batching several prompts
        # would need left padding with attention masks and position ids,
        # which the engine does not do.
        by_prefix = {}
        for setting in missing:
            by_prefix.setdefault(setting.prefix, []).append(setting)
        for prefix, group in by_prefix.items():
            for begin in range(0, len(group), engine.batch_size):
                batch = group[begin:begin + engine.batch_size]
                interventions = [setting.intervention for setting in batch]
                records = generate_and_score(engine, scorer, args.task, interventions, prefix)
                for setting, record in zip(batch, records):
                    entry = {"key": setting_key(setting.fields), **setting.fields, "alpha": setting.alpha, **record}
                    if setting.prefix is not None:
                        entry["prefix"] = setting.prefix
                    output["entries"].append(entry)
                    scores = ("unscored" if record["objectives"] is None
                              else " ".join(f"{name}={value:.3f}" for name, value in record["objectives"].items()))
                    # The prefix mode's continuations can repeat a speaker's
                    # words, so its texts stay out of the log.
                    shown = "" if prefix is not None else repr(record["story"][:70])
                    described = {name: value for name, value in setting.fields.items() if value is not None}
                    log(f"  {described}  {scores}  {shown}")
                write_json(path, output)
                log(f"{mode}: {len(output['entries'])} settings in {path.name}")

    prefix_path = runs_dir / f"prefix_{args.task}.json"
    if "prefix" in args.modes and prefix_path.exists():
        prefix_output = json.loads(prefix_path.read_text())
        speakers = {}
        for entry in prefix_output["entries"]:
            speakers.setdefault(entry["aphasiabank_group"], set()).add(entry["File_DB"])
        write_json(out_dir / f"prefix_{args.task}.json", {
            "model": args.model, "task": args.task, "mode": "prefix", "decoding": DECODING,
            "inputs": prefix_output["inputs"], "prefix_words": PREFIX_WORDS, "continuation_tokens": CONTINUATION_TOKENS,
            "speakers": {group: len(files) for group, files in sorted(speakers.items())},
            "rows": prefix_aggregate(prefix_output["entries"])})
    summary_path = out_dir / f"summary_{args.task}.json"
    write_json(summary_path, {"model": args.model, "task": args.task, **summarize(out_dir, args.task, fingerprints)})
    log(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
