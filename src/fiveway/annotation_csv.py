"""The two annotation sheets of the steered answers: classifier and LLM judge side by side, for human labels.

Reads every results/fiveway/annotation/generations_<model>.json of
steer_annotate.py and writes, in the same folder:

    annotation_binary.csv    the answers steered with A (aphasia): is the speaker aphasic?
    annotation_fiveway.csv   the answers steered with B (Broca) or C (Wernicke): which of five classes?

with the columns

    id             <model>_<prompt>_<cind|sand>vec_to_<aphasia|broca|wernicke>,
                   e.g. llama_7b_trip_cindvec_to_aphasia
    question       the prompt the model answered
    answer         the steered model's answer
    annotator_1    the five-way classifier: "aphasia" when P(Conduction/Anomic) +
                   P(Broca) + P(Wernicke) > 0.5, else "not aphasia" (binary sheet);
                   its most probable class (five-way sheet)
    annotator_2    the LLM judge (llm_judge.py, DeepSeek-V4.1-Flash): its yes or no
                   to "Does the speaker have aphasia?" (binary sheet); the class it
                   names among the five (five-way sheet)
    human_label_1, human_label_2   empty, for two human annotators

Five-way labels use the classifier's class names: Control, Conduction/Anomic,
Broca, Wernicke, Model_scrambles (the judge's "Not human speech (garbled
machine text)"). An answer too short for the classifier (prompts.long_enough)
gets "unscored" from it; the judge still reads it.

annotation_details.json holds, per id, the model, prompt, vector source,
target, layer, strength, the classifier's five probabilities and the judge's
reply, and the agreement between the two annotators.

Usage (after every steer_annotate.py job; the judge's replies are cached):
    GPUS=0 bash slurm/fiveway_submit.sh annotation-csv 8G fiveway/annotation_csv.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.llm_judge import (JUDGE_MODEL, Judge, binary_messages, fiveway_messages,  # noqa: E402
                               parse_binary, parse_fiveway)
from fiveway.models import MODELS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import SUCCESS_THRESHOLD  # noqa: E402

FOLDER = RESULTS / "annotation"
COLUMNS = ["id", "question", "answer", "annotator_1", "annotator_2", "human_label_1", "human_label_2"]


def cohen_kappa(first: list, second: list) -> float:
    """Agreement of two annotators beyond what their label frequencies give by chance."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    counts_first, counts_second = Counter(first), Counter(second)
    # NOTE: chance agreement: the probability that two
    # annotators labelling independently, each with their own label
    # frequencies, happen to give the same label.
    expected = sum(counts_first[label] * counts_second[label] for label in counts_first) / n ** 2
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def main():
    rows = []
    for model in MODELS:
        path = FOLDER / f"generations_{model}.json"
        if not path.exists():
            print(f"no generations for {model} yet: {path}", flush=True)
            continue
        generations = json.loads(path.read_text())
        for prompt_name, prompt in generations["prompts"].items():
            for record in prompt["records"]:
                if record["setting"] is not None:     # the unsteered answer is kept for reference only
                    rows.append({"model": model, "prompt": prompt_name, "question": prompt["question"], **record})

    judge = Judge()
    conversations = [(binary_messages if row["setting"]["target"] == "A" else fiveway_messages)(row["question"],
                                                                                               row["answer"])
                     for row in rows]
    replies = judge.ask_many(conversations)

    sheets = {"binary": [], "fiveway": []}
    details = {}
    for row, reply in zip(rows, replies):
        target = row["setting"]["target"]
        if target == "A":
            sheet = "binary"
            classifier = ("unscored" if row["objectives"] is None else
                          "aphasia" if row["objectives"]["A"] > SUCCESS_THRESHOLD else "not aphasia")
            judged = parse_binary(reply) or "unparsed"
        else:
            sheet = "fiveway"
            classifier = row["predicted_class"] or "unscored"
            judged = parse_fiveway(reply) or "unparsed"
        sheets[sheet].append({"id": row["id"], "question": row["question"], "answer": row["answer"],
                              "annotator_1": classifier, "annotator_2": judged,
                              "human_label_1": "", "human_label_2": ""})
        details[row["id"]] = {"model": row["model"], "prompt": row["prompt"], "setting": row["setting"],
                              "classifier_probabilities": row["probabilities"],
                              "classifier_target_probability": None if row["objectives"] is None
                              else row["objectives"][target],
                              "annotator_1": classifier, "annotator_2": judged, "judge_reply": reply}

    agreement = {}
    for sheet, sheet_rows in sheets.items():
        path = FOLDER / f"annotation_{sheet}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(sheet_rows)
        # Agreement only where both annotators gave a label.
        both = [(row["annotator_1"], row["annotator_2"]) for row in sheet_rows
                if row["annotator_1"] != "unscored" and row["annotator_2"] != "unparsed"]
        first, second = [pair[0] for pair in both], [pair[1] for pair in both]
        agreement[sheet] = {"n_rows": len(sheet_rows), "n_both_labelled": len(both),
                            "raw_agreement": sum(a == b for a, b in both) / len(both),
                            "cohen_kappa": cohen_kappa(first, second),
                            "classifier_labels": dict(Counter(row["annotator_1"] for row in sheet_rows)),
                            "judge_labels": dict(Counter(row["annotator_2"] for row in sheet_rows))}
        print(f"wrote {path} ({len(sheet_rows)} rows)", flush=True)
        print(f"  {json.dumps(agreement[sheet])}", flush=True)

    # How often each annotator calls a B or C answer by its own target class.
    for target, label in (("B", "Broca"), ("C", "Wernicke")):
        target_rows = [details[row["id"]] for row in sheets["fiveway"]
                       if details[row["id"]]["setting"]["target"] == target]
        agreement[f"fiveway_{label}_hit_rate"] = {
            "n": len(target_rows),
            "classifier": sum(row["annotator_1"] == label for row in target_rows) / len(target_rows),
            "judge": sum(row["annotator_2"] == label for row in target_rows) / len(target_rows)}
        print(f"  steered to {label}: {json.dumps(agreement[f'fiveway_{label}_hit_rate'])}", flush=True)

    (FOLDER / "annotation_details.json").write_text(json.dumps({"judge": JUDGE_MODEL, "agreement": agreement,
                                                                "rows": details}, indent=1))
    print(f"wrote {FOLDER / 'annotation_details.json'}", flush=True)


if __name__ == "__main__":
    main()
