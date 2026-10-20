"""Human labels of the steered answers: do the classifier and the judge agree with them, and does steering hit its target?

The experts' labels are not distributed with this release. With labels of your own: the five-way sheet's label
columns are human_label1 and human_label2, the binary sheet adds a model_scrambles_comment column, and
human_2_own_tab_labels.csv has the columns id and human_label_2_own_tab.

Reads the two annotation sheets of annotation_csv.py after two human experts labelled them (rows the classifier
could not score or the judge could not parse were removed by hand):

    results/fiveway/annotation/annotation_binary_labelled.csv    answers steered with A: aphasia or not aphasia
    results/fiveway/annotation/annotation_fiveway_labelled.csv   answers steered with B (Broca) or C (Wernicke):
                                                                 control, conduction/anomic, broca, wernicke or
                                                                 model_scrambles
    results/fiveway/annotation/annotation_details.json           each answer's model, prompt, steering vector
                                                                 and the classifier's five probabilities
    results/fiveway/annotation/human_2_own_tab_labels.csv        the second expert's labels as in their own tab of
                                                                 the annotation sheet (see below)

Model scrambles are machine artifacts rather than language: token loops, stray or <eos> tokens, repeated
sentences (the annotation sheet's model_scrambles_definition tab). They are what steering gives when it
overshoots. A five-way answer counts as model scrambles when either expert labelled it so. A binary answer has
no such label, so it counts when its model_scrambles_comment is filled in, that is, when either expert's comment
in the sheet describes such an artifact.

Every comparison with the experts counts a match with either one of them:

    1. agreement of the classifier (annotator_1) and the LLM judge (annotator_2) with the experts on the binary
       answers, the five-way answers and both together, with and without the model-scrambles answers; precision
       and recall of the model_scrambles class; and how often the two experts agree (percent and Cohen's kappa).
    2. steering: the share of answers an expert labelled with the steering target (aphasia, broca or wernicke),
       with and without the model-scrambles answers, by target, by model, and on the prompt whose answers chose
       the steering setting (steer_annotate.py) against the six other prompts.

The second expert's labels come from their own tab of the annotation sheet (human_2_own_tab_labels.csv), not from
the labelled sheets' human_label_2: wherever the two differ, the labelled sheets hold the first expert's label,
which looks like a label set after the experts compared notes. The report repeats the main tables with the
labelled sheets' human_label_2 for comparison.

Output: results/fiveway/report/annotation_agreement.md

Usage (CPU, seconds):
    python src/fiveway/annotation_agreement.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.paths import RESULTS  # noqa: E402

FOLDER = RESULTS / "annotation"
OUT = RESULTS / "report" / "annotation_agreement.md"
TARGET_LABEL = {"A": "aphasia", "B": "broca", "C": "wernicke"}
APHASIA_CLASSES = {"conduction/anomic", "broca", "wernicke"}
SCRAMBLES = "model_scrambles"
HUMAN_COLUMNS = {"binary": ("human_label_1", "human_label_2"), "fiveway": ("human_label1", "human_label2")}


def load_rows(human_2_own_tab: dict | None = None) -> list[dict]:
    """One dict per labelled answer of both sheets; human_2_own_tab, if given, replaces the second expert's labels."""
    details = json.loads((FOLDER / "annotation_details.json").read_text())["rows"]
    rows = []
    for sheet, (first_column, second_column) in HUMAN_COLUMNS.items():
        with open(FOLDER / f"annotation_{sheet}_labelled.csv", newline="") as handle:
            for record in csv.DictReader(handle):
                detail = details[record["id"]]
                # NOTE: the experts wrote some five-way labels capitalised ("Wernicke"), the
                # classifier and judge all of them ("Model_scrambles"); lowercase makes them comparable.
                first = record[first_column].lower()
                second = (human_2_own_tab[record["id"]] if human_2_own_tab else record[second_column]).lower()
                # NOTE: the second expert's own tab leaves a few labels empty or "???"; those
                # answers keep the first expert's label only.
                experts = [label for label in (first, second) if label not in ("", "???")]
                probabilities = detail["classifier_probabilities"]
                rows.append({
                    "id": record["id"], "sheet": sheet, "model": detail["model"], "prompt": detail["prompt"],
                    "source": detail["setting"]["source"], "target": TARGET_LABEL[detail["setting"]["target"]],
                    "classifier": record["annotator_1"].lower(), "judge": record["annotator_2"].lower(),
                    "experts": experts,
                    "scrambles": (record["model_scrambles_comment"] != "" if sheet == "binary"
                                  else SCRAMBLES in experts),
                    # The classifier's most probable class, which the binary sheet's annotator_1 (aphasia or
                    # not) does not show: the binary answers' model_scrambles predictions.
                    "classifier_class": max(probabilities, key=probabilities.get).lower()})
    return rows


def cohen_kappa(first: list, second: list) -> float:
    """Agreement of two annotators beyond what their label frequencies give by chance (as in annotation_csv.py)."""
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    counts_first, counts_second = Counter(first), Counter(second)
    # NOTE: chance agreement: the probability that two annotators labelling independently, each
    # with their own label frequencies, give the same label. Pooling the two sheets is fine: their labels never
    # coincide, so a binary and a five-way answer can never agree by chance.
    expected = sum(counts_first[label] * counts_second[label] for label in counts_first) / n ** 2
    return (observed - expected) / (1 - expected)


def agreement(rows: list[dict], annotator: str) -> float:
    """Share of answers where the annotator's label matches either expert's."""
    return sum(row[annotator] in row["experts"] for row in rows) / len(rows)


def constant_agreement(rows: list[dict]) -> float:
    """The best agreement of a labeller that gives every answer of a sheet one and the same label."""
    # NOTE: matching either of two experts is a lenient test, so this is the floor that a
    # classifier's agreement has to clear to mean anything.
    hits = 0
    for sheet in HUMAN_COLUMNS:
        sheet_rows = [row for row in rows if row["sheet"] == sheet]
        labels = {label for row in sheet_rows for label in row["experts"]}
        hits += max((sum(label in row["experts"] for row in sheet_rows) for label in labels), default=0)
    return hits / len(rows)


def expert_agreement(rows: list[dict]) -> tuple[int, float, float]:
    """Number of answers with two expert labels, the experts' percent agreement on them, and Cohen's kappa."""
    pairs = [row["experts"] for row in rows if len(row["experts"]) == 2]
    first, second = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    return len(pairs), sum(a == b for a, b in pairs) / len(pairs), cohen_kappa(first, second)


def precision_recall(rows: list[dict], predicts_scrambles) -> str:
    """Precision and recall of a model_scrambles prediction against the experts' model-scrambles answers."""
    predicted = [row for row in rows if predicts_scrambles(row)]
    actual = [row for row in rows if row["scrambles"]]
    true_positives = sum(row["scrambles"] for row in predicted)
    precision = f"{true_positives / len(predicted):.0%}" if predicted else "–"
    return f"{precision} ({true_positives}/{len(predicted)}) | {true_positives / len(actual):.0%} ({true_positives}/{len(actual)})"


def hit_rate(rows: list[dict], annotator: str) -> float:
    """Share of answers labelled with their steering target, by either expert or by the classifier or judge."""
    if annotator == "experts":
        return sum(row["target"] in row["experts"] for row in rows) / len(rows)
    return sum(row[annotator] == row["target"] for row in rows) / len(rows)


def share(rows: list[dict], is_hit) -> str:
    hits = sum(is_hit(row) for row in rows)
    return f"{hits / len(rows):.0%} ({hits}/{len(rows)})" if rows else "–"


SETS = [("two-way", lambda row: row["sheet"] == "binary"), ("five-way", lambda row: row["sheet"] == "fiveway"),
        ("union", lambda row: True)]


def agreement_table(rows: list[dict]) -> list[str]:
    lines = ["| answers | model scrambles | n | classifier | judge | one constant label | experts agree | experts' kappa |",
             "|---|---|---|---|---|---|---|---|"]
    for set_name, in_set in SETS:
        for kept_name, keep in (("with", lambda row: True), ("without", lambda row: not row["scrambles"])):
            selected = [row for row in rows if in_set(row) and keep(row)]
            n_pairs, experts_agree, kappa = expert_agreement(selected)
            lines.append(f"| {set_name} | {kept_name} | {len(selected)} | {agreement(selected, 'classifier'):.0%} | "
                         f"{agreement(selected, 'judge'):.0%} | {constant_agreement(selected):.0%} | "
                         f"{experts_agree:.0%} (of {n_pairs}) | {kappa:.2f} |")
    return lines


def steering_table(rows: list[dict]) -> list[str]:
    lines = ["| target | n | experts say target | classifier says target | judge says target | experts say any aphasia |"
             " n without scrambles | experts say target, without scrambles |",
             "|---|---|---|---|---|---|---|---|"]
    for target in TARGET_LABEL.values():
        selected = [row for row in rows if row["target"] == target]
        clean = [row for row in selected if not row["scrambles"]]
        lines.append(f"| {target} | {len(selected)} | {share(selected, lambda row: row['target'] in row['experts'])} | "
                     f"{share(selected, lambda row: row['classifier'] == row['target'])} | "
                     f"{share(selected, lambda row: row['judge'] == row['target'])} | "
                     f"{share(selected, lambda row: bool(APHASIA_CLASSES & set(row['experts'])) or 'aphasia' in row['experts'])} | "
                     f"{len(clean)} | {share(clean, lambda row: row['target'] in row['experts'])} |")
    return lines


def main():
    with open(FOLDER / "human_2_own_tab_labels.csv", newline="") as handle:
        own_tab = {record["id"]: record["human_label_2_own_tab"] for record in csv.DictReader(handle)}
    # NOTE: the own-tab labels are the main ones because they are the second expert's own
    # pass; the labelled sheets' human_label_2 already follows the first expert wherever the two disagreed, and
    # agreement measured on it would be partly agreement by construction.
    rows = load_rows(own_tab)
    rows_sheet = load_rows()

    # The five-way sheet's annotator_1 is the classifier's most probable class, so the two must coincide there.
    assert all(row["classifier"] == row["classifier_class"] for row in rows if row["sheet"] == "fiveway")

    lines = ["# Human labels of the steered answers", "",
             "Written by `src/fiveway/annotation_agreement.py` from the labelled annotation sheets "
             "(`results/fiveway/annotation/annotation_{binary,fiveway}_labelled.csv`). Two experts labelled "
             "every answer; the classifier is `annotator_1`, the LLM judge (DeepSeek) `annotator_2`. "
             "**An annotator agrees with the experts when its label matches either expert's label**, and an "
             "answer reaches its steering target when either expert labels it so.", ""]
    for sheet in HUMAN_COLUMNS:
        sheet_rows = [row for row in rows if row["sheet"] == sheet]
        lines.append(f"- {sheet}: {len(sheet_rows)} answers, {sum(row['scrambles'] for row in sheet_rows)} of "
                     "them model scrambles" + (" (by the experts' comments, column `model_scrambles_comment`)"
                                               if sheet == "binary" else " (labelled so by either expert)"))
    differ = [sheet_row for sheet_row, row in zip(rows_sheet, rows) if sheet_row["experts"] != row["experts"]]
    copied = [sheet_row for sheet_row in differ if sheet_row["experts"][0] == sheet_row["experts"][1]]
    one_label = [row for row in rows if len(row["experts"]) == 1]
    lines += ["", "**The second expert's labels** are those of their own tab of the annotation sheet "
              "(`results/fiveway/annotation/human_2_own_tab_labels.csv`), not the labelled sheets' `human_label_2`. "
              f"The two differ on {len(differ)} answers ({sum(row['sheet'] == 'binary' for row in differ)} binary, "
              f"{sum(row['sheet'] == 'fiveway' for row in differ)} five-way), and on {len(copied)} of them the "
              "labelled sheets hold the first expert's label: labels set after the experts compared notes, which "
              f"would make the experts' agreement look higher than it is. {len(one_label)} answers have no own-tab "
              "label from the second expert (empty or \"???\") and are compared with the first expert alone. The "
              "last section repeats the main tables with the labelled sheets' `human_label_2`.", ""]

    lines += ["## 1. Classifier and judge against the experts", "",
              "Agreement = share of answers where the label matches either expert. *One constant label*: the "
              "best agreement of a labeller that gives every answer of a sheet the same label. *Without*: the "
              "model-scrambles answers removed. *Union*: the two sheets together, each answer judged in its "
              "own label set.", ""]
    lines += agreement_table(rows)
    lines += ["", "**model_scrambles precision | recall** against the experts' model-scrambles answers. On the "
              "binary sheet the classifier's prediction is its most probable class (the sheet shows only "
              "aphasia or not); the judge was asked yes or no there, so it has no model_scrambles prediction.", "",
              "| answers | classifier precision | classifier recall | judge precision | judge recall |",
              "|---|---|---|---|---|"]
    for set_name, in_set in SETS:
        selected = [row for row in rows if in_set(row)]
        five_way = [row for row in selected if row["sheet"] == "fiveway"]
        judge_cells = (precision_recall(five_way, lambda row: row["judge"] == SCRAMBLES)
                       if set_name == "five-way" else "– | –")
        lines.append(f"| {set_name} | {precision_recall(selected, lambda row: row['classifier_class'] == SCRAMBLES)} | "
                     f"{judge_cells} |")
    lines += ["", "Every five-way class the same way (precision: the share of an annotator's calls of the class that "
              "either expert shares; recall: the share of the answers either expert puts in the class that the "
              "annotator finds):", "",
              "| class | classifier precision | classifier recall | judge precision | judge recall |",
              "|---|---|---|---|---|"]
    five_way = [row for row in rows if row["sheet"] == "fiveway"]
    for label in ("control", "conduction/anomic", "broca", "wernicke", SCRAMBLES):
        # NOTE: precision_recall compares with the answers flagged as model scrambles, so
        # each class gets its own flag here: either expert gave the answer this class.
        flagged = [dict(row, scrambles=label in row["experts"]) for row in five_way]
        lines.append(f"| {label} | {precision_recall(flagged, lambda row: row['classifier'] == label)} | "
                     f"{precision_recall(flagged, lambda row: row['judge'] == label)} |")

    lines += ["", "## 2. Does steering reach its target?", "",
              "Share of steered answers that either expert labels with the steering target (aphasia for A, "
              "broca for B, wernicke for C). *Any aphasia*: either expert gives an aphasia label of any type.", ""]
    lines += steering_table(rows)
    lines += ["", "By model, without the model-scrambles answers (with them in brackets):", "",
              "| model | aphasia | broca | wernicke |", "|---|---|---|---|"]
    for model in dict.fromkeys(row["model"] for row in rows):
        cells = []
        for target in TARGET_LABEL.values():
            selected = [row for row in rows if row["model"] == model and row["target"] == target]
            clean = [row for row in selected if not row["scrambles"]]
            cells.append(f"{share(clean, lambda row: row['target'] in row['experts'])} "
                         f"[{share(selected, lambda row: row['target'] in row['experts'])}]")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", "On the prompt whose answers chose the layer and strength (the vector's own task: Cinderella "
              "or sandwich) against the six prompts the setting never saw, without the model-scrambles answers "
              "(with them in brackets):", "",
              "| target | own prompt | other prompts |", "|---|---|---|"]
    for target in TARGET_LABEL.values():
        cells = []
        for on_own_prompt in (True, False):
            selected = [row for row in rows if row["target"] == target and (row["prompt"] == row["source"]) == on_own_prompt]
            clean = [row for row in selected if not row["scrambles"]]
            cells.append(f"{share(clean, lambda row: row['target'] in row['experts'])} "
                         f"[{share(selected, lambda row: row['target'] in row['experts'])}]")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")

    lines += ["", "## With the labelled sheets' human_label_2", "",
              "The main tables again, with the second expert's labels as in the labelled sheets (after the experts "
              "compared notes). The binary model-scrambles answers stay those of the comments; a five-way answer "
              "is model scrambles when either expert's label says so.", ""]
    lines += agreement_table(rows_sheet)
    lines += [""] + steering_table(rows_sheet)

    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
