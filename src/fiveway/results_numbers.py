"""Every number of the paper's results section (Section 5), computed from the stored results in one place.

The lesions are the grown-heads sets (Head-grown, the paper's CLAMP) of all 11 models that reach their target by
the final-answer rule (scorer.works_finally). The script prints, under one heading per paragraph of Section 5:

    lesions          how many settings the lesions reach, by target
    beyond           what the lesions do to the TAB, MMLU and the other narrative (grown_summary.load_sets' checks)
    patching         activation patching of the lesions (patching.py --source grown_heads)
    overlap          the heads that the lesions for different targets share (subtype_overlap.py)
    sweep            where the Broca and Wernicke steering vectors make the classifier pass 0.5 (steer_generate.py)
    cosine           cos(Broca vector, Wernicke vector) against shuffled labels (subtype_cosine.py)
    experts          the experts' labels of the steered answers (annotation_agreement.py, with the second
                     expert's own-tab labels)
    new prompts      how often the classifier labels answers to new prompts with the target: steered answers
                     against lesioned ones
    steering back    removing or pushing against a lesion's own direction, and random directions (grown_rescue.py)

Usage (CPU, seconds):
    python src/fiveway/results_numbers.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.annotation_agreement import FOLDER, load_rows  # noqa: E402
from fiveway.grown_summary import EM, GROWN, RESCUE, load_sets, patching_counts  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import CLASSES, OBJECTIVES, SUCCESS_THRESHOLD, works_finally  # noqa: E402

SCRAMBLES = CLASSES.index("Model_scrambles")
CONTROL = CLASSES.index("Control")
STUDY_MODELS = ["gpt2-xl", "llama2-7b-chat", "llama2-13b-chat", "pythia-1b", "pythia-1.4b", "pythia-2.8b",
                "pythia-6.9b", "qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b"]
TASKS = ("cinderella", "sandwich")
SHORT_ANSWER = ("word_comprehension", "sentence_comprehension", "repetition")


def heading(title: str):
    print(f"\n=== {title}")


def top_class(probabilities) -> str:
    return CLASSES[int(np.argmax(probabilities))]


def bench_of(row: dict) -> tuple:
    """(check file, condition name in it) of one lesion, as grown_summary.load_sets reads them."""
    if row["source"] == "grown_heads":
        return GROWN / "benchmarks" / f"{row['model']}_{row['task']}_classifier.json", f"grown_{row['objective']}"
    return EM / "benchmarks" / f"{row['model']}_{row['task']}.json", f"2b_heads_grown_{row['objective']}"


def main():
    lesions = [r for r in load_sets() if r["kind"] == "classifier" and r["model"] in STUDY_MODELS]
    working = [r for r in lesions if r["works"]]

    heading("lesions")
    print(f"{len(working)} of {len(lesions)} settings; by target "
          f"{ {t: f'{sum(r['objective'] == t for r in working)}/{sum(r['objective'] == t for r in lesions)}' for t in 'ABC'} }")
    print(f"heads per lesion: median {statistics.median(len(r['heads']) for r in working)}, "
          f"range {min(len(r['heads']) for r in working)}-{max(len(r['heads']) for r in working)}")

    heading("beyond: the lesions on new prompts")
    checks = [r["checks"] for r in working]
    more_features = sum(c["judge_features"] > c["judge_features_clean"] for c in checks)
    print(f"TAB judge features per connected-text answer: intact median {statistics.median(c['judge_features_clean'] for c in checks):.1f}, "
          f"lesioned {statistics.median(c['judge_features'] for c in checks):.1f}; more features than intact in "
          f"{more_features} of {len(checks)} lesions")
    intact_correct, lesioned_correct = [], []
    other_reached, tab_reached, tab_answers_reached, tab_answers = 0, 0, 0, 0
    for row in working:
        path, name = bench_of(row)
        bench = json.loads(path.read_text())
        clean = bench["summary"]["tab"]["clean"]
        lesioned = bench["summary"]["tab"][name]
        intact_correct.append(sum(clean[s]["correct"] for s in SHORT_ANSWER))
        lesioned_correct.append(sum(lesioned[s]["correct"] for s in SHORT_ANSWER))
        other = bench["other_task"].get(bench["conditions"][name])
        target = row["objective"]
        if other and other["valid"] and other["probabilities"]:
            value = sum(other["probabilities"][i] for i in OBJECTIVES[target])
            other_reached += works_finally(target, value, other["probabilities"])
        means = lesioned["connected_text"]["mean_objectives"]
        tab_reached += means is not None and means[target] > SUCCESS_THRESHOLD
        for item in bench["tab"]:
            if item["subtest"] != "connected_text":
                continue
            response = item["responses"][name]
            if response["probabilities"] is not None:
                tab_answers += 1
                tab_answers_reached += works_finally(target, response["objectives"][target], response["probabilities"])
    print(f"short-answer items right, of 15: intact median {statistics.median(intact_correct)}, lesioned median "
          f"{statistics.median(lesioned_correct)}; {sum(i - l >= 2 for i, l in zip(intact_correct, lesioned_correct))} "
          f"of {len(working)} lesions lose at least 2")
    print(f"MMLU drop, nats: median {statistics.median(c['mmlu_drop'] for c in checks):.2f}")
    print(f"the lesion's target on the other narrative: reached by {other_reached} of {len(working)} lesions")
    print(f"the lesion's target on TAB connected text: mean over the five answers above 0.5 for {tab_reached} lesions; "
          f"{tab_answers_reached} of {tab_answers} scored answers reach it")
    # The same answers split by target: a type (Broca, Wernicke) or aphasia in general.
    for label, targets in (("Broca and Wernicke lesions", "BC"), ("aphasia lesions", "A")):
        other_hits, other_n, tab_hits, tab_n = 0, 0, 0, 0
        for row in working:
            if row["objective"] not in targets:
                continue
            path, name = bench_of(row)
            bench = json.loads(path.read_text())
            target = row["objective"]
            other = bench["other_task"].get(bench["conditions"][name])
            if other and other["valid"] and other["probabilities"]:
                other_n += 1
                other_hits += works_finally(target, sum(other["probabilities"][i] for i in OBJECTIVES[target]), other["probabilities"])
            for item in bench["tab"]:
                response = item["responses"][name]
                if item["subtest"] == "connected_text" and response["probabilities"] is not None:
                    tab_n += 1
                    tab_hits += works_finally(target, response["objectives"][target], response["probabilities"])
        print(f"  {label}: other narrative {other_hits} of {other_n} valid answers; TAB connected text {tab_hits} of {tab_n}; "
              f"together {other_hits + tab_hits} of {other_n + tab_n}")

    heading("patching")
    counts = [patching_counts(GROWN / "patching" / f"{r['model']}_{r['task']}_{r['objective']}.json") for r in working]
    print(f"restore all (the heads knocked out only while the answer is written) undoes {sum(not c['restore_all_works'] for c in counts)} of {len(counts)}")
    print(f"every head needed (leaving any one out undoes the lesion): {sum(c['needed'] == c['n'] for c in counts)} of {len(counts)}")
    several = [c for c in counts if c["n"] > 1]
    print(f"some single head works alone: {sum(c['alone'] > 0 for c in several)} of {len(several)} lesions with 2 or more heads")
    print(f"single-head restores that undo the lesion: median share {statistics.median(c['restore_undoes'] / c['n'] for c in counts):.2f}")
    after_undo = Counter()
    for row in working:
        document = json.loads((GROWN / "patching" / f"{row['model']}_{row['task']}_{row['objective']}.json").read_text())
        for condition in document["conditions"]:
            if condition["kind"] != "restore" or condition["probabilities"] is None:
                continue
            if not works_finally(row["objective"], condition["value"], condition["probabilities"]):
                after_undo[top_class(condition["probabilities"])] += 1
    print(f"top class of the answers after an undoing single-head restore: {dict(after_undo)} "
          f"(model scrambles {after_undo['Model_scrambles']} of {sum(after_undo.values())})")

    heading("overlap: heads shared by the lesions for two targets (subtype_overlap.py)")
    overlap = json.loads((RESULTS / "report" / "subtype_overlap.json").read_text())
    for pair in ("AB", "AC", "BC", None):
        selected = [r for r in overlap if pair is None or r["pair"] == pair]
        print(f"{pair or 'all':4s}: {len(selected)} pairs; median share of the smaller set "
              f"{statistics.median(r['share'] for r in selected):.2f}; shared {sum(r['shared'] for r in selected)} "
              f"against {sum(r['chance'] for r in selected):.1f} by chance "
              f"({sum(r['shared'] for r in selected) / sum(r['chance'] for r in selected):.1f} times); "
              f"p < 0.01 in {sum(r['p'] < 0.01 for r in selected)}")

    heading("sweep: layers where a vector makes the classifier pass 0.5 on its own task")
    for target in ("A", "B", "C"):
        layers, combos = 0, 0
        n_layers = 0
        for model in STUDY_MODELS:
            for task in TASKS:
                sweep = json.loads((RESULTS / "steering" / model / f"sweep_{task}.json").read_text())
                entries = [e for e in sweep["entries"] if e["vector"] == target and e["strength"] > 0
                           and e["objectives"] is not None]
                by_layer = {}
                for entry in entries:
                    by_layer.setdefault(entry["layer"], False)
                    by_layer[entry["layer"]] |= entry["objectives"][target] > SUCCESS_THRESHOLD
                n_layers += len({e["layer"] for e in sweep["entries"] if e["vector"] == target})
                layers += sum(by_layer.values())
                combos += any(by_layer.values())
        print(f"{target}: works at {layers} of {n_layers} layers; at some layer in {combos} of 22 model-task pairs")

    heading("cosine: Broca against Wernicke vector (subtype_cosine.py)")
    path = RESULTS / "steering" / "subtype_cosine.json"
    if path.exists():
        cosine = json.loads(path.read_text())
        medians = {key: float(np.median(row["observed"])) for key, row in cosine.items()}
        nulls = {key: float(np.median(row["null_mean"])) for key, row in cosine.items()}
        below = sum(int((np.array(row["observed"]) < np.array(row["null_low"])).sum()) for row in cosine.values())
        total = sum(len(row["observed"]) for row in cosine.values())
        print(f"median cos over layers, per model and task: {min(medians.values()):.2f} to {max(medians.values()):.2f} "
              f"(median {statistics.median(medians.values()):.2f}); shuffled labels {min(nulls.values()):.2f} to "
              f"{max(nulls.values()):.2f}; below the shuffled 2.5th percentile at {below} of {total} layers")
        specific = {key: float(np.median(row["observed_specific"])) for key, row in cosine.items()}
        print(f"specific parts (A projected out): median cos {min(specific.values()):.2f} to {max(specific.values()):.2f}")
    else:
        print("not yet computed")
    # A yardstick for "alike": the same type's vectors from the two tasks. If Broca and Wernicke were separate
    # directions, a Broca vector would be closer to the other task's Broca vector than to its own task's Wernicke.
    within, across_broca, across_wernicke = {"cinderella": [], "sandwich": []}, [], []
    for model in STUDY_MODELS:
        vectors = {task: np.load(RESULTS / "steering" / model / f"vectors_{task}.npz") for task in TASKS}

        def median_cosine(u, v):
            # NOTE: u, v: (n_layers, d_model); the cosine per layer, then the median over layers.
            u, v = u.astype(np.float64), v.astype(np.float64)
            return float(np.median((u * v).sum(-1) / np.linalg.norm(u, axis=-1) / np.linalg.norm(v, axis=-1)))

        for task in TASKS:
            within[task].append(median_cosine(vectors[task]["B"], vectors[task]["C"]))
        across_broca.append(median_cosine(vectors["cinderella"]["B"], vectors["sandwich"]["B"]))
        across_wernicke.append(median_cosine(vectors["cinderella"]["C"], vectors["sandwich"]["C"]))
    print(f"median cos over layers, median over models: Broca and Wernicke within Cinderella "
          f"{statistics.median(within['cinderella']):.2f}, within sandwich {statistics.median(within['sandwich']):.2f}; "
          f"Broca across tasks {statistics.median(across_broca):.2f}, Wernicke across tasks "
          f"{statistics.median(across_wernicke):.2f}")
    print(f"ranges over models: within {min(within['cinderella'] + within['sandwich']):.2f} to "
          f"{max(within['cinderella'] + within['sandwich']):.2f}; across {min(across_broca + across_wernicke):.2f} to "
          f"{max(across_broca + across_wernicke):.2f}; within above both across in "
          f"{sum(min(w1, w2) > max(b, c) for w1, w2, b, c in zip(within['cinderella'], within['sandwich'], across_broca, across_wernicke))} of {len(STUDY_MODELS)} models")

    heading("experts: answers steered toward Broca or Wernicke (five-way sheet, own-tab labels)")
    # The second expert's labels from their own tab of the sheet, as annotation_agreement.py reads them.
    with open(FOLDER / "human_2_own_tab_labels.csv", newline="") as handle:
        own_tab = {record["id"]: record["human_label_2_own_tab"] for record in csv.DictReader(handle)}
    rows = [r for r in load_rows(own_tab) if r["sheet"] == "fiveway"]
    for target in ("broca", "wernicke"):
        for where, selected in (("all prompts", [r for r in rows if r["target"] == target]),
                                ("source task", [r for r in rows if r["target"] == target and r["source"] == r["prompt"]]),
                                ("other prompts", [r for r in rows if r["target"] == target and r["source"] != r["prompt"]])):
            clean = [r for r in selected if not r["scrambles"]]
            hits = sum(target in r["experts"] for r in clean)
            aphasic = sum(any(label in ("broca", "wernicke", "conduction/anomic") for label in r["experts"]) for r in clean)
            other_type = "wernicke" if target == "broca" else "broca"
            crossed = sum(other_type in r["experts"] for r in clean)
            print(f"{target:9s} {where:13s}: targeted type {hits}/{len(clean)} = {hits / len(clean):.0%} of the answers "
                  f"that are not model scrambles; some aphasia {aphasic}/{len(clean)}; labeled {other_type} by an expert "
                  f"{crossed}/{len(clean)}; model scrambles {len(selected) - len(clean)} of {len(selected)}")

    print("on the six new prompts, all answers including model scrambles: share that either expert labels")
    for target in ("broca", "wernicke"):
        selected = [r for r in rows if r["target"] == target and r["source"] != r["prompt"]]
        print(f"  steered toward {target}: broca {sum('broca' in r['experts'] for r in selected)}/{len(selected)}, "
              f"wernicke {sum('wernicke' in r['experts'] for r in selected)}/{len(selected)}")

    heading("new prompts: how often the classifier labels an answer to a new prompt with its target")
    steered = [r for r in rows if r["source"] != r["prompt"]]
    print(f"steered answers (Broca, Wernicke; 7 models): classifier's label is the target for "
          f"{sum(r['classifier'] == r['target'] for r in steered)} of {len(steered)}")
    # The same count for the Broca and Wernicke lesions of the same 7 models: their answers to the other narrative
    # and to the five TAB connected-text prompts, labeled with the target when it is the classifier's top class.
    annotated = {r["model"] for r in rows}
    hits, answers = 0, 0
    for row in working:
        if row["objective"] not in ("B", "C") or row["model"] not in annotated:
            continue
        path, name = bench_of(row)
        bench = json.loads(path.read_text())
        target_class = CLASSES.index("Broca" if row["objective"] == "B" else "Wernicke")
        new_answers = [bench["other_task"].get(bench["conditions"][name])]
        new_answers += [item["responses"][name] for item in bench["tab"] if item["subtest"] == "connected_text"]
        for answer in new_answers:
            if answer and answer.get("probabilities") is not None and answer.get("valid", True):
                answers += 1
                hits += int(np.argmax(answer["probabilities"])) == target_class
    print(f"lesioned answers (Broca, Wernicke lesions of the same 7 models; other narrative and TAB connected text): "
          f"classifier's top class is the target for {hits} of {answers}")

    heading("agreement: where the classifier's lead over the judge comes from (both sheets, own-tab labels)")
    every_row = load_rows(own_tab)
    scrambles = [r for r in every_row if r["scrambles"]]
    others = [r for r in every_row if not r["scrambles"]]
    agree = lambda selected, who: sum(r[who] in r["experts"] for r in selected)
    print(f"model-scramble answers ({len(scrambles)}): classifier agrees with an expert on {agree(scrambles, 'classifier')}, "
          f"judge on {agree(scrambles, 'judge')}; other answers ({len(others)}): classifier {agree(others, 'classifier')}, "
          f"judge {agree(others, 'judge')}")
    fiveway_scrambles = [r for r in scrambles if r["sheet"] == "fiveway"]
    print(f"five-way model scrambles named as such: classifier {sum(r['classifier'] == 'model_scrambles' for r in fiveway_scrambles)}"
          f" of {len(fiveway_scrambles)}, judge {sum(r['judge'] == 'model_scrambles' for r in fiveway_scrambles)}; "
          f"both sheets by the classifier's top class: {sum(r['classifier_class'] == 'model_scrambles' for r in scrambles)} of {len(scrambles)}")
    pairs = [r for r in every_row if len(r["experts"]) == 2]
    print(f"experts agree on {sum(a == b for a, b in (r['experts'] for r in pairs))} of the {len(pairs)} answers that both labeled")

    heading("type-specific vectors (the aphasia direction projected out), steer_generate.py's subtype sweeps")
    for vector, target in (("B_specific", "B"), ("C_specific", "C")):
        layers, n_layers, pairs_working = 0, 0, 0
        for model in STUDY_MODELS:
            for task in TASKS:
                sweep = json.loads((RESULTS / "steering" / model / f"subtype_{task}.json").read_text())
                by_layer = {}
                for entry in sweep["entries"]:
                    if entry["vector"] != vector or entry["strength"] <= 0:
                        continue
                    by_layer.setdefault(entry["layer"], False)
                    by_layer[entry["layer"]] |= entry["objectives"] is not None and entry["objectives"][target] > SUCCESS_THRESHOLD
                layers += sum(by_layer.values())
                n_layers += len(by_layer)
                pairs_working += any(by_layer.values())
        print(f"{vector}: passes 0.5 for its type at {layers} of {n_layers} layers; in {pairs_working} of 22 pairs")
    along_a = {"B": [], "C": []}
    for model in STUDY_MODELS:
        for task in TASKS:
            vectors = np.load(RESULTS / "steering" / model / f"vectors_{task}.npz")
            a = vectors["A"].astype(np.float64)
            for name in ("B", "C"):
                v = vectors[name].astype(np.float64)
                along_a[name] += ((a * v).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(v, axis=-1)).tolist()
    for name, values in along_a.items():
        print(f"cos({name}, aphasia vector) over models, tasks, and layers: {min(values):.3f} to {max(values):.3f}, "
              f"median {statistics.median(values):.3f}")

    heading("steering back (grown_rescue.py)")
    own_removed, random_all, random_any, own_scrambles = 0, 0, 0, 0
    random_each = [0, 0, 0]
    push_own, push_random, control_own, control_random = [], [], 0, 0
    for row in working:
        document = json.loads((RESCUE / row["model"] / f"rescue_{row['task']}.json").read_text())
        entries = [e for e in document["entries"] if e["set_objective"] == row["objective"]]
        target = row["objective"]

        def reaches(entry):
            return entry["objectives"] is not None and works_finally(target, entry["objectives"][target], entry["probabilities"])

        def undoes(entry):
            # NOTE: an answer too short to score undoes nothing.
            return entry["objectives"] is not None and not reaches(entry)

        by_name = {e["intervention"]: e for e in entries if e["intervention"] != "knockout + steer"}
        own = by_name[f"knockout, {target} removed"]
        own_removed += undoes(own)
        own_scrambles += undoes(own) and top_class(own["probabilities"]) == "Model_scrambles"
        randoms = [by_name[f"knockout, random{i} removed"] for i in range(3)]
        random_all += all(undoes(e) for e in randoms)
        random_any += any(undoes(e) for e in randoms)
        for i, entry in enumerate(randoms):
            random_each[i] += undoes(entry)
        pushes = [e for e in entries if e["intervention"] == "knockout + steer"]
        own_pushes = [e for e in pushes if e["vector"] == target]
        random_pushes = [e for e in pushes if e["vector"].startswith("random")]
        push_own += [undoes(e) for e in own_pushes]
        push_random += [undoes(e) for e in random_pushes]
        control_own += any(e["objectives"] is not None and top_class(e["probabilities"]) == "Control" for e in own_pushes)
        control_random += any(e["objectives"] is not None and top_class(e["probabilities"]) == "Control" for e in random_pushes)
    n = len(working)
    print(f"own direction removed from every layer undoes {own_removed} of {n}, {own_scrambles} of them as model scrambles")
    print(f"random directions removed: each alone undoes {random_each} of {n}; all three undo {random_all}; at least one {random_any}")
    print(f"pushes against the direction: own undo {np.mean(push_own):.2f} of {len(push_own)}, random {np.mean(push_random):.2f} of {len(push_random)}")
    print(f"control speech from some push: own direction in {control_own} of {n} lesions, random in {control_random}")


if __name__ == "__main__":
    main()
