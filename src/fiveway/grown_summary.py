"""Grown heads on one page: can cos-sim replace the classifier, does steering undo the knockouts, and Qwen3.5.

Reads
    results/fiveway/em_attention/    the classifier's grown-heads sets of the seven older models (space 2b)
    results/fiveway/grown_heads/     grown_heads.py: the cos-sim sets of every model, and the classifier's
                                     sets of the models that have no em_attention.py results (Qwen3.5)
    results/fiveway/steering_grown/  grown_rescue.py: steering against the classifier's grown sets
with their benchmarks/ and tab_judge/ files, and writes results/fiveway/report/grown_heads.md.

Usage:
    python src/fiveway/grown_summary.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import OBJECTIVES, SUCCESS_THRESHOLD, works_finally  # noqa: E402

GROWN = RESULTS / "grown_heads"
EM = RESULTS / "em_attention"
RESCUE = RESULTS / "steering_grown"
MODELS_NAME = {"gpt2-xl": "GPT-2 XL", "llama2-7b-chat": "Llama-2-7B-chat", "llama2-13b-chat": "Llama-2-13B-chat",
               "pythia-1b": "Pythia-1B", "pythia-1.4b": "Pythia-1.4B", "pythia-2.8b": "Pythia-2.8B",
               "pythia-6.9b": "Pythia-6.9B", "qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3.5-2b": "Qwen3.5-2B",
               "qwen3.5-4b": "Qwen3.5-4B", "qwen3.5-9b": "Qwen3.5-9B"}


def number(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def checks(bench: dict | None, name: str, objective: str, judge_path: Path) -> dict:
    """What a set costs and carries: MMLU, the TAB subtests, TAB connected text, the other task, the judge."""
    if not bench or name not in bench["conditions"] or not bench["summary"].get("tab"):
        return {}
    tab, clean = bench["summary"]["tab"][name], bench["summary"]["tab"]["clean"]
    mmlu = bench["summary"].get("mmlu", {})
    connected = tab["connected_text"]["mean_objectives"]
    other = bench["other_task"].get(bench["conditions"][name])
    result = {
        # NOTE: only once both subjects are complete (100 and 237
        # questions): a run still in its MMLU step has part-way means.
        "mmlu_drop": (statistics.mean(mmlu[s]["clean"]["mean_log_probability_correct"]
                                      - mmlu[s][name]["mean_log_probability_correct"] for s in mmlu)
                      if all(mmlu.get(s, {}).get(name, {}).get("n") == size
                             for s, size in (("global_facts", 100), ("high_school_world_history", 237))) else None),
        "lost": sum(clean[sub]["correct"] - tab[sub]["correct"]
                    for sub in ("word_comprehension", "sentence_comprehension", "repetition")),
        "tab_connected": None if connected is None else connected[objective],
        "other": (sum(other["probabilities"][i] for i in OBJECTIVES[objective])
                  if other and other["valid"] and other["probabilities"] else None)}
    # The TAB paper's own scoring of connected text: the LLM judge's 19 aphasia features (llm_judge.py --tab),
    # as the mean number present per answer and the share of answers judged an overall communication impairment.
    if judge_path.exists():
        summary = json.loads(judge_path.read_text())["summary"]
        judged, clean_judged = summary.get(name), summary.get("clean")
        result["judge_features"] = None if judged is None else judged["mean_features_present"]
        result["judge_impaired"] = None if judged is None else judged["share_impaired"]
        result["judge_features_clean"] = None if clean_judged is None else clean_judged["mean_features_present"]
    return result


def load_sets() -> list:
    """One row per model, task, objective and kind (classifier, cosine): the reported grown set and its checks."""
    rows = []
    for model in MODELS:
        for task in TASKS:
            for kind in ("classifier", "cosine"):
                path = GROWN / f"{model}_{task}_{kind}.json"
                if path.exists() and "reported" in json.loads(path.read_text()):
                    results = json.loads(path.read_text())
                    bench_path = GROWN / "benchmarks" / f"{model}_{task}_{kind}.json"
                    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
                    for objective, row in results["reported"].items():
                        entry = results["objectives"][objective]
                        rows.append({"model": model, "task": task, "objective": objective, "kind": kind,
                                     "source": "grown_heads", **row,
                                     # The classifier's sets count by the final-answer rule (scorer.works_finally);
                                     # the cos-sim sets keep their own criterion.
                                     **({"works": works_finally(objective, row["value"], row["probabilities"])}
                                        if kind == "classifier" else {}),
                                     "baseline_value": entry["baseline_value"],
                                     "works_unablated": entry["works_unablated"],
                                     "layer_types": results["geometry"].get("layer_types"),
                                     "seconds": entry.get("seconds"),
                                     "checks": checks(bench, f"grown_{objective}", objective,
                                                      GROWN / "tab_judge" / f"{model}_{task}_{kind}.json")})
                elif kind == "classifier" and (EM / f"{model}_{task}.json").exists():
                    results = json.loads((EM / f"{model}_{task}.json").read_text())
                    if "reported" not in results:
                        continue
                    bench_path = EM / "benchmarks" / f"{model}_{task}.json"
                    bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
                    for objective, spaces in results["reported"].items():
                        row = spaces["2b_heads_grown"]
                        own = results["objectives"][objective]["2b_heads_grown"]
                        found = own["pruned"]["circuit"] if row["works"] else own["best"]
                        rows.append({"model": model, "task": task, "objective": objective, "kind": kind,
                                     "source": "em_attention", **row, "classifier_value": row["value"],
                                     "works": works_finally(objective, row["value"], row["probabilities"]),
                                     "cosine_value": None if found["cosine"] is None else found["cosine"][objective],
                                     "baseline_value": None, "works_unablated": False,
                                     "layer_types": results["geometry"].get("layer_types"),
                                     "seconds": own.get("seconds"),
                                     "checks": checks(bench, f"2b_heads_grown_{objective}", objective,
                                                      EM / "tab_judge" / f"{model}_{task}.json")})
    return rows


PATCHING_FOLDERS = {"grown heads": GROWN / "patching", "search": RESULTS / "patching",
                    "em_search": RESULTS / "em_search" / "patching"}


def works(value) -> bool:
    """Whether an objective value works: the answer counts (the value exists) and it is above the threshold."""
    return value is not None and value > SUCCESS_THRESHOLD


def patching_counts(path: Path) -> dict:
    """One patching.py results file in numbers: of the set's components, how many undo the knockout when restored or
    left out, and how many work alone."""
    document = json.loads(path.read_text())
    by_name = {condition["name"]: condition for condition in document["conditions"]}
    objective = document["config"]["objective"]
    names = [component["name"] for component in document["components"]]

    def reaches(name):
        # NOTE: a patched answer still reaches the target by the final-answer rule
        # (scorer.works_finally), so a restore undoes a lesion when its answer no longer does.
        return works_finally(objective, by_name[name]["value"], by_name[name]["probabilities"])

    return {"model": document["config"]["model"], "task": document["config"]["task"], "objective": objective,
            "n": len(names), "corrupt": by_name["corrupt"]["value"], "has_restore_all": "restore_all" in by_name,
            "restore_all": by_name["restore_all"]["value"] if "restore_all" in by_name else None,
            "restore_all_works": reaches("restore_all") if "restore_all" in by_name else None,
            "restore_undoes": sum(not reaches(f"restore_{name}") for name in names),
            "needed": sum(not reaches(f"leave_one_out_{name}") for name in names),
            "alone": sum(reaches(f"alone_{name}") for name in names),
            "reproduced": all(document["checks"].values()) if document["checks"] else None}


SHORT_ANSWER = ("word_comprehension", "sentence_comprehension", "repetition")


def tab_patching_counts(path: Path) -> dict:
    """One benchmarks.py TAB file in numbers: short-answer items lost to the knockout, to restore all and to the best
    single restore, and the set's objective on connected text."""
    document = json.loads(path.read_text())
    config, tab = document["config"], document["summary"]["tab"]

    def lost(name):
        return sum(tab["clean"][subtest]["correct"] - tab[name][subtest]["correct"] for subtest in SHORT_ANSWER)

    def connected(name):
        means = tab[name]["connected_text"]["mean_objectives"]
        return None if means is None else means[config["objective"]]

    singles = [name for name in tab if name.startswith("restore_") and name != "restore_all"]
    return {"model": config["model"], "task": config["circuit_task"], "objective": config["objective"],
            "n": len(document["circuit"]["heads"]), "lost_corrupt": lost("corrupt"),
            "lost_restore_all": lost("restore_all"), "lost_best_single": min(lost(name) for name in singles),
            "connected": {name: connected(name) for name in ("clean", "corrupt", "restore_all")}}


def jaccard(a: list, b: list) -> float:
    a, b = {tuple(h) for h in a}, {tuple(h) for h in b}
    return len(a & b) / len(a | b) if a | b else 0.0


def cost_lines(label: str, selected: list) -> str:
    return (f"| {label} | {len(selected)} | {number(median([r['n_heads'] for r in selected]), 0)} | "
            f"{number(median([r['classifier_value'] for r in selected]))} | "
            f"{sum((r['classifier_value'] or 0) > SUCCESS_THRESHOLD for r in selected)} | "
            f"{number(median([r['checks'].get('mmlu_drop') for r in selected]))} | "
            f"{number(median([r['checks'].get('lost') for r in selected]), 1)} | "
            f"{number(median([r['checks'].get('tab_connected') for r in selected]))} | "
            f"{number(median([r['checks'].get('judge_features') for r in selected]), 1)} | "
            f"{number(median([r['checks'].get('judge_impaired') for r in selected]), 1)} |")


def main():
    rows = load_sets()
    index = {(r["model"], r["task"], r["objective"], r["kind"]): r for r in rows}
    models = [m for m in MODELS if any(r["model"] == m for r in rows)]
    lines = ["# Grown heads: the classifier against cos-sim, steering against the knockouts, and Qwen3.5", "",
             "Generated by `src/fiveway/grown_summary.py`. A grown-heads set: whole heads knocked out in the order "
             "of their single-head values, the top 1, 2, 3, ... together until the objective works, then pruned "
             "(em_attention.py's space 2b; grown_heads.py). Classifier: the five-way classifier's objective above "
             f"{SUCCESS_THRESHOLD}. Cos-sim: mean MiniLM cosine to the target group's human transcripts minus "
             "mean cosine to Control's, above 0 (the cos-sim ablation's objective).", ""]

    # 1. Cos-sim against the classifier.
    lines += ["## 1. Can cos-sim replace the classifier?", "",
              "Per model, task and objective (A, B, C): the classifier's set (heads; works or not) and the cos-sim "
              "set (heads; whether cos-sim calls it working, 'unablated' when the intact answer already does; the "
              "classifier's objective on its answer).", "",
              "| model | task | " + " | ".join(f"{o}: classifier | {o}: cos-sim" for o in "ABC") + " |",
              "|---|---|" + "---|---|" * 3]
    for model in models:
        for task in TASKS:
            cells = []
            for objective in "ABC":
                c, s = index.get((model, task, objective, "classifier")), index.get((model, task, objective, "cosine"))
                cells.append("…" if c is None else f"{c['n_heads']}h {'works' if c['works'] else 'fails'}")
                cells.append("…" if s is None else
                             f"{s['n_heads']}h {'unablated' if s['works_unablated'] else 'works' if s['works'] else 'fails'}"
                             f", classifier {number(s['classifier_value'])}")
            if any(cell != "…" for cell in cells):
                lines.append(f"| {MODELS_NAME.get(model, model)} | {task} | " + " | ".join(cells) + " |")

    cosine = [r for r in rows if r["kind"] == "cosine"]
    classifier = [r for r in rows if r["kind"] == "classifier"]
    paired = [(index[(r["model"], r["task"], r["objective"], "classifier")], r) for r in cosine
              if (r["model"], r["task"], r["objective"], "classifier") in index]
    lines += ["", "**Counts.**", "",
              "| | combinations | cos-sim works | of which the intact answer already works | classifier objective above "
              "0.5 on the cos-sim set's answer | classifier's own set works |", "|---|---|---|---|---|---|"]
    for task in TASKS + ["all"]:
        selected = [(c, s) for c, s in paired if task == "all" or s["task"] == task]
        lines.append(f"| {task} | {len(selected)} | {sum(s['works'] for _, s in selected)} | "
                     f"{sum(s['works_unablated'] for _, s in selected)} | "
                     f"{sum((s['classifier_value'] or 0) > SUCCESS_THRESHOLD for _, s in selected)} | "
                     f"{sum(c['works'] for c, _ in selected)} |")
    both = [(c, s) for c, s in paired if c["works"] and s["works"] and not s["works_unablated"]]
    agree = Counter(("both" if c["works"] and (s["classifier_value"] or 0) > SUCCESS_THRESHOLD else
                     "classifier only" if c["works"] else
                     "cos-sim only" if (s["classifier_value"] or 0) > SUCCESS_THRESHOLD else "neither")
                    for c, s in paired)
    lines += ["", f"Judged by the classifier, on the same combination: both sets aphasic {agree['both']}, only the "
              f"classifier's {agree['classifier only']}, only the cos-sim set's {agree['cos-sim only']}, neither "
              f"{agree['neither']}. Head overlap (Jaccard) where both searches work and the intact answer does not "
              f"already pass cos-sim: median {number(median([jaccard(c['heads'], s['heads']) for c, s in both]))} "
              f"over {len(both)} combinations.", "",
              "**Cost and carry-over** (sets that work under their own objective). Classifier objective: on the "
              "set's own task answer. MMLU: drop of the right answer's mean log-probability (nats). TAB lost: "
              "word, sentence and repetition items lost of 15. TAB connected text: mean classifier objective on "
              "the five held-out prompts. Judge: the TAB paper's LLM judge on the same five answers, mean aphasia "
              "features of 19 per answer and the share of answers judged an overall communication impairment.", "",
              "| sets | n | median heads | classifier objective, median | classifier objective above 0.5 | "
              "MMLU drop | TAB lost | TAB connected text | judge features | judge impaired |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines.append(cost_lines("classifier's grown sets", [r for r in classifier if r["works"]]))
    lines.append(cost_lines("cos-sim's grown sets", [r for r in cosine if r["works"]]))
    lines.append(cost_lines("cos-sim's, intact answer not already passing",
                            [r for r in cosine if r["works"] and not r["works_unablated"]]))

    # 2. Steering against the classifier's grown sets.
    lines += ["", "## 2. Does steering toward Control undo the grown-heads knockouts?", "",
              "grown_rescue.py, for every working classifier set, with the objective's own direction (A: aphasic "
              "minus Control; B: Broca minus Control; C: Wernicke minus Control) and three random directions of "
              "the same length. Removed: the direction projected out after every block; the knockout is undone "
              f"when the objective falls to {SUCCESS_THRESHOLD} or below. Pushes: the direction added at strengths "
              "-0.5, -1, -2, -4 (relative to the residual stream) after one of 8 layers; the share of the 32 "
              "pushes that undo the knockout.", "",
              "| model | task | objective | heads | knockout | own direction removed | random directions removed, "
              "undoing (of 3) | pushes undoing: own | pushes undoing: random (mean) |", "|---|---|---|---|---|---|---|---|---|"]
    rescue_rows = []
    for model in models:
        for task in TASKS:
            path = RESCUE / model / f"summary_{task}.json"
            if not path.exists():
                continue
            for objective, entry in json.loads(path.read_text())["objectives"].items():
                shares = entry["share_of_pushes_undoing"]
                random_share = statistics.mean(v for k, v in shares.items() if k.startswith("random")) if shares else None
                own_removed = entry["references"].get(f"knockout, {objective} removed")
                rescue_rows.append({"own_undoes": entry["removing_own_direction_undoes"],
                                    "random_undoing": entry["random_removals_undoing"],
                                    "own_share": shares.get(objective), "random_share": random_share})
                lines.append(f"| {MODELS_NAME.get(model, model)} | {task} | {objective} | {entry['n_heads']} | "
                             f"{number(entry['references'].get('knockout'))} | {number(own_removed)} | "
                             f"{entry['random_removals_undoing']} | {number(shares.get(objective))} | "
                             f"{number(random_share)} |")
    if rescue_rows:
        lines += ["", f"Over {len(rescue_rows)} knockouts: removing the own direction undoes "
                  f"{sum(bool(r['own_undoes']) for r in rescue_rows)}; at least one random removal undoes "
                  f"{sum(r['random_undoing'] > 0 for r in rescue_rows)} (all three: "
                  f"{sum(r['random_undoing'] == 3 for r in rescue_rows)}); the own direction is removed alone "
                  f"successfully where no random one is in "
                  f"{sum(bool(r['own_undoes']) and r['random_undoing'] == 0 for r in rescue_rows)}. Mean share of "
                  f"pushes undoing: own {number(statistics.mean(r['own_share'] for r in rescue_rows if r['own_share'] is not None))}, "
                  f"random {number(statistics.mean(r['random_share'] for r in rescue_rows if r['random_share'] is not None))}."]

    # 3. Which heads: full attention or linear attention (Qwen3.5).
    hybrid = [r for r in classifier + cosine if r["layer_types"] and any("linear" in t for t in r["layer_types"])]
    if hybrid:
        lines += ["", "## 3. Qwen3.5: softmax-attention heads and linear-attention heads in the grown sets", "",
                  "Every fourth Qwen3.5 layer is softmax attention (8 heads in 0.8B and 2B, 16 in 4B and 9B), the "
                  "others Gated DeltaNet linear attention (16 or 32 value heads, each with its own recurrent state). "
                  "Both kinds count as heads. Softmax share: of the set, and in brackets of all the model's heads.", "",
                  "| model | task | objective | kind | works | heads | softmax | linear-attention | softmax share |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in hybrid:
            linear = sum("linear" in r["layer_types"][layer] for layer, _ in r["heads"])
            heads_per_layer = json.loads((GROWN / f"{r['model']}_{r['task']}_{r['kind']}.json").read_text())["geometry"]["heads_per_layer"]
            model_share = sum(n for t, n in zip(r["layer_types"], heads_per_layer) if "linear" not in t) / sum(heads_per_layer)
            lines.append(f"| {MODELS_NAME.get(r['model'], r['model'])} | {r['task']} | {r['objective']} | {r['kind']} | "
                         f"{r['works']} | {r['n_heads']} | {r['n_heads'] - linear} | {linear} | "
                         f"{(r['n_heads'] - linear) / r['n_heads']:.0%} ({model_share:.0%}) |")
    # 4. Cinderella and sandwich sets: overlap, union, intersection, task-specific parts (task_overlap.py).
    overlap_paths = [RESULTS / "task_overlap" / f"{model}.json" for model in MODELS
                     if (RESULTS / "task_overlap" / f"{model}.json").exists()]
    if overlap_paths:
        lines += ["", "## 4. Cinderella and sandwich: shared heads, union, intersection, task-specific parts", "",
                  "task_overlap.py, one model per family. C: the Cinderella grown set, S: the sandwich set (classifier, "
                  "each found on its own task). Shared heads against the number expected by chance (|C| |S| / heads "
                  "in the model) and the hypergeometric p-value. Each set below is knocked out and answered on both "
                  "tasks (the objective on each answer; works above 0.5) and on the five TAB connected-text prompts "
                  "(mean objective: the other elicitation tasks), with MMLU's drop as its cost. Joint: the union pruned "
                  "while it works on both tasks at once.", "",
                  "| model | objective | C | S | shared | chance | p |", "|---|---|---|---|---|---|---|"]
        overlap_rows = []
        for path in overlap_paths:
            document = json.loads(path.read_text())
            model = document["model"]
            bench_path = RESULTS / "task_overlap" / "benchmarks" / f"{model}.json"
            bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
            for objective, entry in document["objectives"].items():
                if "skipped" in entry:
                    lines.append(f"| {MODELS_NAME.get(model, model)} | {objective} | skipped: {entry['skipped']} | | | | |")
                    continue
                sizes = entry["sizes"]
                lines.append(f"| {MODELS_NAME.get(model, model)} | {objective} | {sizes['cinderella']} | {sizes['sandwich']} | "
                             f"{sizes['shared']} | {entry['expected_shared_by_chance']:.1f} | {entry['p_value']:.2g} |")
                for condition, answers in entry.get("answers", {}).items():
                    found = checks(bench, f"{objective}_{condition}", objective,
                                   RESULTS / "task_overlap" / "tab_judge" / f"{model}.json")
                    overlap_rows.append((model, objective, condition, len(entry["sets"][condition]), answers, found))
        lines += ["", "TAB connected text, two scorers: the classifier (mean objective over the five answers) and the "
                  "TAB paper's LLM judge (mean aphasia features of 19 per answer, unablated in brackets; share of "
                  "answers judged an overall communication impairment).", "",
                  "| model | objective | set | heads | Cinderella | sandwich | TAB connected text: classifier | "
                  "judge features | judge impaired | TAB lost | MMLU drop |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for model, objective, condition, size, answers, found in overlap_rows:
            cells = [f"{number(answers[task]['value'])}{' works' if answers[task]['works'] else ''}" for task in TASKS]
            lines.append(f"| {MODELS_NAME.get(model, model)} | {objective} | {condition.replace('_', ' ')} | {size} | "
                         + " | ".join(cells) + f" | {number(found.get('tab_connected'))} | "
                         f"{number(found.get('judge_features'), 1)} ({number(found.get('judge_features_clean'), 1)}) | "
                         f"{number(found.get('judge_impaired'), 1)} | "
                         f"{number(found.get('lost'), 0)} | {number(found.get('mmlu_drop'))} |")

    # 5. Activation patching of the grown sets, against the search's and em_search's sets.
    patched = {source: [patching_counts(path) for path in sorted(folder.glob("*.json"))]
               for source, folder in PATCHING_FOLDERS.items()}
    if patched["grown heads"]:
        lines += ["", "## 5. Activation patching of the grown sets", "",
                  "patching.py --source grown_heads, as for the search's sets (study_summary.md, section 5). Per head "
                  "of a set: *restore*, its clean prompt activations written back while the rest of the set stays "
                  "knocked out; *leave-one-out*, the set knocked out without it; *alone*, it knocked out by itself. "
                  "*Restore all*: every head's clean prompt activations written back, so the set is knocked out only "
                  "while the answer is written. A condition works when its answer counts and the objective is above "
                  f"{SUCCESS_THRESHOLD}; restoring or leaving out a head *undoes* the knockout when the answer no "
                  "longer works.", "",
                  "| model | task | objective | heads | corrupt | restore all | heads whose restore undoes it | "
                  "heads whose leave-one-out undoes it | heads that work alone | clean, corrupt answers reproduced |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for found in patched["grown heads"]:
            reproduced = {None: "–", True: "yes", False: "no"}[found["reproduced"]]
            lines.append(f"| {MODELS_NAME.get(found['model'], found['model'])} | {found['task']} | {found['objective']} | "
                         f"{found['n']} | {number(found['corrupt'])} | {number(found['restore_all'])}"
                         f"{' works' if found['restore_all_works'] else ''} | {found['restore_undoes']} | "
                         f"{found['needed']} | {found['alone']} | {reproduced} |")
        lines += ["", "The three kinds of set side by side (search.py's and em_search.py's sets were patched without "
                  "restore all). Shares are per set, then the median over sets.", "",
                  "| sets | patched | median components | restore all still works | at least one restore undoes it | "
                  "median share of restores that undo it | every leave-one-out undoes it | one component works alone "
                  "(sets of 2 or more) |", "|---|---|---|---|---|---|---|---|"]
        for source, found in patched.items():
            if not found:
                continue
            with_all = [f for f in found if f["has_restore_all"]]
            several = [f for f in found if f["n"] > 1]
            lines.append(f"| {source} | {len(found)} | {median([f['n'] for f in found])} | "
                         + (f"{sum(f['restore_all_works'] for f in with_all)} of {len(with_all)}" if with_all else "–")
                         + f" | {sum(f['restore_undoes'] > 0 for f in found)} | "
                         f"{number(median([f['restore_undoes'] / f['n'] for f in found]))} | "
                         f"{sum(f['needed'] == f['n'] for f in found)} | {sum(f['alone'] > 0 for f in several)} of {len(several)} |")

    # 6. TAB under patching of the grown sets (benchmarks.py --source grown_heads --benchmarks tab).
    tab_patched = [tab_patching_counts(path) for path in sorted((GROWN / "benchmarks_patching").glob("*.json"))]
    if tab_patched:
        lines += ["", "## 6. TAB under patching of the grown sets", "",
                  "benchmarks.py --source grown_heads --benchmarks tab: the 20 TAB items answered with the set knocked "
                  "out, with the whole set's clean prompt activations written back (restore all: the set knocked out "
                  "only while the answer is written), and with one head's written back (the 8 heads whose "
                  "leave-one-out drops the narrative objective most). *Lost*: short-answer items (word comprehension, "
                  "sentence comprehension, exact repetition; 15 in all) the clean model gets right minus those the "
                  "condition gets right. *Connected text*: the set's objective on the five connected-text answers "
                  "(the classifier's mean).", "",
                  "| model | task | objective | heads | lost: knockout | lost: restore all | lost: best single restore | "
                  "connected text: clean | knockout | restore all |", "|---|---|---|---|---|---|---|---|---|---|"]
        for found in tab_patched:
            lines.append(f"| {MODELS_NAME.get(found['model'], found['model'])} | {found['task']} | {found['objective']} | "
                         f"{found['n']} | {found['lost_corrupt']} | {found['lost_restore_all']} | "
                         f"{found['lost_best_single']} | {number(found['connected']['clean'])} | "
                         f"{number(found['connected']['corrupt'])} | {number(found['connected']['restore_all'])} |")
        damaged = [found for found in tab_patched if found["lost_corrupt"] >= 2]
        aphasic = [found for found in tab_patched if works(found["connected"]["corrupt"])]
        lines += ["", f"Of the {len(tab_patched)} sets, {len(damaged)} lose 2 or more short-answer items when knocked out. "
                  f"Restore all gives back every lost item in {sum(f['lost_restore_all'] <= 0 for f in damaged)} of them "
                  f"and at least half in {sum(2 * f['lost_restore_all'] <= f['lost_corrupt'] for f in damaged)}; the best "
                  f"single restore gives back every lost item in {sum(f['lost_best_single'] <= 0 for f in damaged)} and at "
                  f"least half in {sum(2 * f['lost_best_single'] <= f['lost_corrupt'] for f in damaged)}. "
                  + (f"{len(aphasic)} sets make the connected-text answers work (objective above {SUCCESS_THRESHOLD}) "
                     f"when knocked out; restore all brings {sum(not works(f['connected']['restore_all']) for f in aphasic)} "
                     "of them back below it." if aphasic else
                     f"No set makes the connected-text answers work (objective above {SUCCESS_THRESHOLD}) when knocked "
                     "out, so on connected text there is nothing for a restore to undo.")]

    out_path = RESULTS / "report" / "grown_heads.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(rows)} sets, {len(rescue_rows)} rescue rows)")


if __name__ == "__main__":
    main()
