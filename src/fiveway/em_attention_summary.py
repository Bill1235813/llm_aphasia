"""The em greedy search in six spaces on one page: does an attention-matrix lesion act like an MLP lesion or like a head?

Reads results/fiveway/em_attention/<model>_<task>.json and its benchmarks/
files (em_attention.py), and results/fiveway/em_attention/tab_judge/ once
llm_judge.py --tab has run on them, and writes results/fiveway/report/em_attention.md:

    1 MLP lesions only            em_search stage 1        4 attention lesions only    the same growth
    2 heads only                  em_search stage 2        2b heads, grown             the same growth
    3 MLP lesions + heads         em_search stage 1 / 3    5 MLP + attention lesions   stage 1 / sweep

For every space: how often a set works (objective above 0.5 on the task
answer), by model family and per model; paired comparisons on the same
model, task and objective; and, for the working sets, what they cost
elsewhere (MMLU, the TAB subtests), whether they carry over (TAB connected
text, the other task), what the answers look like, and how long the search took.

Usage:
    python src/fiveway/em_attention_summary.py
"""

from __future__ import annotations

import json
import statistics
import sys
import zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS, TASKS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.scorer import OBJECTIVES, SUCCESS_THRESHOLD  # noqa: E402

FOLDER = RESULTS / "em_attention"
NAMES = {"gpt2-xl": "GPT-2 XL", "llama2-7b-chat": "Llama-2-7B-chat", "llama2-13b-chat": "Llama-2-13B-chat",
         "pythia-1b": "Pythia-1B", "pythia-1.4b": "Pythia-1.4B", "pythia-2.8b": "Pythia-2.8B", "pythia-6.9b": "Pythia-6.9B"}
SPACES = {"1_mlp_lesions": "1 MLP lesions", "4_attention_lesions": "4 attention lesions",
          "2_heads": "2 heads (single layer)", "2b_heads_grown": "2b heads (grown)",
          "3_mlp_lesions_and_heads": "3 MLP lesions + heads", "5_mlp_and_attention_lesions": "5 MLP + attention lesions"}
PAIRS = [("4_attention_lesions", "1_mlp_lesions"), ("4_attention_lesions", "2b_heads_grown"),
         ("4_attention_lesions", "2_heads"), ("5_mlp_and_attention_lesions", "3_mlp_lesions_and_heads")]
FAMILIES = {"Pythia": [m for m in MODELS if m.startswith("pythia")], "GPT-2": ["gpt2-xl"],
            "Llama": ["llama2-7b-chat", "llama2-13b-chat"]}


def number(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def load_rows() -> list:
    """One row per model, task, objective and space: the reported set and its checks."""
    rows = []
    for model in NAMES:
        for task in TASKS:
            path = FOLDER / f"{model}_{task}.json"
            bench_path = FOLDER / "benchmarks" / f"{model}_{task}.json"
            if not path.exists():
                continue
            results = json.loads(path.read_text())
            bench = json.loads(bench_path.read_text()) if bench_path.exists() else None
            if "reported" not in results:
                continue
            for objective, spaces in results["reported"].items():
                for space, row in spaces.items():
                    name = f"{space}_{objective}"
                    checks = {}
                    if bench and name in bench["conditions"] and bench["summary"].get("tab"):
                        tab, clean = bench["summary"]["tab"][name], bench["summary"]["tab"]["clean"]
                        mmlu = bench["summary"].get("mmlu", {})
                        connected = tab["connected_text"]["mean_objectives"]
                        other = bench["other_task"].get(bench["conditions"][name])
                        checks = {
                            # NOTE: only once both subjects are complete (100 and
                            # 237 questions): a run still in its MMLU step has part-way means.
                            "mmlu_drop": (statistics.mean(mmlu[s]["clean"]["mean_log_probability_correct"]
                                                          - mmlu[s][name]["mean_log_probability_correct"] for s in mmlu)
                                          if all(mmlu.get(s, {}).get(name, {}).get("n") == size for s, size in
                                                 (("global_facts", 100), ("high_school_world_history", 237)))
                                          else None),
                            "lost": {sub: clean[sub]["correct"] - tab[sub]["correct"]
                                     for sub in ("word_comprehension", "sentence_comprehension", "repetition")},
                            "tab_connected": None if connected is None else connected[objective],
                            "other": (sum(other["probabilities"][i] for i in OBJECTIVES[objective])
                                      if other and other["valid"] and other["probabilities"] else None)}
                    judge_path = FOLDER / "tab_judge" / f"{model}_{task}.json"
                    if checks and judge_path.exists():
                        judged = json.loads(judge_path.read_text())["summary"].get(name)
                        checks["judge_features"] = None if judged is None else judged["mean_features_present"]
                    seconds = None
                    own = results["objectives"].get(objective, {}).get(space)
                    if own is not None:
                        seconds = own.get("seconds")
                    rows.append({"model": model, "task": task, "objective": objective, "space": space, **row,
                                 "checks": checks, "seconds": seconds})
    return rows


def main():
    rows = load_rows()
    index = {(r["model"], r["task"], r["objective"], r["space"]): r for r in rows}
    combos = sorted({(r["model"], r["task"], r["objective"]) for r in rows})
    lines = ["# The em greedy search in six spaces: attention-matrix lesions against MLP lesions and heads", "",
             "Generated by `src/fiveway/em_attention_summary.py` from `results/fiveway/em_attention/`. "
             f"{len(combos)} of 42 model, task and objective combinations so far.", "",
             "A set works when the five-way classifier's objective on the task answer is above 0.5. Spaces 1-3 are "
             "em_search.py's; 4, 5 and 2b are em_attention.py's. Lesions are always the top 1% of a matrix's "
             "weights by |w dL/dw| on Control narratives; heads are removed whole. Spaces 4 and 1 use the same "
             "procedure (lesions ranked by their single value, grown until the objective passes 0.5), as do 2b "
             "and 4; space 5 sweeps attention lesions on top of the best MLP lesions as space 3 sweeps heads.", "",
             "## 1. How often each space works", "",
             "| space | " + " | ".join(f"{family}" for family in FAMILIES) + " | all |",
             "|---|" + "---|" * (len(FAMILIES) + 1)]
    for space, label in SPACES.items():
        cells = []
        for family, models in FAMILIES.items():
            done = [r for r in rows if r["space"] == space and r["model"] in models]
            cells.append(f"{sum(r['works'] for r in done)} / {len(done)}")
        done = [r for r in rows if r["space"] == space]
        cells.append(f"{sum(r['works'] for r in done)} / {len(done)}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "**Paired**, on the same model, task and objective: both work / only the first / only the second / "
              "neither.", "", "| comparison | both | only first | only second | neither |", "|---|---|---|---|---|"]
    for first, second in PAIRS:
        counts = Counter()
        for combo in combos:
            a, b = index.get((*combo, first)), index.get((*combo, second))
            if a is None or b is None:
                continue
            counts[("both" if a["works"] and b["works"] else "first" if a["works"] else
                    "second" if b["works"] else "neither")] += 1
        lines.append(f"| {SPACES[first]} vs {SPACES[second]} | {counts['both']} | {counts['first']} | "
                     f"{counts['second']} | {counts['neither']} |")
        if first == "5_mlp_and_attention_lesions":
            # NOTE: when MLP lesions alone work, spaces 3 and 5 both
            # report stage 1's set and agree by construction; this row leaves those out.
            counts = Counter()
            for combo in combos:
                a, b = index.get((*combo, first)), index.get((*combo, second))
                if a is None or b is None or a.get("from_em_stage_1"):
                    continue
                counts[("both" if a["works"] and b["works"] else "first" if a["works"] else
                        "second" if b["works"] else "neither")] += 1
            lines.append(f"| {SPACES[first]} vs {SPACES[second]}, without stage 1's reused set | {counts['both']} | "
                         f"{counts['first']} | {counts['second']} | {counts['neither']} |")

    lines += ["", "**By model** (working combinations out of 6, two tasks by three objectives)", "",
              "| model | " + " | ".join(SPACES.values()) + " |", "|---|" + "---|" * len(SPACES)]
    for model in NAMES:
        cells = []
        for space in SPACES:
            done = [r for r in rows if r["model"] == model and r["space"] == space]
            cells.append(f"{sum(r['works'] for r in done)}" + ("" if len(done) == 6 else f" of {len(done)}"))
        lines.append(f"| {NAMES[model]} | " + " | ".join(cells) + " |")

    lines += ["", "## 2. What a working set costs and carries, by space", "",
              "Working sets only. MMLU: drop of the mean log-probability of the right answer (nats, averaged over "
              "global facts and world history) from the unablated model. TAB: items lost of 5 per subtest. Held "
              "out: the objective on the five TAB connected-text answers (mean) and on the other task's answer.", "",
              "| space | n | median size | not minimal | MMLU drop, median | MMLU drop > 0.5 | "
              "word / sentence / repetition lost | TAB connected text > 0.5 | other task > 0.5 | "
              "judge: aphasia features, mean (of 19) | classes | self-repeating |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for space, label in SPACES.items():
        working = [r for r in rows if r["space"] == space and r["works"]]
        if not working:
            lines.append(f"| {label} | 0 | | | | | | | | | | |")
            continue
        sizes = [r["n_mlp_lesions"] + r["n_attention_lesions"] + r["n_heads"] for r in working]
        checked = [r for r in working if r["checks"]]
        drops = [r["checks"]["mmlu_drop"] for r in checked if r["checks"]["mmlu_drop"] is not None]
        lost = {sub: [r["checks"]["lost"][sub] for r in checked]
                for sub in ("word_comprehension", "sentence_comprehension", "repetition")}
        connected = [r["checks"]["tab_connected"] for r in checked if r["checks"]["tab_connected"] is not None]
        other = [r["checks"]["other"] for r in checked]
        judged = [r["checks"]["judge_features"] for r in checked if r["checks"].get("judge_features") is not None]
        classes = Counter(r["predicted_class"] for r in working)
        ratios = [len(zlib.compress(r["story"].encode(), 9)) / len(r["story"].encode())
                  for r in working if len(r["story"]) >= 100]
        # NOTE: a grown set whose pruning stopped at em_attention.py's
        # PRUNE_BUDGET is larger than minimal, and so is what it costs elsewhere.
        not_minimal = sum(r.get("minimal") is False for r in working)
        lines.append(
            f"| {label} | {len(working)} | {statistics.median(sizes):g} | {not_minimal} | "
            f"{number(statistics.median(drops)) if drops else '–'} | {sum(d > 0.5 for d in drops)} / {len(drops)} | "
            + " / ".join(f"{statistics.mean(v):.1f}" if v else "–" for v in lost.values()) + " | "
            f"{sum(c > SUCCESS_THRESHOLD for c in connected)} / {len(connected)} | "
            f"{sum((o or 0) > SUCCESS_THRESHOLD for o in other)} / {len(other)} | "
            f"{number(statistics.mean(judged), 1) if judged else '–'} | "
            + ", ".join(f"{k} {v}" for k, v in classes.most_common(3)) + " | "
            f"{sum(x < 0.25 for x in ratios)} / {len(ratios)} |")

    lines += ["", "## 3. Where the working lesions and heads sit", "",
              "Relative depth of each component (layer / number of layers) in the working sets; attention lesions by "
              "matrix.", "", "| space | component | n | median depth | first 10% of layers | last half |",
              "|---|---|---|---|---|---|"]
    layers = {"gpt2-xl": 48, "llama2-7b-chat": 32, "llama2-13b-chat": 40, "pythia-1b": 16, "pythia-1.4b": 24,
              "pythia-2.8b": 32, "pythia-6.9b": 32}
    for space, label in SPACES.items():
        depth = {}
        for r in rows:
            if r["space"] != space or not r["works"]:
                continue
            for name in r["lesions"]:
                layer, kind = name[1:].split(".")
                group = f"{kind} lesion" if kind in ("q", "k", "v", "o") else "MLP lesion"
                depth.setdefault(group, []).append(int(layer) / layers[r["model"]])
            for layer, _ in r["heads"]:
                depth.setdefault("head", []).append(layer / layers[r["model"]])
        for group, values in sorted(depth.items()):
            lines.append(f"| {label} | {group} | {len(values)} | {statistics.median(values):.2f} | "
                         f"{sum(v < 0.1 for v in values)} | {sum(v >= 0.5 for v in values)} |")

    lines += ["", "## 4. Time per search (minutes)", "",
              "Generation and scoring time summed over the answers each search used (spaces 4, 2b and 5; space 5 "
              "takes no time when it reuses stage 1). Each answer is timed on the GPU that generated it, an H200 or "
              "an A100 (jobs moved between partitions), so times compare only roughly across rows.", "",
              "| model | task | " + " | ".join(
                  f"{o} {s}" for o in OBJECTIVES for s in ("4", "2b", "5")) + " |",
              "|---|---|" + "---|" * 9]
    for model in NAMES:
        for task in TASKS:
            cells, any_value = [], False
            for objective in OBJECTIVES:
                for space in ("4_attention_lesions", "2b_heads_grown", "5_mlp_and_attention_lesions"):
                    r = index.get((model, task, objective, space))
                    seconds = None if r is None else r["seconds"]
                    any_value |= seconds is not None
                    cells.append("…" if seconds is None else f"{seconds / 60:.1f}")
            if any_value:
                lines.append(f"| {NAMES[model]} | {task} | " + " | ".join(cells) + " |")
    out_path = RESULTS / "report" / "em_attention.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({len(lines)} lines; {len(rows)} rows)")


if __name__ == "__main__":
    main()
