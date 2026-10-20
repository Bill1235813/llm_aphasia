"""One Markdown page with every result of the five-way study, read from results/fiveway/.

report.py covers the knockout search in depth (tables per tag, figures). This
page puts the whole study side by side, one short table per question, so the
paper's numbers can be read off in one place:

  1. Search: per model, task and objective, the stage that worked, the pruned
     knockout set's size and scores, how that set scores on the other two
     objectives (a greedy search is path-dependent: one objective's set can
     satisfy another objective whose own search found nothing), and its
     held-out score (held_out()).
  2. The earlier study's knockouts under the five-way protocol (known_circuits.py).
  3. Cos-sim ablation: the classifier-free search against the classifier's verdict.
  4. Steering: probes, severity, cross-task transfer, steering strengths, random
     controls and the rescue of the A knockout set.
  5. Activation patching of every knockout set.
  6. TAB and MMLU under each knockout set and its restorations.
  7. The emergent-modularity baseline: subtype assignments, clusters, the
     classifier's view of the lesions, and overlap with the search's knockout sets.
  8. em_search.py: MLP weight lesions, then heads; the same tables for its sets.

NOTE: the searches return *knockout sets*: the pruned set
of components whose joint knockout pushes the classifier objective above 0.5
on the task answer. Nothing here shows that such a set computes the
behaviour, so the page does not call them circuits; the result files keep
the earlier field name (pruned.circuit).

Sections whose files do not exist yet are skipped. It reads results only and
loads no model.

Usage:
    python src/fiveway/study_summary.py      # writes results/fiveway/report/study_summary.md
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.models import MODELS  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402

CLASSES = ["Control", "Cond/Anomic", "Broca", "Wernicke", "Scrambles"]
ORDER = list(MODELS)            # family by family, small to large
TASKS = ["cinderella", "sandwich"]
OBJECTIVES = ["A", "B", "C"]


def objectives_of(probabilities):
    """A, B and C from a five-class probability row."""
    return {"A": sum(probabilities[1:4]), "B": probabilities[2], "C": probabilities[3]}


def compact(heads, blocks):
    """'L0H25 L1H8 L1B30', shortened when long."""
    names = [f"L{layer}H{head}" for layer, head in heads] + [f"L{layer}B{block}" for layer, block in blocks]
    return " ".join(names) if len(names) <= 8 else " ".join(names[:8]) + f" ... ({len(names)} in all)"


def number(value, digits=2):
    """A score, or a dash for an answer too short to be scored."""
    return "-" if value is None else f"{value:.{digits}f}"


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def held_out(folder: Path, model: str, task: str, objective: str) -> str:
    """The knockout set on prompts it was not found on: the classifier's mean P(objective) over the five TAB
    connected-text answers (benchmarks.py), unablated -> knocked out (bold above 0.5), or '' without a file.

    NOTE: the search picks the set by its score on one answer;
    a set that made the model aphasic in general would score above 0.5 here too.
    """
    document = load(folder / "benchmarks" / f"{model}_{task}_{objective}.json")
    if document is None:
        return ""
    tab = document["summary"]["tab"]
    clean, knocked_out = [(tab[condition]["connected_text"].get("mean_objectives") or {}).get(objective)
                          for condition in ("clean", "corrupt")]
    shown = number(knocked_out)
    return f"{number(clean)} -> {f'**{shown}**' if knocked_out is not None and knocked_out > 0.5 else shown}"


def judged(source: str, model: str, task: str, objective: str) -> str:
    """The LLM judge's labels of a knockout set's answer and its compression ratio (llm_judge.py --circuits), or ''."""
    for row in (load(RESULTS / "report" / "circuit_judge.json") or {"circuits": []})["circuits"]:
        if (row["source"], row["model"], row["task"], row["objective"]) == (source, model, task, objective):
            ratio = "short" if row["compression_ratio"] is None else f"{row['compression_ratio']:.2f}"
            return f"{row['judge_binary']} / {row['judge_fiveway']} ({ratio})"
    return ""


def search_section(lines):
    lines += ["## 1. Knockout search (five-way classifier, rep1.25)", "",
              "Stage that first worked (1 single-layer heads, 2 forward sweep of heads, 3 forward sweep of heads and "
              "MLP blocks), size of the pruned knockout set, its objective value, and the pruned set's own A / B / C "
              "values. *none (best x)*: no stage reached 0.5; x is the highest value any stage reached.", "",
              "| model | task | baseline argmax (P scrambles) | A | B | C | gate |", "|---|---|---|---|---|---|---|"]
    cross = []
    for model in ORDER:
        for task in TASKS:
            results = load(RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json")
            if results is None:
                continue
            base = results["baseline"]["probabilities"]
            base_text = (f"{CLASSES[int(np.argmax(base))]} ({base[4]:.2f})" if base else "invalid")
            cells = []
            for name in OBJECTIVES:
                entry = results["objectives"].get(name)
                if entry is None or not entry.get("done"):
                    cells.append("running")
                    continue
                if entry.get("found_at_stage"):
                    circuit = entry["pruned"]["circuit"]
                    size = len(circuit["heads"]) + len(circuit["mlp_blocks"])
                    values = objectives_of(circuit["probabilities"])
                    cells.append(f"stage {entry['found_at_stage'][0]}, {size} comp., {circuit['value']:.2f}")
                    cross.append((model, task, name, compact(circuit["heads"], circuit["mlp_blocks"]), values,
                                  circuit["probabilities"][4], circuit["story"]))
                else:
                    best = max((stage.get("found") or stage.get("best") or {}).get("value") or 0.0
                               for stage in entry["stages"].values())
                    cells.append(f"none (best {best:.2f})")
            lines.append(f"| {model} | {task} | {base_text} | " + " | ".join(cells)
                         + f" | {'on' if results.get('gate', True) else 'off'} |")
    lines += ["", "Every knockout set on all three objectives (bold: above 0.5); held out: the classifier's mean "
              "P(objective searched for) on the five TAB connected-text prompts, unablated -> knocked out, prompts "
              "the set was not found on; the LLM judge's binary / five-way label of its answer with the answer's "
              "compression ratio (llm_judge.py --circuits; prose about 0.5-0.6, a self-repeating string far lower; "
              "'short' under 100 characters); and the start of the answer:", "",
              "| model | task | searched for | knockout set | A | B | C | P scrambles | held out | judge | answer |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, task, name, circuit, values, scrambles, story in cross:
        marks = " | ".join(f"**{values[o]:.2f}**" if values[o] > 0.5 else f"{values[o]:.2f}" for o in OBJECTIVES)
        text = story[:110].replace("\n", " / ").replace("|", "/")
        lines.append(f"| {model} | {task} | {name} | {circuit} | {marks} | {scrambles:.2f} | "
                     f"{held_out(RESULTS, model, task, name)} | {judged('search', model, task, name)} | {text} |")
    lines.append("")


def known_section(lines):
    files = sorted((RESULTS / "known_circuits").glob("*.json"))
    if not files:
        return
    lines += ["## 2. The earlier study's knockouts under the five-way protocol", "",
              "*3-head set*: the earlier study's Llama-2-7B knockout set L0H25, L1H8, L1H29 (its \"circuit\").", "",
              "| model | task | decoding | knockout | A | B | C | P scrambles | answer |",
              "|---|---|---|---|---|---|---|---|---|"]
    for path in files:
        document = json.loads(path.read_text())
        config = document["config"]
        for record in document["records"]:
            probabilities = record["probabilities"]
            values = record["objectives"]
            text = record["story"][:100].replace("\n", " / ").replace("|", "/")
            # NOTE: the runs named that knockout "circuit"; the page names it as what it is.
            name = record["name"].replace("circuit", "3-head set")
            lines.append(f"| {config['model']} | {config['task']} | {config['decoding']} | {name} | "
                         f"{values['A']:.3f} | {values['B']:.3f} | {values['C']:.3f} | "
                         f"{probabilities['Model_scrambles']:.3f} | {text} |")
    lines.append("")


def cosine_section(lines):
    rows = []
    for model in ORDER:
        for task in TASKS:
            cosine = load(RESULTS / "search" / f"{model}_{task}_rep1.25_cosine.json")
            classifier = load(RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json")
            if cosine is None:
                continue
            for name in OBJECTIVES:
                entry = cosine["objectives"].get(name)
                if entry is None or not entry.get("done"):
                    continue
                baseline_margin = entry["baseline_value"]
                if not entry.get("found_at_stage"):
                    rows.append(f"| {model} | {task} | {name} | {baseline_margin:.3f} | none | | | | |")
                    continue
                circuit = entry["pruned"]["circuit"]
                probabilities = circuit["probabilities"]
                members = {tuple(c) for c in circuit["heads"]} | {("B",) + tuple(c) for c in circuit["mlp_blocks"]}
                overlap = ""
                if classifier and classifier["objectives"].get(name, {}).get("found_at_stage"):
                    other = classifier["objectives"][name]["pruned"]["circuit"]
                    others = {tuple(c) for c in other["heads"]} | {("B",) + tuple(c) for c in other["mlp_blocks"]}
                    overlap = f"{len(members & others) / len(members | others):.2f}"
                rows.append(f"| {model} | {task} | {name} | {baseline_margin:.3f}"
                            f"{' (already works)' if baseline_margin > 0 else ''} | "
                            f"{compact(circuit['heads'], circuit['mlp_blocks'])} | {CLASSES[int(np.argmax(probabilities))]} | "
                            f"{objectives_of(probabilities)[name]:.2f} | {probabilities[4]:.2f} | {overlap} |")
    if not rows:
        return
    lines += ["## 3. Cos-sim objective instead of the classifier", "",
              "The same search with a classifier-free objective: mean MiniLM cosine to the target group's human "
              "transcripts minus mean cosine to Control's, working above 0. The classifier's view of the knockout set "
              "the cos-sim search found, and its overlap (Jaccard) with the classifier search's set.", "",
              "| model | task | objective | cos-sim margin, unablated | cos-sim knockout set | classifier argmax | "
              "classifier P(objective) | P scrambles | Jaccard |", "|---|---|---|---|---|---|---|---|---|"] + rows + [""]


def steering_section(lines):
    rows, generation = [], []
    for model in ORDER:
        analysis = load(RESULTS / "steering" / model / "analysis.json")
        # NOTE: an analysis still being written lacks its last section.
        if analysis is not None and "cross_task" in analysis:
            for task in TASKS:
                section = analysis[task]
                auc = {name: section["probe_auc"][name] for name in OBJECTIVES}
                transfer = analysis["cross_task"]["transfer_auc"][
                    "cinderella_to_sandwich" if task == "cinderella" else "sandwich_to_cinderella"]["A"]
                severity = section["severity"]["spearman_aq"]
                rows.append(f"| {model} | {task} | " + " | ".join(f"{min(auc[n]):.2f}-{max(auc[n]):.2f}" for n in OBJECTIVES)
                            + f" | {min(transfer):.2f}-{max(transfer):.2f} | {min(severity):.2f} | "
                            f"{np.mean(section['pca']['pc1_abs_cosine_A']):.2f} | "
                            f"{min(section['A_norm_over_resid_norm']):.2f}-{max(section['A_norm_over_resid_norm']):.2f} |")
        for task in TASKS:
            summary = load(RESULTS / "steering" / model / f"summary_{task}.json")
            if summary is None:
                continue
            smallest = summary["smallest_strength_above_threshold"]

            def strengths(name):
                values = [v for v in smallest.get(name, {}).values() if v is not None]
                return f"{len(values)}/{len(smallest.get(name, {}))}, median {np.median(values):.2f}" if values else "0"

            rescue = summary.get("rescue") or {}
            rescue_text = ""
            if rescue.get("value_A"):
                values = rescue["value_A"]
                randoms = [v for k, v in values.items() if k.startswith("knockout, random")]
                rescue_text = (f"{values['knockout']:.2f} -> {values['knockout, A removed']:.3f} "
                               f"(random removed: {np.mean(randoms):.2f})")
            generation.append(f"| {model} | {task} | {summary['unsteered']['A']:.3f} | {strengths('A')} | "
                              f"{strengths('B')} | {strengths('C')} | {summary['random']['share_above_threshold']:.2f} | "
                              f"{rescue_text} |")
    if rows:
        lines += ["## 4. Steering", "", "### Representation (human transcripts, speaker-grouped folds, each text once)", "",
                  "| model | task | probe AUC A | probe AUC B | probe AUC C | transfer AUC A | best severity rho | "
                  "mean abs cos(PC1, A) | abs A over residual norm |", "|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    if generation:
        lines += ["### Steering the model's own answer", "",
                  "Layers (of 8 spread over the depth) at which adding the vector turns the answer into its objective "
                  "(P above 0.5) at some strength r of 4 or less, and the median smallest such r (r = size of the added "
                  "vector relative to the mean residual norm); the share of random-direction settings that do; and the "
                  "rescue of the search's A knockout set by removing the A direction after every block.", "",
                  "| model | task | unsteered P(A) | A works | B works | C works | random share | rescue: knockout -> A removed |",
                  "|---|---|---|---|---|---|---|---|"] + generation + [""]


def patching_section(lines, folder: Path = RESULTS, title: str = "## 5. Activation patching of the knockout sets"):
    files = sorted((folder / "patching").glob("*.json"))
    if not files:
        return
    lines += [title, "",
              "Objective value: clean, corrupt (the whole set knocked out); per component, restoring its clean "
              "prompt activations (restore), and knocking out the set without it (leave-one-out).", "",
              "| model | task | objective | clean | corrupt | restore (per component) | leave-one-out (per component) |",
              "|---|---|---|---|---|---|---|"]
    for path in files:
        document = json.loads(path.read_text())
        config = document["config"]
        by_name = {condition["name"]: condition for condition in document["conditions"]}
        components = document["components"][:10]
        restore = " ".join(f"{c['name']} {number(c['restore_value'])}" for c in components)
        leave = " ".join(f"{c['name']} {number(c['leave_one_out_value'])}" for c in components)
        more = f" (+{len(document['components']) - 10})" if len(document["components"]) > 10 else ""
        lines.append(f"| {config['model']} | {config['task']} | {config['objective']} | {number(by_name['clean']['value'], 3)} | "
                     f"{number(by_name['corrupt']['value'], 3)} | {restore}{more} | {leave}{more} |")
    lines.append("")


def benchmarks_section(lines, folder: Path = RESULTS, title: str = "## 6. TAB and MMLU under the knockout sets"):
    files = sorted((folder / "benchmarks").glob("*.json"))
    if not files:
        return
    lines += [title, "",
              "TAB judge (llm_judge.py, the TAB paper's 19 features of connected text): mean features present per "
              "answer / share of answers judged impaired overall.", ""]
    for path in files:
        document = json.loads(path.read_text())
        config = document["config"]
        summary = document["summary"]
        judged = (load(folder / "tab_judge" / path.name) or {}).get("summary", {})
        lines += [f"**{config['model']}, knockout set of {config['circuit_task']} objective {config['objective']}**", "",
                  "| condition | word compr. | sentence compr. | repetition | connected text P(A) | TAB judge | "
                  "MMLU global facts acc / log P | MMLU world history acc / log P |", "|---|---|---|---|---|---|---|---|"]
        for condition, tab in summary.get("tab", {}).items():
            def mmlu_cell(subject):
                entry = summary.get("mmlu", {}).get(subject, {}).get(condition)
                return f"{entry['accuracy']:.2f} / {entry['mean_log_probability_correct']:.2f}" if entry else ""

            def tab_cell(subtest):
                entry = tab.get(subtest) or {}
                if subtest == "connected_text":
                    # NOTE: no connected-text answer long enough to score: no mean.
                    return number((entry.get("mean_objectives") or {}).get("A"))
                return f"{entry['correct']}/{entry['n']}" if "correct" in entry else ""

            verdict = judged.get(condition)
            judge_cell = f"{verdict['mean_features_present']:.1f} / {verdict['share_impaired']:.1f}" if verdict else ""
            lines.append(f"| {condition} | {tab_cell('word_comprehension')} | {tab_cell('sentence_comprehension')} | "
                         f"{tab_cell('repetition')} | {tab_cell('connected_text')} | {judge_cell} | "
                         f"{mmlu_cell('global_facts')} | {mmlu_cell('high_school_world_history')} |")
        lines.append("")


def em_section(lines):
    short = dict(zip(["Control", "Conduction/Anomic", "Broca", "Wernicke", "Model_scrambles"], CLASSES))
    rows = []
    for model in ORDER:
        lesions = load(RESULTS / "em_baseline" / model / "lesions.json")
        clusters = load(RESULTS / "em_baseline" / model / "clusters.json")
        if lesions is None or clusters is None:
            continue
        for task, document in lesions["tasks"].items():
            task_clusters = clusters["tasks"].get(task, {})
            sizes = task_clusters.get("sizes", {})
            succeeded = ", ".join(f"{group} {size}" for group, size in sizes.items() if size is not None) or "none"
            comparison = task_clusters.get("comparison") or {}
            classifier = comparison.get("classifier") or {}
            single = classifier.get("single_lesions") or {}
            class_counts = sorted(single.get("predicted_class_counts", {}).items(), key=lambda item: -item[1])
            single_classes = ", ".join(f"{short[name]} {count}" for name, count in class_counts)
            single_works = ", ".join(f"{name} {len(works)}" for name, works in single.get("works", {}).items())
            cluster_cells = []
            for group, verdict in (classifier.get("clusters") or {}).items():
                if verdict.get("predicted_class") is None:
                    cluster_cells.append(f"{group}: unscored")
                else:
                    cluster_cells.append(f"{group}: {short[verdict['predicted_class']]} "
                                         f"(A {number(verdict['objectives']['A'])})")
            # NOTE: the top 1% of each submodule's weights is
            # lesioned, so a circuit's components hold 1% of them by chance; more
            # would mean the search found gradient-important weights.
            shares = []
            for objective, circuit in (comparison.get("circuits") or {}).items():
                n_top = sum(component["n_top1pct"] for component in circuit["components"])
                n_all = sum(component["n_weights"] for component in circuit["components"])
                shares.append(f"{objective} {n_top / n_all:.3f}")
            rows.append(f"| {model} | {task} | {len(document['lesions'])} ({document['n_answers_changed']}) | "
                        f"{document['unlesioned']['assigned']} | {succeeded} | {single_classes} | {single_works} | "
                        f"{'; '.join(cluster_cells)} | {', '.join(shares) or '-'} |")
    if not rows:
        return
    lines += ["## 7. Emergent-modularity baseline (Wang et al., 2025)", "",
              "Top 1% of |w dL/dw| per submodule on Control narratives; one submodule lesioned at a time and assigned "
              "the human group its answer is closest to by BLEU-1 plus BERTScore (standardized); per subtype, a "
              "cluster grown until its joint lesion is assigned that subtype. *EM success*: the subtypes whose cluster "
              "got there, with its size. Then the five-way classifier's view: the predicted classes of the single "
              "lesions, how many of them work on A / B / C, and each cluster's predicted class and P(aphasia). Last, "
              "the share of the top-1% weights inside the search's knockout sets (chance: 0.010).", "",
              "| model | task | lesions (answer changed) | unlesioned assigned | EM success | "
              "single lesions, classifier | single lesions that work | clusters, classifier | knockout sets' top-1% share |",
              "|---|---|---|---|---|---|---|---|---|"] + rows + [""]


def em_search_section(lines):
    rows = []
    for model in ORDER:
        for task in TASKS:
            results = load(RESULTS / "em_search" / f"{model}_{task}.json")
            if results is None:
                continue
            search = load(RESULTS / "search" / f"{model}_{task}_rep1.25_classifier.json") or {"objectives": {}}
            for objective, entry in results["objectives"].items():
                stages = entry["stages"]
                first = stages["1_mlp_lesions"]
                stage1 = f"{'works' if first['works'] else 'no'}: {first['best_size']} matrices, {number(first['best']['value'])}"
                second = stages["2_single_layer_heads"]
                stage2 = f"{'works' if second['works'] else 'no'} ({number((second['best'] or {}).get('value'))})"
                third = stages.get("3_heads_on_mlp_lesion")
                stage3 = "-" if third is None else (f"{'works' if third['works'] else 'no'}: {len(third['base_lesions'])} "
                                                    f"matrices + {len(third['found']['heads'])} heads, "
                                                    f"{number(third['found']['value'])}")
                circuit_cell, answer = "none", ""
                if entry.get("pruned"):
                    circuit = entry["pruned"]["circuit"]
                    probabilities = circuit["probabilities"]
                    circuit_cell = (f"{len(circuit.get('lesions', []))} matrices + {len(circuit['heads'])} heads, "
                                    f"**{number(circuit['value'])}**, {CLASSES[int(np.argmax(probabilities))]}, "
                                    f"P scrambles {number(probabilities[4])}")
                    answer = circuit["story"][:110].replace("\n", " / ").replace("|", "/")
                earlier = search["objectives"].get(objective, {})
                found = earlier.get("found_at_stage")
                earlier_cell = (f"stage {found[0]}, {earlier['pruned']['circuit']['n_components']} comp."
                                if found else "none")
                rows.append(f"| {model} | {task} | {objective} | {stage1} | {stage2} | {stage3} | {circuit_cell} | "
                            f"{held_out(RESULTS / 'em_search', model, task, objective)} | "
                            f"{judged('em_search', model, task, objective)} | {earlier_cell} | {answer} |")
    if not rows:
        return
    lines += ["## 8. MLP weight lesions, then heads (em_search.py)", "",
              "Stage 1: the emergent-modularity lesions (top 1% of |w dL/dw| on Control narratives) of MLP matrices "
              "only, added in the order of their single-lesion value until the objective passes 0.5, no cap (best "
              "prefix: size, value). Stage 2: search.py's single-layer heads, reused. Stage 3: search.py's forward "
              "sweep of heads with the best stage-1 lesion in every knockout (base size + heads added, value). The "
              "pruned knockout set (bold: its objective value); held out: the classifier's mean P(objective) on "
              "the five TAB connected-text prompts, unablated -> knocked out; the LLM judge's binary / five-way "
              "label of its answer with the answer's compression ratio; and search.py's own result for "
              "comparison.", "",
              "| model | task | objective | stage 1 | stage 2 | stage 3 | knockout set | held out | judge | "
              "search.py | answer |",
              "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + [""]
    patching_section(lines, RESULTS / "em_search", "### Activation patching of the em_search knockout sets")
    benchmarks_section(lines, RESULTS / "em_search", "### TAB and MMLU under the em_search knockout sets")


def main():
    lines = ["# Five-way study: all results", "",
             "Generated by `src/fiveway/study_summary.py` from `results/fiveway/`. Protocol and decisions: "
             "`src/fiveway/README.md`.", ""]
    for section in (search_section, known_section, cosine_section, steering_section, patching_section,
                    benchmarks_section, em_section, em_search_section):
        section(lines)
    out = RESULTS / "report" / "study_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
