"""Checks of report.py on hand-made search files and on the real ones.

1. Constants. report.py's copies of the classifier's class names, objective
   columns and thresholds equal scorer.py's, and its minimum answer length
   prompts.py's.
2. Hand-made searches. Results files in search.py's schema with every case the
   report has to handle (circuits found at stages 1, 2 and 3, MLP blocks, no
   circuit, an objective still running, an objective not asked for, a baseline
   that already works, an unscorable baseline, negative cos-sim margins stored
   as null next to the answer cache that holds them, in search.py's current
   cache format and the one from before its gate switch, a search run without
   the Cinderella gate, a cut-off cache line, a 40 x 40 grid, a stale figure)
   go through report.main(). Its JSON must hold the numbers worked out by hand
   below, its Markdown a row per objective, and every figure must exist.
3. Real searches. For every results file under results/fiveway/search whose
   answer cache is on this disk, what the report reads off the file must agree
   with the cached answers it comes from:
     - each sweep step's value is the cached value of the knockout of all the
       components committed up to that step, and its n_components is their
       number (the x axis of the sweep figures);
     - stage 1's best value (report.best_values) is the maximum over every
       stage-1 knockout: each single head, and each layer's top-k heads ranked
       as search.py ranks them;
     - report.counts counts exactly the cached answers that search.py's own
       Evaluator.valid counts under the file's gate, and the unablated
       answer's cached value is the file's baseline value;
     - the file's single-head values are the cached ones, and the values
       report.py reads from the cache for the cosine objective fill exactly
       the file's nulls with negative margins or unscored answers;
     - the pruned circuit's probabilities and answer are the cached ones, and
       its compact text names exactly its components;
     - both tags have the same unablated answer.
   Then report.main() runs on the real files, writing to runs/ rather than results/.

Usage (in the shell; nothing here runs a model):
    CUDA_VISIBLE_DEVICES= .venv/bin/python src/fiveway/test_report.py
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import prompts, report, scorer, search  # noqa: E402
from fiveway.engine import Intervention  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402

TEST_DIR = RUNS / "fiveway" / "report_test"
P_BASELINE = [0.9, 0.04, 0.03, 0.02, 0.01]   # A = 0.09, B = 0.03, C = 0.02
UNCOUNTED_HEADS = [(4, 4), (6, 6), (7, 7)]   # hand-made cos-sim heads whose answers do not count under the gate


def knockout(heads, mlp_blocks, value, probabilities=P_BASELINE, story="An answer of a knockout, long enough."):
    """A knockout as search.py's summary() stores it."""
    return {"heads": heads, "mlp_blocks": mlp_blocks, "n_components": len(heads) + len(mlp_blocks), "value": value,
            "probabilities": probabilities, "cosine": {"A": 0.0, "B": 0.0, "C": 0.0}, "answer": True, "story": story}


def stage1(single_head_values, per_layer_values, best, works):
    """Stage 1 as search.py stores it; per_layer_values holds each layer's best value (None: nothing counted)."""
    return {"works": works, "best": best, "n_working_single_heads": 0, "n_working_sets": 0,
            "single_head_values": single_head_values,
            "per_layer_best": [None if value is None else knockout([[layer, 0], [layer, 1]], [], value)
                               for layer, value in enumerate(per_layer_values)]}


def sweep(steps, works):
    """A forward sweep as search.py stores it; steps are (kind, [layer, index], value)."""
    heads, blocks, trajectory = [], [], []
    for kind, added, value in steps:
        (heads if kind == "head" else blocks).append(added)
        trajectory.append({"layer": added[0], "kind": kind, "added": added, "value": value, "gain": 0.0,
                           "n_components": len(heads) + len(blocks)})
    return {"works": works, "found": knockout(list(heads), list(blocks), steps[-1][2]), "trajectory": trajectory}


def results_file(model, task, kind, geometry, baseline, objectives, asked=("A", "B", "C"), gate=None):
    """A results file in search.py's schema; gate=None leaves "gate" out, as files from before search.py's switch."""
    results = {"config": {"model": model, "task": task, "decoding": "rep1.25", "objectives": list(asked),
                          "objective_kind": kind, "all_stages": False, "tag": ""},
               "geometry": geometry, "objectives": objectives, "baseline": baseline}
    if gate is not None:
        results["gate"] = gate
    return results


def write_hand_made_searches(search_dir: Path, cache_dir: Path) -> np.ndarray:
    """Four results files and one answer cache; returns the true cos-sim A margins of the single heads."""
    small = {"n_layers": 16, "n_heads": 8, "d_head": 256, "d_model": 2048, "d_mlp": 8192, "n_blocks": 32}
    baseline = {"answer": True, "yesno": "Yes.", "story": "The unablated story of Cinderella, long enough.",
                "full_text": "The unablated story of Cinderella, long enough.", "valid": True,
                "probabilities": P_BASELINE, "cosine": {"A": -0.01, "B": -0.05, "C": 0.01}}
    grid = [[round(0.05 + 0.001 * (8 * layer + head), 6) for head in range(8)] for layer in range(16)]
    grid[3][2] = None     # an answer that did not count
    grid[5][1] = 0.7      # the one single head that works

    # pythia-1b cinderella, classifier, searched without the gate: A at stage 1, B at stage 3 with MLP blocks,
    # C nowhere.
    a_circuit = knockout([[5, 1]], [], 0.7, [0.2, 0.5, 0.1, 0.1, 0.1])
    b_circuit = knockout([[1, 2]], [[0, 3], [1, 0]], 0.55, [0.1, 0.2, 0.55, 0.05, 0.1])
    classifier = results_file("pythia-1b", "cinderella", "classifier", small, baseline, {
        "A": {"baseline_value": 0.09, "found_at_stage": "1_single_layer_heads", "done": True,
              "stages": {"1_single_layer_heads": stage1(grid, [0.1] * 5 + [0.8, 0.1, None] + [0.1] * 8, a_circuit, True)},
              "pruned": {"circuit": a_circuit, "steps": []}},
        "B": {"baseline_value": 0.03, "found_at_stage": "3_forward_heads_mlp", "done": True,
              "stages": {"1_single_layer_heads": stage1(grid, [0.04, 0.04, 0.2] + [0.04] * 13,
                                                        knockout([[2, 0], [2, 1]], [], 0.2), False),
                         "2_forward_heads": sweep([("head", [0, 1], 0.05), ("head", [0, 2], 0.1), ("head", [1, 0], 0.1),
                                                   ("head", [1, 3], 0.3), ("head", [2, 5], 0.25)], False),
                         "3_forward_heads_mlp": sweep([("head", [0, 1], 0.1), ("block", [0, 3], 0.2),
                                                       ("head", [1, 2], 0.4), ("block", [1, 0], 0.6)], True)},
              "pruned": {"circuit": b_circuit, "steps": [{"removed": ["head", [0, 1]], "value": 0.55}]}},
        "C": {"baseline_value": 0.02, "found_at_stage": None, "done": True,
              "stages": {"1_single_layer_heads": stage1(grid, [0.05] * 16, knockout([[0, 0]], [], 0.05), False),
                         "2_forward_heads": sweep([("head", [0, 1], 0.2), ("head", [3, 3], 0.1)], False),
                         "3_forward_heads_mlp": sweep([("block", [0, 1], 0.45), ("head", [2, 2], 0.3)], False)}}},
        gate=False)

    # pythia-1b cinderella, cos-sim: A at stage 2, B nowhere, C at stage 1 from a baseline that already works.
    # The true margins of the single heads run from -0.03 to 0.055; search.py stores the negative ones as null.
    margins = np.array([[round(-0.03 + 0.01 * head + 0.001 * layer, 6) for head in range(8)] for layer in range(16)])
    stored = [[None if value < 0 else float(value) for value in row] for row in margins]
    # Three heads whose answers do not count under the gate (their cache records below say why).
    for layer, head in UNCOUNTED_HEADS:
        stored[layer][head] = None
    cosine_baseline = dict(baseline, probabilities=[0.900001, 0.04, 0.03, 0.02, 0.01])
    a_cosine = knockout([[0, 0], [5, 1]], [], 0.03, [0.1, 0.35, 0.2, 0.25, 0.1])
    c_cosine = knockout([[2, 2]], [], 0.04, [0.3, 0.1, 0.1, 0.4, 0.1])
    cosine = results_file("pythia-1b", "cinderella", "cosine", small, cosine_baseline, {
        "A": {"baseline_value": -0.01, "found_at_stage": "2_forward_heads", "done": True,
              "stages": {"1_single_layer_heads": stage1(stored, [-0.005] * 16, knockout([[0, 0]], [], -0.005), False),
                         "2_forward_heads": sweep([("head", [0, 0], -0.002), ("head", [5, 1], 0.03)], True)},
              "pruned": {"circuit": a_cosine, "steps": []}},
        "B": {"baseline_value": -0.05, "found_at_stage": None, "done": True,
              "stages": {"1_single_layer_heads": stage1(stored, [-0.02] * 16, knockout([[0, 0]], [], -0.02), False),
                         "2_forward_heads": sweep([("head", [0, 0], -0.01)], False),
                         "3_forward_heads_mlp": sweep([("block", [0, 0], 0.0)], False)}},
        "C": {"baseline_value": 0.01, "found_at_stage": "1_single_layer_heads", "done": True,
              "stages": {"1_single_layer_heads": stage1(stored, [0.04] * 16, c_cosine, True)},
              "pruned": {"circuit": c_cosine, "steps": []}}}, gate=True)
    # The answer cache in search.py's current format, which has no "valid" flag: whether an answer counts
    # depends on the search's gate (search.Evaluator.valid).
    lines = []
    for layer in range(16):
        for head in range(8):
            margin = float(margins[layer, head])
            record = {"key": f"H:L{layer}H{head}", "heads": [[layer, head]], "mlp_blocks": [], "yesno": "Yes.",
                      "answer": True, "story": "x" * 30, "full_text": "x" * 30, "probabilities": P_BASELINE,
                      "cosine": {"A": margin, "B": margin - 0.05, "C": margin - 0.1}}
            if (layer, head) == (4, 4):   # "no" to the gate question: scored, but it does not count while the gate applies
                record.update(yesno="No.", answer=False)
            if (layer, head) == (6, 6):   # too short to be scored at all
                record.update(story="Ok.", full_text="Ok.", probabilities=None, cosine=None)
            if (layer, head) == (7, 7):   # a record from before search.py's gate switch: a "no", flagged and unscored
                record.update(yesno="No.", answer=False, valid=False, probabilities=None, cosine=None)
            lines.append(json.dumps(record))
    # A two-head knockout must be ignored, and a cut-off last line skipped.
    lines.append(json.dumps({"key": "H:L0H0,L5H1", "heads": [[0, 0], [5, 1]], "mlp_blocks": [], "yesno": "Yes.",
                             "answer": True, "story": "y" * 30, "full_text": "y" * 30, "probabilities": P_BASELINE,
                             "cosine": {"A": 9.0, "B": 9.0, "C": 9.0}}))
    lines.append('{"key": "H:L0H1", "heads": [[0, 1]], "mlp_bl')
    (cache_dir / "pythia-1b" / "rep1.25_cosine").mkdir(parents=True)
    (cache_dir / "pythia-1b" / "rep1.25_cosine" / "cache_cinderella.jsonl").write_text("\n".join(lines) + "\n")

    # llama2-13b-chat sandwich, classifier: an unscorable baseline, a 40 x 40 grid, A at stage 2, B and C running.
    large = {"n_layers": 40, "n_heads": 40, "d_head": 128, "d_model": 5120, "d_mlp": 13824, "n_blocks": 54}
    large_grid = [[None if (layer + head) % 3 == 0 else round(0.01 * ((layer * head) % 7), 6) for head in range(40)]
                  for layer in range(40)]
    llama_circuit = knockout([[0, 5], [3, 1]], [], 0.65, [0.1, 0.65, 0.0, 0.0, 0.25])
    llama = results_file("llama2-13b-chat", "sandwich", "classifier", large,
                         {"answer": None, "yesno": "", "story": "No.", "full_text": "No.", "valid": False,
                          "probabilities": None, "cosine": None}, {
        "A": {"baseline_value": None, "found_at_stage": "2_forward_heads", "done": True,
              "stages": {"1_single_layer_heads": stage1(large_grid, [0.3] + [0.1] * 39, knockout([[0, 0]], [], 0.3), False),
                         "2_forward_heads": sweep([("head", [0, 5], 0.2), ("head", [0, 7], 0.2), ("head", [3, 1], 0.7)],
                                                  True)},
              "pruned": {"circuit": llama_circuit, "steps": [{"removed": ["head", [0, 7]], "value": 0.65}]}}})

    # gpt2-xl sandwich, cos-sim, asked for A only, no answer cache; its answer needs escaping in Markdown.
    gpt2 = {"n_layers": 48, "n_heads": 25, "d_head": 64, "d_model": 1600, "d_mlp": 6400, "n_blocks": 25}
    story = "Spread the peanut butter | then the jelly <b>\nThen eat."
    gpt2_circuit = knockout([[0, 3]], [], 0.02, [0.95, 0.02, 0.01, 0.01, 0.01], story)
    gpt2_file = results_file("gpt2-xl", "sandwich", "cosine", gpt2,
                             {"answer": None, "yesno": "", "story": story, "full_text": story, "valid": True,
                              "probabilities": [0.95, 0.02, 0.01, 0.01, 0.01], "cosine": {"A": -0.2}}, {
        "A": {"baseline_value": -0.2, "found_at_stage": "1_single_layer_heads", "done": True,
              "stages": {"1_single_layer_heads": stage1([[None] * 25 for _ in range(48)], [0.02] * 48, gpt2_circuit, True)},
              "pruned": {"circuit": gpt2_circuit, "steps": []}}}, asked=("A",))

    for name, payload in (("pythia-1b_cinderella_rep1.25_classifier", classifier),
                          ("pythia-1b_cinderella_rep1.25_cosine", cosine),
                          ("llama2-13b-chat_sandwich_rep1.25_classifier", llama),
                          ("gpt2-xl_sandwich_rep1.25_cosine", gpt2_file)):
        (search_dir / f"{name}.json").write_text(json.dumps(payload))
    return margins


def check_hand_made():
    test_dir = TEST_DIR / "synthetic"
    # NOTE: the folder is this test's own; emptying it first
    # means a figure from an earlier run cannot pass for one this run should draw.
    shutil.rmtree(test_dir, ignore_errors=True)
    search_dir, cache_dir, out_dir = test_dir / "search", test_dir / "cache", test_dir / "report"
    search_dir.mkdir(parents=True)
    margins = write_hand_made_searches(search_dir, cache_dir)
    # Two figures left from an earlier report: one of a search that is here again but whose objective A now
    # works at stage 1 (no sweeps figure; the report must remove it), one of a search that is not here (kept).
    (out_dir / "figures").mkdir(parents=True)
    stale = "pythia-1b_cinderella_rep1.25_classifier_A_sweeps.png"
    other = "pythia-1.4b_cinderella_rep1.25_classifier_A_sweeps.png"
    for name in (stale, other):
        (out_dir / "figures" / name).write_text("an earlier figure")
    sys.argv = ["report.py", "--results-dir", str(search_dir), "--cache-dir", str(cache_dir), "--out-dir", str(out_dir)]
    report.main()
    summary = json.loads((out_dir / "search_summary.json").read_text())

    def row(tag, model, task, objective):
        return next(r for r in summary["searches"][tag]["rows"]
                    if (r["model"], r["task"], r["objective"]) == (model, task, objective))

    # Expected rows: (tag, model, task, objective) -> numbers worked out from the files above.
    stage_keys = report.STAGES
    expected = {
        ("rep1.25_classifier", "pythia-1b", "cinderella", "A"):
            dict(baseline_value=0.09, baseline_works=False, found_at_stage=stage_keys[0], best=[0.8, None, None],
                 best_value=0.8, n_components_found=1, n_components_pruned=1, circuit_text="L5H1", value=0.7,
                 argmax="Conduction/Anomic"),
        ("rep1.25_classifier", "pythia-1b", "cinderella", "B"):
            dict(baseline_value=0.03, baseline_works=False, found_at_stage=stage_keys[2], best=[0.2, 0.3, 0.6],
                 best_value=0.6, n_components_found=4, n_components_pruned=3, circuit_text="L0B3 L1H2 L1B0",
                 value=0.55, argmax="Broca"),
        ("rep1.25_classifier", "pythia-1b", "cinderella", "C"):
            dict(baseline_value=0.02, baseline_works=False, found_at_stage=None, best=[0.05, 0.2, 0.45],
                 best_value=0.45, n_components_found=None, n_components_pruned=None, circuit_text=None, value=None,
                 argmax=None),
        ("rep1.25_classifier", "llama2-13b-chat", "sandwich", "A"):
            dict(baseline_value=None, baseline_works=False, found_at_stage=stage_keys[1], best=[0.3, 0.7, None],
                 best_value=0.7, n_components_found=3, n_components_pruned=2, circuit_text="L0H5 L3H1", value=0.65,
                 argmax="Conduction/Anomic"),
        ("rep1.25_cosine", "pythia-1b", "cinderella", "A"):
            dict(baseline_value=-0.01, baseline_works=False, found_at_stage=stage_keys[1], best=[-0.005, 0.03, None],
                 best_value=0.03, n_components_found=2, n_components_pruned=2, circuit_text="L0H0 L5H1", value=0.03,
                 argmax="Conduction/Anomic"),
        ("rep1.25_cosine", "pythia-1b", "cinderella", "B"):
            dict(baseline_value=-0.05, baseline_works=False, found_at_stage=None, best=[-0.02, -0.01, 0.0],
                 best_value=0.0, n_components_found=None, n_components_pruned=None, circuit_text=None, value=None,
                 argmax=None),
        ("rep1.25_cosine", "pythia-1b", "cinderella", "C"):
            dict(baseline_value=0.01, baseline_works=True, found_at_stage=stage_keys[0], best=[0.04, None, None],
                 best_value=0.04, n_components_found=1, n_components_pruned=1, circuit_text="L2H2", value=0.04,
                 argmax="Wernicke"),
        ("rep1.25_cosine", "gpt2-xl", "sandwich", "A"):
            dict(baseline_value=-0.2, baseline_works=False, found_at_stage=stage_keys[0], best=[0.02, None, None],
                 best_value=0.02, n_components_found=1, n_components_pruned=1, circuit_text="L0H3", value=0.02,
                 argmax="Control"),
    }
    for (tag, model, task, objective), numbers in expected.items():
        got = row(tag, model, task, objective)
        best = numbers.pop("best")
        assert [got["best_value_per_stage"][stage] for stage in stage_keys] == best, (tag, model, task, objective, got)
        for field, value in numbers.items():
            assert got[field] == value, (tag, model, task, objective, field, got[field], value)
    for tag in ("rep1.25_classifier", "rep1.25_cosine"):
        assert len(summary["searches"][tag]["rows"]) == 4, summary["searches"][tag]["rows"]
        assert len(summary["searches"][tag]["no_results_file"]) == 2 * len(report.MODELS) - 2
    assert summary["searches"]["rep1.25_classifier"]["not_finished"] == [["llama2-13b-chat", "sandwich", "B"],
                                                                       ["llama2-13b-chat", "sandwich", "C"]]
    assert summary["searches"]["rep1.25_cosine"]["not_finished"] == []   # gpt2-xl was asked for A only
    # The gate: off in the classifier search of pythia-1b cinderella, on in its cos-sim search, none for sandwich.
    assert [r["gate"] for r in summary["searches"]["rep1.25_classifier"]["rows"]] == [None, False, False, False]
    assert [r["gate"] for r in summary["searches"]["rep1.25_cosine"]["rows"]] == [None, True, True, True]
    print("hand-made searches: every row's numbers as expected", flush=True)

    # The comparison: (classifier circuit, cos-sim circuit, Jaccard, classifier value of the cos-sim answer, works, argmax).
    comparison = {row["objective"]: row for row in summary["comparison"]}
    assert len(summary["comparison"]) == 3 and all(row["model"] == "pythia-1b" for row in summary["comparison"])
    assert (comparison["A"]["classifier_circuit"], comparison["A"]["cosine_circuit"], comparison["A"]["jaccard"],
            comparison["A"]["cosine_answer_classifier_value"], comparison["A"]["cosine_answer_works_under_classifier"],
            comparison["A"]["cosine_answer_argmax"]) == ("L5H1", "L0H0 L5H1", 0.5, 0.8, True, "Conduction/Anomic")
    assert (comparison["B"]["cosine_circuit"], comparison["B"]["jaccard"],
            comparison["B"]["cosine_answer_classifier_value"]) == (None, None, None)
    assert (comparison["C"]["classifier_circuit"], comparison["C"]["jaccard"],
            comparison["C"]["cosine_answer_classifier_value"], comparison["C"]["cosine_answer_works_under_classifier"],
            comparison["C"]["cosine_answer_argmax"]) == (None, None, 0.4, False, "Wernicke")
    # Only C's cos-sim search started from an unablated answer that already works.
    assert [(comparison[name]["cosine_baseline_value"], comparison[name]["cosine_baseline_works"])
            for name in "ABC"] == [(-0.01, False), (-0.05, False), (0.01, True)]
    print("hand-made searches: comparison as expected", flush=True)

    # The baselines: one per (model, task), from the first tag with a file.
    baselines = [(row["model"], row["task"], row["valid"], row["probabilities"]) for row in summary["baselines"]]
    assert baselines == [("gpt2-xl", "sandwich", True, [0.95, 0.02, 0.01, 0.01, 0.01]),
                         ("llama2-13b-chat", "sandwich", False, None),
                         ("pythia-1b", "cinderella", True, P_BASELINE)], baselines

    # The cos-sim single-head values come back from the cache: every true margin, NaN for the answers that do
    # not count under the search's gate (a "no", a "no" in the old cache format, and an answer too short).
    entry = json.loads((search_dir / "pythia-1b_cinderella_rep1.25_cosine.json").read_text())["objectives"]["A"]
    single_heads = report.cached_single_heads(cache_dir / "pythia-1b" / "rep1.25_cosine" / "cache_cinderella.jsonl")
    assert len(single_heads) == 128, len(single_heads)
    recovered = report.single_head_values(entry, "A", single_heads, True)
    expected_values = margins.copy()
    for layer, head in UNCOUNTED_HEADS:
        expected_values[layer, head] = np.nan
    assert np.array_equal(recovered, expected_values, equal_nan=True)
    assert (np.isnan(report.single_head_values(entry, "A", {}, True)).sum()
            == int((margins < 0).sum()) + len(UNCOUNTED_HEADS))
    # Without the gate the scored "no" counts, as in search.py; the answer too short to score still does not.
    ungated = report.single_head_values(entry, "A", {head: single_heads[head] for head in [(4, 4), (6, 6)]}, False)
    assert ungated[4, 4] == margins[4, 4] and np.isnan(ungated[6, 6])
    for record in single_heads.values():
        if "valid" not in record:
            for gate in (True, False):
                assert report.counts(record, gate) == GateOnly(gate).valid(record)
    print("hand-made searches: cos-sim single-head values recovered from the cache", flush=True)

    # The Markdown: a table row per objective, bold where the baseline works, escaped answers.
    markdown = (out_dir / "search_summary.md").read_text()
    sections = markdown.split("\n## ")
    table_rows = {section.splitlines()[0]: sum(line.startswith("| ") for line in section.splitlines()) - 1
                  for section in sections[1:]}
    assert table_rows == {"Searches `rep1.25_classifier`": 4, "Searches `rep1.25_cosine`": 4,
                          "Unablated baselines": 3,
                          "Classifier search (`rep1.25_classifier`) against cos-sim search (`rep1.25_cosine`)": 3,
                          "Figures": -1}, table_rows
    assert "| pythia-1b | cinderella | C | **0.010** | 1 single-layer heads |" in markdown
    assert "Spread the peanut butter \\| then the jelly &lt;b> ⏎ Then eat." in markdown
    assert "Not finished yet: llama2-13b-chat sandwich B, llama2-13b-chat sandwich C." in markdown
    assert markdown.count("Searched without the Cinderella gate") == 1
    assert "so a knockout's \"no\" counts too): pythia-1b cinderella." in markdown
    assert "| pythia-1b | cinderella | C | **0.010** | none | `L2H2` |" in markdown   # the comparison's C row
    print("hand-made searches: Markdown tables as expected", flush=True)

    # Every figure: stage 1 for each objective, sweeps where stage 2 ran, an overview per tag.
    names = ["llama2-13b-chat_sandwich_rep1.25_classifier_A_stage1", "llama2-13b-chat_sandwich_rep1.25_classifier_A_sweeps",
             "pythia-1b_cinderella_rep1.25_classifier_A_stage1", "pythia-1b_cinderella_rep1.25_classifier_B_stage1",
             "pythia-1b_cinderella_rep1.25_classifier_B_sweeps", "pythia-1b_cinderella_rep1.25_classifier_C_stage1",
             "pythia-1b_cinderella_rep1.25_classifier_C_sweeps", "pythia-1b_cinderella_rep1.25_cosine_A_stage1",
             "pythia-1b_cinderella_rep1.25_cosine_A_sweeps", "pythia-1b_cinderella_rep1.25_cosine_B_stage1",
             "pythia-1b_cinderella_rep1.25_cosine_B_sweeps", "pythia-1b_cinderella_rep1.25_cosine_C_stage1",
             "gpt2-xl_sandwich_rep1.25_cosine_A_stage1", "overview_rep1.25_classifier", "overview_rep1.25_cosine"]
    expected_files = {f"{name}.{extension}" for name in names for extension in ("pdf", "png")}
    written = {path.name for path in (out_dir / "figures").iterdir()}
    assert written == expected_files | {other}, (written - expected_files, expected_files - written)
    assert all((out_dir / "figures" / name).stat().st_size > 5000 for name in expected_files)
    print(f"hand-made searches: all {len(expected_files)} figure files written, the stale one removed", flush=True)


class GateOnly:
    """The one part of search.Evaluator that search.Objective.value uses: valid(), under a search's gate.

    NOTE: a real Evaluator loads a model and the classifier;
    borrowing search.py's own valid() keeps the rule these checks read the
    cache with the very one the search used, without loading anything.
    """

    def __init__(self, gate: bool):
        self.gate = gate

    valid = search.Evaluator.valid


def read_cache(path: Path) -> dict:
    """search.py's answer cache as {key: record}; later lines win, as when search.py loads it."""
    cache = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        cache[record["key"]] = record
    return cache


def cached_value(cache: dict, objective, heads, mlp_blocks=()):
    """An objective's value for the cached answer of a knockout (None if the answer did not count)."""
    return objective.value(cache[Intervention.knockout(heads=heads, mlp_blocks=mlp_blocks).key()])


def check_real():
    files = sorted((RESULTS / "search").glob("*.json"))
    baselines = {}
    n_checked = 0
    for path in files:
        model, task, tag = path.stem.split("_", 2)
        results = json.loads(path.read_text())
        baselines.setdefault((model, task), []).append(results["baseline"])
        cache_path = RUNS / "fiveway" / model / tag / f"cache_{task}.jsonl"
        if not cache_path.exists():
            print(f"  {path.name}: no answer cache on this disk, skipped", flush=True)
            continue
        cache = read_cache(cache_path)
        kind = results["config"]["objective_kind"]
        n_layers, n_heads = results["geometry"]["n_layers"], results["geometry"]["n_heads"]
        single_heads = report.cached_single_heads(cache_path)
        # Files written before search.py's gate switch have no "gate"; those searches always applied it.
        gate = results.get("gate", True)
        evaluator = GateOnly(gate)
        # report.counts must count exactly the answers search.py counts, on every cached answer.
        assert all(report.counts(record, gate) == evaluator.valid(record) for record in cache.values()), path.name
        for name, entry in results["objectives"].items():
            objective = search.Objective(name, kind, evaluator)
            assert objective.value(cache[Intervention().key()]) == entry["baseline_value"], (path.name, name)

            # Stage 1, rebuilt from the cache: each single head, then each layer's top-k heads,
            # ranked as search.py ranks them (an answer that did not count as -1).
            singles = np.full((n_layers, n_heads), -1.0)
            for layer in range(n_layers):
                for head in range(n_heads):
                    if cached_value(cache, objective, [(layer, head)]) is not None:
                        singles[layer, head] = cached_value(cache, objective, [(layer, head)])
            candidates = [v for v in singles.ravel() if v != -1.0]
            for layer in range(n_layers):
                ranked = list(np.argsort(-singles[layer], kind="stable"))
                for k in [k for k in search.K_VALUES if k <= n_heads]:
                    top_k = cached_value(cache, objective, [(layer, int(head)) for head in ranked[:k]])
                    if top_k is not None:
                        candidates.append(top_k)
            assert abs(report.best_values(entry)["1_single_layer_heads"] - max(candidates)) < 1e-9, (path.name, name)

            # The file's single-head values are the cached ones; for the cosine objective the
            # report fills exactly the file's nulls, with negative margins or NaN (unscored).
            stored = np.array([[np.nan if v is None else v for v in row]
                               for row in entry["stages"]["1_single_layer_heads"]["single_head_values"]])
            from_cache = np.where(singles == -1.0, np.nan, singles)
            if kind == "classifier":
                assert np.allclose(stored, from_cache, atol=1e-6, equal_nan=True), (path.name, name)
            else:
                recovered = report.single_head_values(entry, name, single_heads, gate)
                assert np.allclose(recovered, from_cache, atol=1e-12, equal_nan=True), (path.name, name)
                filled = np.isnan(stored) & ~np.isnan(recovered)
                assert np.all(recovered[filled] < 0), (path.name, name)
                assert np.allclose(stored[~np.isnan(stored)], recovered[~np.isnan(stored)], atol=1e-6)

            # The sweeps: each step's value is that of everything committed so far.
            for stage in ("2_forward_heads", "3_forward_heads_mlp"):
                if stage not in entry["stages"]:
                    continue
                heads, blocks = [], []
                for step in entry["stages"][stage]["trajectory"]:
                    (heads if step["kind"] == "head" else blocks).append(tuple(step["added"]))
                    assert step["n_components"] == len(heads) + len(blocks)
                    assert abs(cached_value(cache, objective, heads, blocks) - step["value"]) < 1e-9, \
                        (path.name, name, stage, step)

            # The pruned circuit is a cached answer, and its compact text names exactly its components.
            if "pruned" in entry:
                circuit = entry["pruned"]["circuit"]
                record = cache[Intervention.knockout(heads=circuit["heads"], mlp_blocks=circuit["mlp_blocks"]).key()]
                assert record["probabilities"] == circuit["probabilities"] and record["story"] == circuit["story"]
                parsed = set()
                for token in report.compact(circuit).split():
                    kind_letter = "H" if "H" in token else "B"
                    layer, index = token[1:].split(kind_letter)
                    parsed.add(("head" if kind_letter == "H" else "block", int(layer), int(index)))
                assert parsed == report.components(circuit), (path.name, name)
            n_checked += 1
        print(f"  {path.name}: {len(results['objectives'])} objectives agree with {cache_path.name} "
              f"({len(cache)} cached answers)", flush=True)
    for (model, task), stored in baselines.items():
        for other in stored[1:]:
            assert other["story"] == stored[0]["story"], (model, task)
            if other["probabilities"] is not None:
                assert max(abs(p - q) for p, q in zip(other["probabilities"], stored[0]["probabilities"], strict=True)) < 1e-5
    print(f"real searches: {n_checked} objectives checked against their caches; "
          f"baselines agree across tags for {sum(len(s) > 1 for s in baselines.values())} model-task pairs", flush=True)

    out_dir = TEST_DIR / "real"
    sys.argv = ["report.py", "--out-dir", str(out_dir)]
    report.main()
    summary = json.loads((out_dir / "search_summary.json").read_text())
    n_rows = sum(len(section["rows"]) for section in summary["searches"].values())
    print(f"real searches: report.main() wrote {n_rows} rows to {out_dir}", flush=True)


def main():
    assert report.CLASSES == scorer.CLASSES
    assert report.OBJECTIVE_COLUMNS == scorer.OBJECTIVES
    assert report.THRESHOLDS == {"classifier": scorer.SUCCESS_THRESHOLD,
                                 "cosine": scorer.CosineObjective.SUCCESS_THRESHOLD}
    assert report.STAGES == ["1_single_layer_heads", "2_forward_heads", "3_forward_heads_mlp"]
    assert report.MIN_STORY_CHARS == prompts.MIN_STORY_CHARS
    for text in ["", "tOoOO" + "\u200d" * 170, "Peanut Butter, Jelly, Bread!", "soft\u00adhyphen", "a b\nc"]:
        assert report.answer_length(text) == prompts.answer_length(text)
    assert not prompts.long_enough("tOoOO" + "\u200d" * 170) and prompts.long_enough("Peanut Butter, Jelly, Bread!")
    print("constants match scorer.py and prompts.py", flush=True)
    check_hand_made()
    check_real()
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
