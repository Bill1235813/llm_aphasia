"""Checks of em_attention.py's pruning of large grown sets (shrink() then prune()), on a made-up objective (CPU).

A grown set is the shortest prefix of a ranking that works, so its weakest
members cannot be cut from the end. Here the objective works only when a
hidden handful of components, spread through the ranking, are all knocked
out. shrink() followed by batched_prune() must come down to exactly that handful, and
do it with far fewer answers than prune() alone (about n^2 / 2): it is what
lets em_attention.py prune grown sets of hundreds of heads.

Usage:
    python src/fiveway/test_em_attention.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_attention import batched_prune, prune_found, shrink  # noqa: E402
from fiveway.search import prune  # noqa: E402


class HiddenSetObjective:
    """Works when every hidden component is knocked out; among working sets, smaller ones score a little higher."""

    def __init__(self, hidden: set):
        self.hidden = hidden

    def value(self, record):
        chosen = set(record["chosen"])
        return 0.9 - 0.0001 * len(chosen) if self.hidden <= chosen else 0.1

    def works(self, value):
        return value is not None and value > 0.5


class CountingEvaluator:
    """Stands in for search.Evaluator: a record per knockout, and a count of the distinct knockouts asked for."""

    def __init__(self):
        self.seen = set()

    def records(self, interventions):
        self.seen.update(intervention.key() for intervention in interventions)
        return [{"chosen": [(layer, head) for layer, head in intervention.heads], "heads": [list(h) for h in intervention.heads],
                 "mlp_blocks": [], "lesions": [], "probabilities": None, "cosine": None, "answer": None, "story": ""}
                for intervention in interventions]


def main():
    ranked = [("head", (layer, head)) for layer in range(41) for head in range(16)]   # 656 components, strongest first
    hidden = {ranked[i][1] for i in (3, 90, 211, 400, 655)}   # 655: the last one, which made the prefix work
    objective, log = HiddenSetObjective(hidden), lambda message: None

    evaluator = CountingEvaluator()
    found = {"heads": [list(c) for _, c in ranked], "mlp_blocks": [], "lesions": [], "story": "", "value": 0.8,
             "probabilities": None, "cosine": None, "answer": None}
    kept, steps = shrink(evaluator, objective, ranked, log)
    assert {c for _, c in kept} >= hidden, "shrink dropped a needed component"
    shrink_answers = len(evaluator.seen)
    pruned = prune_found(evaluator, objective, found, ranked, log)
    final = {tuple(h) for h in pruned["circuit"]["heads"]}
    assert final == hidden, (sorted(final), sorted(hidden))
    print(f"shrink: 656 -> {len(kept)} components with {shrink_answers} answers in {len(steps)} steps; "
          f"shrink + prune: exactly the 5 hidden components with {len(evaluator.seen)} answers "
          f"(prune alone would need about {656 * 657 // 2:,})", flush=True)

    # prune() alone on a small set gives the same minimal set.
    small = [("head", c) for c in sorted(hidden)] + ranked[:20]
    alone = prune(CountingEvaluator(), objective, {"heads": [list(c) for _, c in small], "mlp_blocks": [], "lesions": [],
                                                    "story": "", "value": 0.8, "probabilities": None, "cosine": None,
                                                    "answer": None}, log)
    assert {tuple(h) for h in alone["circuit"]["heads"]} == hidden
    print("prune alone on a 25-component set: the same 5 hidden components", flush=True)

    # batched_prune() alone, with no shrink first, on 160 components.
    evaluator = CountingEvaluator()
    start = [("head", c) for c in sorted(hidden)] + [c for c in ranked[:200] if c[1] not in hidden][:155]
    kept, steps, minimal = batched_prune(evaluator, objective, start, log)
    assert {c for _, c in kept} == hidden and minimal, sorted(c for _, c in kept)
    print(f"batched prune alone: 160 -> exactly the 5 hidden components in {len(steps)} rounds with "
          f"{len(evaluator.seen)} answers (prune alone would need about {160 * 161 // 2:,})", flush=True)

    # The budget: when a round would not fit in PRUNE_BUDGET answers, pruning stops there and says so.
    import fiveway.em_attention as em_attention
    saved_budget, em_attention.PRUNE_BUDGET = em_attention.PRUNE_BUDGET, len(start) - 1
    try:
        kept, steps, minimal = em_attention.batched_prune(CountingEvaluator(), objective, start, log)
        assert kept == start and not minimal and not steps, (len(kept), minimal)
    finally:
        em_attention.PRUNE_BUDGET = saved_budget
    print("budget: a round that does not fit stops pruning, and the set is reported as not minimal", flush=True)
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
