"""Check the engine's weight lesions (Engine.add_lesions) against zeroing the weights themselves.

em_search.py lesions MLP weight matrices through a forward hook that
subtracts x @ D per batch element, where D holds the matrix's top-1% weights
(em_baseline.py) and zeros elsewhere. Here, on the Cinderella story prompt:

  1. Same model. For a few lesions (one matrix, two, twenty), the hook's
     next-token logits are those of the model whose weights em_baseline.py's
     lesioned() really zeroed, up to float32 rounding, with the same top
     token; and the greedy text of the first CHECK_TOKENS tokens is the same.
  2. Batch independence. A lesioned element's logits do not depend on the
     other elements of its batch (as test_engine.py checks for head knockouts).
  3. Restoring. A lesion with every lesioned matrix's clean prompt outputs
     written back (patch_lesions) gives the clean model's logits after the
     prompt: on the prompt pass nothing else differs from the clean run. (Not
     bit for bit: the clean outputs are read from a separate run of the prompt
     alone, whose matrix kernels round differently from a padded batch's.)
     Only for o, gate, up and down: q, k and v of GPT-2 and Pythia share one
     fused module, which the engine patches whole or not at all.

--kinds q k v o checks the attention matrices' lesions (em_attention.py),
including GPT-2's and Pythia's fused q/k/v matrices, whose lesions take rows
out of one shared weight.

Usage (CPU, a few minutes):
    GPUS=0 CPUS=16 bash slurm/fiveway_submit.sh em-search-check 48G fiveway/test_em_search.py --model gpt2-xl
    GRES=gpu:1 bash slurm/fiveway_submit.sh em-attention-check 48G fiveway/test_em_search.py \\
        --model pythia-1b --kinds q k v o
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import find_submodules, lesioned, load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import RUNS  # noqa: E402
from fiveway.prompts import task_prompt_ids  # noqa: E402

CHECK_TOKENS = 30
MAX_LOGIT_DIFFERENCE = 1e-3   # float32 rounding of x @ W - x @ D against x @ (W - D), over 48 layers
MAX_RESTORE_DIFFERENCE = 1e-4   # float32 rounding of a batch of one against a padded batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-xl")
    parser.add_argument("--kinds", nargs="+", default=["gate", "up", "down"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    submodules = [submodule for submodule in find_submodules(engine.model) if submodule.kind in args.kinds]
    names = [submodule.name for submodule in submodules]
    load_lesions(engine, names)
    modules_by_name = {submodule.name: submodule.module for submodule in submodules}
    top_positions = torch.load(RUNS / "fiveway" / "em_baseline" / args.model / "top1pct.pt")["positions"]
    _, story_ids = task_prompt_ids(engine, "cinderella")

    lesion_sets = [names[:1], [names[21], names[6]], names[::5][:20]]
    clean = engine.next_token_logits(story_ids, [Intervention()])[0]

    # 1. The hook against zeroed weights.
    for lesions in lesion_sets:
        hooked = engine.next_token_logits(story_ids, [Intervention.knockout(lesions=lesions)])[0]
        with lesioned(modules_by_name, top_positions, lesions):
            zeroed = engine.next_token_logits(story_ids, [Intervention()])[0]
            zeroed_text = engine.generate(story_ids, [Intervention()], CHECK_TOKENS, "rep1.25")[0]
        hooked_text = engine.generate(story_ids, [Intervention.knockout(lesions=lesions)], CHECK_TOKENS, "rep1.25")[0]
        difference = (hooked - zeroed).abs().max().item()
        moved = (zeroed - clean).abs().max().item()
        print(f"{len(lesions):2d} matrices: max |hook - zeroed| {difference:.2e} (the lesion itself moves the logits "
              f"by up to {moved:.2f}); same top token {hooked.argmax() == zeroed.argmax()}; "
              f"same text {hooked_text == zeroed_text}\n    {hooked_text[:90]!r}", flush=True)
        assert difference < MAX_LOGIT_DIFFERENCE and hooked.argmax() == zeroed.argmax(), lesions
        assert hooked_text == zeroed_text, (hooked_text, zeroed_text)

    # 2. Batch independence: each lesion alone (padded with copies of itself) and all of them in one batch.
    together = engine.next_token_logits(story_ids, [Intervention.knockout(lesions=lesions) for lesions in lesion_sets])
    for row, lesions in enumerate(lesion_sets):
        alone = engine.next_token_logits(story_ids, [Intervention.knockout(lesions=lesions)])[0]
        assert torch.equal(alone, together[row]), lesions
    print("batch independence: every lesion's logits are bit-identical alone and in a mixed batch", flush=True)

    # 3. Restoring every lesioned matrix on the prompt gives the clean prompt pass back.
    for lesions in [[name for name in lesions if name.split(".")[1] not in ("q", "k", "v")] for lesions in lesion_sets]:
        if not lesions:
            continue
        intervention = Intervention(lesions=tuple(lesions), patch_lesions=tuple(lesions))
        restored = engine.next_token_logits(story_ids, [intervention])[0]
        difference = (restored - clean).abs().max().item()
        print(f"{len(lesions):2d} matrices restored on the prompt: max |restored - clean| {difference:.2e}", flush=True)
        assert difference < MAX_RESTORE_DIFFERENCE and restored.argmax() == clean.argmax(), lesions
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
