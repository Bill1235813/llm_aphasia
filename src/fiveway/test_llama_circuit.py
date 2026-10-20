"""Replay the earlier study's three-head circuit on Llama-2-7B-chat with the engine.

The earlier transformer_lens pipeline stored the greedy Cinderella story after
each step of its cross-layer search (data/interp/diag_llama2_v4_exp02_greedy_stories.json,
220 new tokens): L0H25, then L1H8, then L1H29 (the circuit), then more heads.
Generating the clean story and the first steps in one batch checks the chat
template, the head knockout (zeroing a head's slice of the o_proj input
against transformer_lens's hook_z) and the decoding together. The earlier
unablated greedy story (exp00_decoding_llama2-7b-chat.json, 180 tokens) is
checked too.

Usage (CPU is enough, one batch):
    GPUS=0 CPUS=32 bash slurm/fiveway_submit.sh llama-circuit 96G fiveway/test_llama_circuit.py
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import DATA  # noqa: E402
from fiveway.prompts import task_prompt_ids  # noqa: E402
from fiveway.test_engine import shared_prefix  # noqa: E402

N_STEPS = 5   # the clean story plus the first five greedy steps fit in one batch


def main():
    start = time.time()
    engine = Engine("llama2-7b-chat", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    steps = json.loads((DATA / "interp" / "diag_llama2_v4_exp02_greedy_stories.json").read_text())[:N_STEPS]
    _, story_ids = task_prompt_ids(engine, "cinderella")
    heads, interventions = [], [Intervention()]
    for step in steps:
        heads.append(tuple(step["added"]))
        interventions.append(Intervention.knockout(heads=heads))
    stories = engine.generate(story_ids, interventions, 220, "greedy")
    print(f"generated {len(stories)} stories in {time.time() - start:.0f} s", flush=True)

    stored_clean = next(r for r in json.loads((DATA / "interp" / "exp00_decoding_llama2-7b-chat.json").read_text())["records"]
                        if r["task"] == "cinderella" and r["decoding"] == "greedy")["story"]
    # The clean story was stored at 180 new tokens; compare the stored length only.
    shared = shared_prefix(stories[0], stored_clean)
    print(f"clean: shared {shared} of {len(' '.join(stored_clean.split()))} characters", flush=True)
    for step, story in zip(steps, stories[1:]):
        shared = shared_prefix(story, step["story"])
        total = len(" ".join(step["story"].split()))
        print(f"step {step['step']} (+L{step['added'][0]}H{step['added'][1]}): shared {shared} of {total} characters"
              f"{'  EXACT' if shared == total else ''}", flush=True)
        if shared < total:
            print(f"    ours  : {story[:240]!r}\n    stored: {step['story'][:240]!r}", flush=True)


if __name__ == "__main__":
    main()
