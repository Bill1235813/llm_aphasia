"""Where and why the engine's answer parts from a stored answer of the earlier study.

For one model, task and decoding, regenerate the unablated answer, find the
first generated token that differs from the stored transformer_lens answer,
and print the decision there: the two candidate tokens' scores after the
decoding's logit processing. A gap of a few thousandths of a logit is a
near-tie that float rounding (another device, transformer_lens's folded
weights) can flip; a large gap would mean the two pipelines compute different
things.

Usage:
    GPUS=0 CPUS=16 bash slurm/fiveway_submit.sh divergence 32G fiveway/test_divergence.py \
        --model pythia-1b --task cinderella --decoding rep1.25
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import REPETITION_PENALTY, Engine, Intervention  # noqa: E402
from fiveway.paths import DATA  # noqa: E402
from fiveway.prompts import STORY_TOKENS, task_prompt_ids  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--decoding", required=True)
    args = parser.parse_args()
    engine = Engine(args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    records = json.loads((DATA / "interp" / f"exp00_decoding_{args.model}.json").read_text())["records"]
    stored = next(r for r in records if r["task"] == args.task and r["decoding"] == args.decoding)["story"]
    _, prompt_ids = task_prompt_ids(engine, args.task)
    ours = engine.generate(prompt_ids, [Intervention()], STORY_TOKENS, args.decoding)[0]

    # Token ids of both answers, as continuations of the prompt.
    separator = "" if engine.spec.is_chat else " "
    our_ids = engine.tokenizer(separator + ours, add_special_tokens=False)["input_ids"]
    stored_ids = engine.tokenizer(separator + stored, add_special_tokens=False)["input_ids"]
    step = next(i for i, (a, b) in enumerate(zip(our_ids, stored_ids)) if a != b)
    context = torch.tensor([prompt_ids + our_ids[:step]])
    with torch.no_grad():
        logits = engine.model(context.to(engine.device)).logits[:, -1].float().cpu()
    scores = RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY[args.decoding])(context, logits.clone())
    ours_token, stored_token = our_ids[step], stored_ids[step]
    # transformer_lens loads models with center_unembed=True, which subtracts
    # each position's mean logit. Argmax ignores that shift; the repetition
    # penalty does not, since it divides positive and multiplies negative logits.
    centered = RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY[args.decoding])(
        context, logits - logits.mean(dim=-1, keepdim=True))
    print(f"with mean-centered logits (transformer_lens): ours {float(centered[0, our_ids[step]]):.6f} "
          f"stored {float(centered[0, stored_ids[step]]):.6f}  -> picks "
          f"{engine.tokenizer.decode([int(centered[0].argmax())])!r}")
    top = torch.topk(scores[0], 5)
    print(f"first difference at generated token {step}:")
    print(f"  context end: {engine.tokenizer.decode(our_ids[max(0, step - 12):step])!r}")
    print(f"  ours   {engine.tokenizer.decode([ours_token])!r}: score {scores[0, ours_token]:.6f}")
    print(f"  stored {engine.tokenizer.decode([stored_token])!r}: score {scores[0, stored_token]:.6f}")
    print(f"  gap {float(scores[0, ours_token] - scores[0, stored_token]):.6f} logits")
    print("  top 5:", [(engine.tokenizer.decode([int(i)]), round(float(v), 5)) for v, i in zip(top.values, top.indices)])


if __name__ == "__main__":
    main()
