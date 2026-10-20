"""GPU checks of the batched engine (run as a SLURM job, one model per run).

1. Reproduction. The unablated answers of the earlier transformer_lens study
   (data/interp/exp00_decoding_<model>.json: both tasks, greedy and rep1.25)
   must come out again, and for Llama-2-7B-chat also the stories of the
   three-head circuit's greedy path (diag_llama2_v4_exp02_greedy_stories.json).
   transformer_lens folds layer norms into the weights, so a few texts may
   part ways after a near-tie in the logits; the report gives the length of
   the shared beginning.
2. Batch invariance. An intervention gives bit-identical logits and the same
   text whatever else is in its batch and wherever it sits.
3. Neutral interventions. Patching a head that is not knocked out, and
   steering with alpha 0, leave the text unchanged.
4. Throughput: seconds per batch of story generations.

Usage:
    bash slurm/fiveway_submit.sh test-engine 48G fiveway/test_engine.py --model pythia-1b
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import DATA, RUNS  # noqa: E402
from fiveway.prompts import STORY_TOKENS, YESNO_TOKENS, task_prompt_ids  # noqa: E402


def shared_prefix(a: str, b: str) -> int:
    """Number of leading characters two texts share, ignoring whitespace differences."""
    a, b = " ".join(a.split()), " ".join(b.split())
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, {torch.get_num_threads()} CPU threads", flush=True)
    start = time.time()
    engine = Engine(args.model, device)
    print(f"loaded {args.model} in {time.time() - start:.0f} s  geometry {engine.geometry}", flush=True)
    report = {"model": args.model, "reproduction": [], "invariance": {}, "neutral": {}, "timing": {}}

    # --- 1. reproduction of the earlier unablated answers -----------------
    reference_path = DATA / "interp" / f"exp00_decoding_{args.model}.json"
    if reference_path.exists():
        records = json.loads(reference_path.read_text())["records"]
        for record in records:
            if record["decoding"] not in ("greedy", "rep1.25"):
                continue
            yesno_ids, story_ids = task_prompt_ids(engine, record["task"])
            story = engine.generate(story_ids, [Intervention()], STORY_TOKENS, record["decoding"])[0]
            entry = {"task": record["task"], "decoding": record["decoding"],
                     "story_exact": " ".join(story.split()) == " ".join(record["story"].split()),
                     "story_shared_chars": shared_prefix(story, record["story"]),
                     "story_chars": len(" ".join(record["story"].split()))}
            if yesno_ids is not None:
                yesno = engine.generate(yesno_ids, [Intervention()], YESNO_TOKENS, record["decoding"])[0]
                entry["yesno_exact"] = " ".join(yesno.split()) == " ".join(record["yesno_continuation"].split())
                entry["yesno"] = yesno
            report["reproduction"].append(entry)
            print("  reproduce", entry, flush=True)
            if not entry["story_exact"]:
                print(f"    ours  : {story[:300]!r}\n    theirs: {record['story'][:300]!r}", flush=True)

    circuit_path = DATA / "interp" / "diag_llama2_v4_exp02_greedy_stories.json"
    if args.model == "llama2-7b-chat":
        steps = json.loads(circuit_path.read_text())
        _, story_ids = task_prompt_ids(engine, "cinderella")
        heads, interventions = [], []
        for step in steps:
            heads.append(tuple(step["added"]))
            interventions.append(Intervention.knockout(heads=heads))
        # The story dump used 220 new tokens.
        stories = engine.generate(story_ids, interventions, 220, "greedy")
        for step, story in zip(steps, stories):
            entry = {"step": step["step"], "heads": len(heads[:step["step"] + 1]),
                     "exact": " ".join(story.split()) == " ".join(step["story"].split()),
                     "shared_chars": shared_prefix(story, step["story"]),
                     "chars": len(" ".join(step["story"].split()))}
            report["reproduction"].append({"circuit_step": entry})
            print("  circuit", entry, flush=True)
            if not entry["exact"]:
                print(f"    ours  : {story[:200]!r}\n    theirs: {step['story'][:200]!r}", flush=True)

    # --- 2. batch invariance ---------------------------------------------
    _, story_ids = task_prompt_ids(engine, "sandwich")
    n_layers, n_heads, n_blocks = (engine.geometry[k] for k in ("n_layers", "n_heads", "n_blocks"))
    probes = [Intervention(),
              Intervention.knockout(heads=[(0, 1)]),
              Intervention.knockout(heads=[(1, 2), (n_layers // 2, n_heads - 1)]),
              Intervention.knockout(mlp_blocks=[(0, 3)]),
              Intervention.knockout(heads=[(0, 0), (2, 1)], mlp_blocks=[(1, n_blocks - 1)])]
    fillers = [Intervention.knockout(heads=[(layer, layer % n_heads)]) for layer in range(n_layers)]
    order_a = probes + fillers[:7]
    order_b = fillers[7:12] + list(reversed(probes)) + fillers[:3]
    logits_a = engine.next_token_logits(story_ids, order_a)[:len(probes)]
    logits_b = engine.next_token_logits(story_ids, order_b)[5:5 + len(probes)].flip(0)
    texts_a = engine.generate(story_ids, order_a, STORY_TOKENS)[:len(probes)]
    texts_b = list(reversed(engine.generate(story_ids, order_b, STORY_TOKENS)[5:5 + len(probes)]))
    single = [engine.generate(story_ids, [probe], STORY_TOKENS)[0] for probe in probes]
    report["invariance"] = {
        "logits_bitwise_equal": bool(torch.equal(logits_a, logits_b)),
        "logits_max_abs_diff": float((logits_a - logits_b).abs().max()),
        "texts_equal_across_orders": texts_a == texts_b,
        "texts_equal_alone": texts_a == single,
        "probe_texts_distinct": len(set(texts_a)),
    }
    print("  invariance", report["invariance"], flush=True)

    # --- 3. neutral interventions -------------------------------------------
    clean = engine.generate(story_ids, [Intervention()], STORY_TOKENS)[0]
    engine.vectors["random"] = torch.randn(n_layers, engine.geometry["d_model"], device=device)
    neutral = [Intervention(patch_heads=((0, 1), (1, 2))),
               Intervention(steer=((n_layers // 2, "random", 0.0),)),
               Intervention(patch_blocks=((0, 1), (2, 3)))]
    texts = engine.generate(story_ids, neutral, STORY_TOKENS)
    knocked = engine.generate(story_ids, [Intervention.knockout(heads=[(0, 1), (1, 2)]),
                                          Intervention(heads=((0, 1), (1, 2)), patch_heads=((0, 1), (1, 2)))],
                              STORY_TOKENS)
    report["neutral"] = {"patch_without_knockout_equals_clean": texts[0] == clean,
                         "steer_alpha0_equals_clean": texts[1] == clean,
                         "block_patch_without_knockout_equals_clean": texts[2] == clean,
                         "knockout_text": knocked[0][:200], "knockout_patched_text": knocked[1][:200]}
    print("  neutral", report["neutral"], flush=True)

    # --- 4. throughput: one full batch of story generations -----------------
    batch = [fillers[i % len(fillers)] for i in range(engine.batch_size)]
    start = time.time()
    engine.generate(story_ids, batch, STORY_TOKENS)
    report["timing"] = {"seconds_per_full_batch": time.time() - start, "batch_size": engine.batch_size,
                        "max_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                              if device.type == "cuda" else None)}
    print("  timing", report["timing"], flush=True)

    out = RUNS / "tests" / f"test_engine_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
