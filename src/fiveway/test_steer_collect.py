"""Check steer_collect.py's batched residual means against plain one-transcript forward passes.

mean_residuals() runs several transcripts per forward pass, sorted by length
and right-padded, and averages each block's output over the transcript's
tokens inside a hook. For a few transcripts of different lengths (the longest
one cut at the context limit), its means must equal those of each transcript
run alone, without padding, read through Hugging Face's own
output_hidden_states for blocks 0 to n-2 and through a hook on the final
normalization's input for the last block (output_hidden_states reports the
last block only after that normalization). The check also prints the
throughput, to size the real runs. A CPU job is enough.

Usage:
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-check 32G fiveway/test_steer_collect.py --model pythia-1b
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine  # noqa: E402
from fiveway.prompts import task_prompt_ids  # noqa: E402
from fiveway.steer_collect import (FINAL_NORM, MAX_TOKENS, answer_token_ids, load_transcripts,  # noqa: E402
                                   mean_residuals)

TOLERANCE = 1e-4   # largest relative difference allowed between the batched and the single-transcript means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="cinderella")
    parser.add_argument("--n-transcripts", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    engine = Engine(args.model, device)
    log(f"{args.model} on {device} ({torch.get_num_threads()} CPU threads): {engine.geometry}")
    prompt_ids = task_prompt_ids(engine, args.task)[1]
    room = min(MAX_TOKENS, engine.model.config.max_position_embeddings) - len(prompt_ids)
    rows = load_transcripts(args.task, None)
    full_ids = [answer_token_ids(engine, row["text_response"]) for row in rows]
    # Transcripts spread over the length range, from the shortest to the longest
    # (which is cut at the context limit), shuffled so that their order differs
    # from the sorted order mean_residuals batches them in: a mix-up there would show.
    by_length = np.argsort([len(ids) for ids in full_ids], kind="stable")
    picks = by_length[np.linspace(0, len(rows) - 1, args.n_transcripts).astype(int)]
    picks = picks[np.random.default_rng(0).permutation(len(picks))]
    transcript_ids = [full_ids[index][:room] for index in picks]
    log(f"{args.task}: transcripts {picks.tolist()} with {[len(ids) for ids in transcript_ids]} tokens "
        f"(the longest has {max(len(full_ids[index]) for index in picks)} before the cut)")

    batch_start = time.time()
    batched = mean_residuals(engine, prompt_ids, transcript_ids, args.batch_size, log)
    seconds = time.time() - batch_start
    n_tokens = sum(len(prompt_ids) + len(ids) for ids in transcript_ids)
    log(f"batched: {n_tokens} tokens in {seconds:.1f} s ({n_tokens / seconds:.0f} tokens/s)")

    final_norm = getattr(engine.model.base_model, FINAL_NORM[engine.spec.family])
    worst = 0.0
    for position, ids in enumerate(transcript_ids):
        last_block = {}
        handle = final_norm.register_forward_pre_hook(lambda module, args: last_block.update(output=args[0]))
        with torch.no_grad():
            output = engine.model.base_model(input_ids=torch.tensor([prompt_ids + ids], device=device),
                                             output_hidden_states=True, use_cache=False)
        handle.remove()
        # NOTE: hidden_states: n_layers + 1 tensors of (1, positions, d_model):
        # the embeddings, blocks 0 to n-2, then the last block after the final norm.
        assert len(output.hidden_states) == engine.geometry["n_layers"] + 1
        hidden = list(output.hidden_states[1:-1]) + [last_block["output"]]
        single = np.stack([layer_hidden[0, len(prompt_ids):].mean(dim=0).float().cpu().numpy()
                           for layer_hidden in hidden])
        relative = np.linalg.norm(batched[position] - single, axis=-1) / np.linalg.norm(single, axis=-1)
        worst = max(worst, float(relative.max()))
        log(f"  transcript {position} ({len(ids)} tokens): "
            f"largest relative difference over layers {relative.max():.2e}")
    log(f"largest relative difference {worst:.2e} (tolerance {TOLERANCE:.0e})")
    assert worst < TOLERANCE, "batched means differ from single-transcript means"
    log("passed")


if __name__ == "__main__":
    main()
