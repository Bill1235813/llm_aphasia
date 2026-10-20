"""Check steer_generate.py's settings against steering hooks written from scratch.

steer_generate.py turns each setting into an engine Intervention: a vector added
after block L with alpha = r * resid_norm[L] / |v[L]|, a direction projected out
after every block, the knockout search's heads zeroed. The engine then applies a
different intervention to every element of a batch. Here a few of the script's
own settings are generated again on the plain Hugging Face model, with hooks
written independently of the engine and the same intervention in every batch
element:

    steering   a forward hook on block L adds alpha * v[L] to the block's output,
               alpha recomputed here from the vectors file
    removal    a forward hook on every block subtracts (h . u) u, u = v[L] / |v[L]|
    knockout   a forward pre-hook on each attention output projection zeroes the
               knocked-out heads' columns

The texts of steer_generate.generate_and_score must equal the independent ones.
The independent prompt of the prefix mode is the story prompt's ids followed by
the separately tokenized first 25 words of the transcript. The unsteered answer
must also begin like the knockout search's baseline. Both sides generate only
CHECK_TOKENS tokens, so that a CPU job finishes quickly: a greedy continuation's
first tokens do not depend on how many tokens follow.

Usage:
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-gen-check 32G fiveway/test_steer_generate.py --model pythia-1b
"""

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import steer_generate  # noqa: E402
from fiveway.engine import REPETITION_PENALTY, Engine  # noqa: E402
from fiveway.models import MODELS, attention_output_projection, blocks  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.prompts import task_prompts  # noqa: E402
from fiveway.steer_collect import LAYERNORM_FAMILIES  # noqa: E402

CHECK_TOKENS = 40   # tokens generated per answer on both sides
LAYER = 4           # the steering layer of the checked settings


class NoScorer:
    """Stands in for FiveWayScorer: this check compares texts only, so nothing needs scoring."""

    def probabilities(self, texts):
        return np.zeros((len(texts), 5))


def shared_prefix(a: str, b: str) -> int:
    """Number of leading characters two texts share."""
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


@torch.no_grad()
def independent_text(engine, prompt_ids: list, added=None, removed=None, heads=()) -> str:
    """The greedy rep1.25 continuation of prompt_ids under hooks written here.

    added: (layer, (d_model,) tensor) added after that block; removed: an
    (n_layers, d_model) tensor whose direction at each layer is projected out
    after that block; heads: (layer, head) pairs zeroed.
    """
    model_blocks = blocks(engine.model)
    d_head = engine.geometry["d_head"]
    handles = []

    def block_output(output):
        # NOTE: a Hugging Face block returns a tuple whose first
        # element is the residual stream after the block, or that tensor alone.
        return output[0] if isinstance(output, tuple) else output

    def replaced(output, hidden):
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    if added is not None:
        layer, vector = added

        def add_hook(module, args, output):
            # NOTE: (batch, positions, d_model) + (d_model,): the same
            # vector at every element and position.
            return replaced(output, block_output(output) + vector)
        handles.append(model_blocks[layer].register_forward_hook(add_hook))
    if removed is not None:
        units = removed / removed.norm(dim=-1, keepdim=True)    # NOTE: (n_layers, d_model)
        for layer, block in enumerate(model_blocks):
            def remove_hook(module, args, output, unit=units[layer]):
                hidden = block_output(output)
                # NOTE: hidden @ unit: (batch, positions), each position's
                # coordinate along the direction; times unit back to (batch, positions, d_model).
                return replaced(output, hidden - (hidden @ unit)[..., None] * unit)
            handles.append(block.register_forward_hook(remove_hook))
    for layer, head in heads:
        def zero_hook(module, args, head=head):
            # NOTE: args[0]: (batch, positions, n_heads * d_head), head h
            # in columns h * d_head to (h + 1) * d_head.
            concatenated = args[0].clone()
            concatenated[..., head * d_head:(head + 1) * d_head] = 0
            return (concatenated,) + tuple(args[1:])
        handles.append(attention_output_projection(model_blocks[layer]).register_forward_pre_hook(zero_hook))

    # NOTE: a whole batch of copies, as large as the engine's
    # batches, so the matrix kernels see the shapes they see in the engine and
    # the two greedy paths cannot part at a near-tie for that reason alone.
    input_ids = torch.tensor([prompt_ids] * engine.batch_size, device=engine.device)
    try:
        output = engine.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                       max_new_tokens=CHECK_TOKENS, do_sample=False,
                                       repetition_penalty=REPETITION_PENALTY[steer_generate.DECODING])
    finally:
        for handle in handles:
            handle.remove()
    return engine.decode(output[0, len(prompt_ids):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", default="cinderella")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    vectors_dir = RESULTS / "steering" / args.model
    task_vectors = steer_generate.load_vectors(vectors_dir / f"vectors_{args.task}.npz")
    resid_norm = task_vectors["resid_norm"]                     # NOTE: (n_layers,)
    vectors = {name: task_vectors[name] for name in ("A", "B", "C", "B_specific", "C_specific")}
    vectors.update(steer_generate.random_directions(
        task_vectors["A"], centre=MODELS[args.model].family in LAYERNORM_FAMILIES))
    search = json.loads((RESULTS / "search" / f"{args.model}_{args.task}_rep1.25_classifier.json").read_text())
    found = search["objectives"]["A"]["pruned"]["circuit"]
    circuit = {"heads": found["heads"], "mlp_blocks": found["mlp_blocks"], "value": found["value"]}
    assert not circuit["mlp_blocks"], "the independent knockout here zeroes heads only"

    # The script's own settings, built exactly as its main() builds them.
    grid = Namespace(strengths=[-1.0, 1.0], layers=[LAYER], layer=LAYER, prefix_strengths=[1.0],
                     rescue_strengths=[-1.0])
    transcripts = steer_generate.prefix_transcripts(args.task, 1)
    settings = {}
    for mode in ("sweep", "random", "rescue", "prefix"):
        for setting in steer_generate.build_settings(mode, grid, vectors, resid_norm, transcripts, circuit, "sandwich"):
            fields = setting.fields
            settings[(mode, fields.get("intervention"), fields["vector"], fields["strength"],
                      fields.get("group"))] = setting
    checked = {
        "unsteered": settings[("sweep", None, None, 0.0, None)],
        "A, r=+1": settings[("sweep", None, "A", 1.0, None)],
        "B, r=-1": settings[("sweep", None, "B", -1.0, None)],
        "random1, r=+1": settings[("random", None, "random1", 1.0, None)],
        "knockout, A removed": settings[("rescue", "knockout, A removed", None, None, None)],
        "knockout + A, r=-1": settings[("rescue", "knockout + steer", "A", -1.0, None)],
    }
    prefix_setting = settings[("prefix", None, "A", 1.0, "Control")]

    # The independent side's prompts and alphas, built here.
    engine = Engine(args.model, device)
    for name, vector in vectors.items():
        engine.vectors[name] = torch.tensor(vector, dtype=torch.float32, device=device)
    log(f"{args.model} on {device} ({torch.get_num_threads()} CPU threads); batches of {engine.batch_size}")
    _, story_prompt = task_prompts(engine.tokenizer, engine.spec.is_chat, args.task)
    story_ids = engine.encode(story_prompt)
    human_words = " ".join(transcripts[0]["text_response"].split()[:25])
    assert transcripts[0]["Group"] == "Control" and prefix_setting.prefix == human_words
    separator = "" if engine.spec.is_chat else " "
    prefix_ids = story_ids + engine.tokenizer(separator + human_words, add_special_tokens=False)["input_ids"]
    knockout_heads = [tuple(head) for head in circuit["heads"]]

    def alpha_of(name, strength):
        """alpha = r * resid_norm[L] / |v[L]|, from the vectors file."""
        return strength * float(resid_norm[LAYER]) / float(np.linalg.norm(vectors[name][LAYER].astype(np.float64)))

    def added(name, strength):
        """(LAYER, alpha * v[LAYER])."""
        return LAYER, alpha_of(name, strength) * torch.tensor(vectors[name][LAYER], dtype=torch.float32, device=device)

    for label, setting in list(checked.items()) + [("prefix, A, r=+1", prefix_setting)]:
        for layer, name, alpha in setting.intervention.steer:
            assert layer == LAYER and abs(alpha - alpha_of(name, setting.fields["strength"])) < 1e-9 * abs(alpha), label
    log("every checked setting steers at its layer with alpha = r * resid_norm / |v|")

    # The engine's side, through the script's own generation function.
    # NOTE: generate_and_score reads these two budgets from
    # its module when it runs, so the check shortens the script's answers to
    # the independent side's length and compares whole texts.
    steer_generate.STORY_TOKENS = steer_generate.CONTINUATION_TOKENS = CHECK_TOKENS
    records = steer_generate.generate_and_score(engine, NoScorer(), args.task,
                                                [setting.intervention for setting in checked.values()], None)
    engine_texts = {label: record["full_text"] for label, record in zip(checked, records)}
    engine_texts["prefix, A, r=+1"] = steer_generate.generate_and_score(
        engine, NoScorer(), args.task, [prefix_setting.intervention], prefix_setting.prefix)[0]["full_text"]
    log("engine texts generated")
    baseline = search["baseline"]["full_text"]
    # NOTE: the shortened answer's last characters can
    # decode differently from the same tokens inside the longer baseline.
    assert shared_prefix(engine_texts["unsteered"], baseline) >= len(engine_texts["unsteered"]) - 3, \
        "the unsteered answer does not begin like the knockout search's baseline"
    log("unsteered answer begins like the knockout search's baseline")

    independent = {
        "unsteered": independent_text(engine, story_ids),
        "A, r=+1": independent_text(engine, story_ids, added=added("A", 1.0)),
        "B, r=-1": independent_text(engine, story_ids, added=added("B", -1.0)),
        "random1, r=+1": independent_text(engine, story_ids, added=added("random1", 1.0)),
        "knockout, A removed": independent_text(engine, story_ids, removed=engine.vectors["A"], heads=knockout_heads),
        "knockout + A, r=-1": independent_text(engine, story_ids, added=added("A", -1.0), heads=knockout_heads),
        "prefix, A, r=+1": independent_text(engine, prefix_ids, added=added("A", 1.0)),
    }
    failures = 0
    for label, text in independent.items():
        same = engine_texts[label] == text
        failures += not same
        # NOTE: a prefix setting continues a human transcript, so its text stays out of the log.
        shown = "" if label.startswith("prefix") else f": {text[:70]!r}"
        log(f"  {label:20s} {'same' if same else 'DIFFERENT'} "
            f"({shared_prefix(engine_texts[label], text)} of {len(text)} characters shared){shown}")
    distinct = len(set(independent.values()))
    log(f"{distinct} distinct texts among {len(independent)} settings")
    assert failures == 0, f"{failures} settings differ"
    assert distinct >= 5, "the settings barely change the text, so the check would miss a wrong vector or layer"
    log("passed")


if __name__ == "__main__":
    main()
