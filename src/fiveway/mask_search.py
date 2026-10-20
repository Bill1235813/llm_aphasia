"""Learned knockout masks: how much to knock out each component, trained by gradient descent, then checked by generation.

For one generator and one task, for each objective (A = P(aphasia),
B = P(Broca), C = P(Wernicke)) and each space of components
(knockout_spaces.py: 1 MLP weight lesions, 2 attention heads, 3 both), every
component i gets a knockout amount k_i = sigmoid(theta_i) between 0 (intact)
and 1 (knocked out): a head's output is scaled by 1 - k_i, a lesion subtracts
k_i * x @ D_i (Engine.soft_knockout). The model stays frozen; only theta is
trained, on the task's human transcripts (the classifier's AphasiaBank corpus),
each read as the model's answer to the task's story prompt, as in
steer_collect.py:

    target transcripts T   the objective's groups (A: Anomic, Conduction, Broca,
                           Wernicke; B: Broca; C: Wernicke)
    contrast transcripts C every other group (A: Control; B and C: the other four)

    r(x) = (log p_k(x) - log p_0(x)) / tokens(x)   the masked model's per-token
           log-likelihood ratio to the unablated model, on the transcript's tokens
    loss = mean over T of softplus(-r / TAU) + mean over C of softplus(r / TAU)
           + lambda * sum_i k_i

so the masked model should make target speech more likely than the unablated
model does (r > 0 on T) and contrast speech less likely (r < 0 on C), with few
components knocked out.

NOTE: why the ratio to the unablated model rather than the
plain contrast log p_k(T) - log p_k(C): knocking out everything flattens the
model's predictions, and both groups' likelihoods then sink toward the same
value. The unablated model finds aphasic speech less likely than control
speech, so the plain contrast would rise toward 0 from destroying the model
alone. Measured against the unablated model, a destroyed model has r far below
0 on T and pays for it; a mask that makes all spoken narrative more likely has
r above 0 on C and pays for that.

Training follows a lambda path, like the lasso's: lambda_max is the largest
gain per unit amount any one component offers at the start, and stage j trains
STEPS_PER_STAGE steps at lambda_max / 2^j (j = 1 .. N_STAGES), warm-started
from stage j - 1's theta with a fresh Adam optimizer, so components join as
knocking one out costs less. The mask at the end of each stage is a checkpoint.

Each checkpoint is then checked the way the searches check knockouts: the
model answers the task under it (rep1.25 decoding, the Cinderella gate when the
unablated model passes it) and the five-way classifier scores the answer, in
two variants:

    soft    the learned amounts themselves (Intervention.soft, 3 decimals)
    binary  yes/no: knocked out where k_i > 0.5 (Intervention.knockout)

The reported result of a variant is its first (sparsest) checkpoint that works
(objective above 0.5), else its checkpoint with the highest value. A working
binary set is pruned as in search.py, after drop_weakest() has cheaply removed
the components with the lowest amounts that it does not need. Last, every
reported set answers prompts no search saw (knockout_spaces.held_out).

An experiment's time is its training time (kept across requeues stage by
stage) plus the generation and scoring time of every answer it used, stored
with each cached answer, whichever job made it.

1 speaker in 5 is held out of training; the validation losses on them are
reported, not used to choose anything. Transcripts never leave runs/: the
results hold masks, losses and the model's own answers only.

Output: results/fiveway/mask_search/<model>_<task>.json. Training state is
saved after every stage and every answer is cached, in
runs/fiveway/<model>/rep1.25_mask/, so a requeued job resumes.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=8:00:00 bash slurm/fiveway_submit.sh mask-p1b-cind 32G \\
        fiveway/mask_search.py --model pythia-1b --task cinderella
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.knockout_spaces import (SPACES, held_out_scores, mlp_matrix_names, same_baseline,  # noqa: E402
                                     space_components)
from fiveway.models import TASKS  # noqa: E402
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from fiveway.prompts import STORY_TOKENS, task_prompt_ids  # noqa: E402
from fiveway.scorer import CLASSES, FiveWayScorer  # noqa: E402
from fiveway.search import Evaluator, Objective, generation_seconds, knockout_of, prune, summary  # noqa: E402
from fiveway.steer_collect import answer_token_ids, find_duplicates, load_transcripts  # noqa: E402

DECODING = "rep1.25"
TAG = "rep1.25_mask"
GROUPS = ["Control", "Anomic", "Conduction", "Broca", "Wernicke"]
TARGET_GROUPS = {"A": ["Anomic", "Conduction", "Broca", "Wernicke"], "B": ["Broca"], "C": ["Wernicke"]}

THETA_INIT = -4.0       # k = sigmoid(-4) = 0.018: every component starts almost intact
THETA_LIMIT = 8.0       # theta is kept in [-8, 8] (k from 0.0003 to 0.9997), see train()
LEARNING_RATE = 0.2
N_STAGES = 6
STEPS_PER_STAGE = 50
PER_SIDE = 12           # target and contrast transcripts per step
TAU = 0.1               # nats per token: the scale of r at which the loss saturates
LAMBDA_BATCHES = 4      # batches averaged to find lambda_max
VALIDATION_PER_SIDE = 48
LOG_BATCH = 16          # transcripts per forward pass when no gradient is needed
MICRO_BATCH = 4         # transcripts per forward and backward pass (Llama-2-13B: about 5 GB of activations each)
BINARY_THRESHOLD = 0.5
DROP_WEAKEST_ABOVE = 24  # binary sets larger than this go through drop_weakest() before prune()


def speaker_of(file_name: str) -> str:
    """The AphasiaBank speaker of a transcript file: "site34a.cha_aphasia" -> "site34", "PWA-900-1.gem.cex" -> "900".

    NOTE: AphasiaBank names a file by site, speaker number and
    session: a trailing letter (site34a, site34b), a "-<n>" suffix (900-1,
    900-2) or a " <n>" suffix ("site15a 2") marks a session of the same
    speaker, whose transcripts must fall on the same side of the
    train/validation split.
    """
    stem = file_name.split(".")[0].removeprefix("PWA-").split(" ")[0]
    numbered = re.match(r"(\d+)-", stem)
    return numbered.group(1) if numbered else re.sub(r"[a-z]$", "", stem)


@contextmanager
def tf32():
    """TensorFloat-32 matrix products inside the block: training only, never generation.

    NOTE: TF32 tensor cores multiply float32
    matrices several times faster at about 3 significant digits, plenty for a
    gradient. Generation stays exact float32 (TF32 is off by default), so
    every checked answer is made as in the searches.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def transcript_log_likelihoods(engine, prompt_length: int, sequences: list) -> tuple:
    """Each sequence's summed log-probability of its tokens after the prompt, and their count: (batch,), (batch,)."""
    width = max(len(sequence) for sequence in sequences)
    # NOTE: right padding, as in steer_collect.py: a causal model's
    # real tokens never attend to the padding after them.
    input_ids = torch.full((len(sequences), width), engine.eos_id, device=engine.device)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long, device=engine.device)
    for row, sequence in enumerate(sequences):
        input_ids[row, :len(sequence)] = torch.tensor(sequence, device=engine.device)
        attention_mask[row, :len(sequence)] = 1
    # NOTE: the logit at position t predicts token t + 1, so the
    # transcript's tokens prompt_length .. width - 1 are predicted at positions
    # prompt_length - 1 .. width - 2. logits_to_keep keeps the last
    # width - prompt_length + 1 positions: (batch, width - prompt_length + 1, vocab).
    logits = engine.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                          logits_to_keep=width - prompt_length + 1).logits
    targets = input_ids[:, prompt_length:]   # NOTE: (batch, width - prompt_length)
    # NOTE: log_softmax over the vocabulary, then gather picks each
    # position's actual next token: (batch, width - prompt_length).
    log_probs = logits[:, :-1].log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
    counted = attention_mask[:, prompt_length:].to(log_probs.dtype)
    return (log_probs * counted).sum(dim=1), counted.sum(dim=1)


def amounts_of(k: torch.Tensor, components: list, geometry: dict) -> tuple:
    """(head amounts (n_layers, n_heads) or None, {lesion name: amount}) from the amount vector k."""
    n_lesions = sum(kind == "lesion" for kind, _ in components)
    lesion_amounts = {name: k[index] for index, (kind, name) in enumerate(components[:n_lesions])}
    if n_lesions == len(components):
        return None, lesion_amounts
    # NOTE: space_components() lists the heads after the lesions,
    # layer by layer and head by head, so they view as (n_layers, n_heads).
    return k[n_lesions:].view(geometry["n_layers"], geometry["n_heads"]), lesion_amounts


def backward_data_loss(engine, data: dict, components: list, amounts, target: list, contrast: list) -> tuple:
    """Backpropagate the data loss on these transcripts into the leaf behind ``amounts()``; returns (loss, r).

    NOTE: the loss is a sum of one term per
    transcript, so it is computed and backpropagated MICRO_BATCH transcripts
    at a time and the gradients add up to the whole batch's exactly, while
    only one micro-batch's activations are held in memory. Gradient
    checkpointing would allow larger micro-batches.
    """
    indices = target + contrast
    # NOTE: a target transcript's term is softplus(-r / TAU) and a
    # contrast transcript's softplus(r / TAU): sign +1 or -1 in softplus(-sign * r / TAU).
    # Each side is averaged over its own transcripts, hence the weights.
    signs = torch.tensor([1.0] * len(target) + [-1.0] * len(contrast), device=engine.device)
    weights = torch.tensor([1 / len(target)] * len(target) + [1 / len(contrast)] * len(contrast),
                           device=engine.device)
    total, r_values = 0.0, []
    for start in range(0, len(indices), MICRO_BATCH):
        chunk = indices[start:start + MICRO_BATCH]
        head_amounts, lesion_amounts = amounts_of(amounts(), components, engine.geometry)
        with engine.soft_knockout(head_amounts, lesion_amounts):
            summed, counts = transcript_log_likelihoods(engine, data["prompt_length"],
                                                        [data["sequences"][i] for i in chunk])
        r = (summed - data["clean"][chunk]) / counts
        part = slice(start, start + len(chunk))
        loss = (weights[part] * F.softplus(-signs[part] * r / TAU)).sum()
        loss.backward()
        total += float(loss)
        r_values.append(r.detach())
    return total, torch.cat(r_values)


@torch.no_grad()
def validation(engine, data: dict, components: list, k: torch.Tensor, target: list, contrast: list) -> dict:
    """The loss and mean r of the held-out speakers, in batches."""
    r_values = []
    for indices in (target, contrast):
        values = []
        for start in range(0, len(indices), LOG_BATCH):
            batch = indices[start:start + LOG_BATCH]
            head_amounts, lesion_amounts = amounts_of(k, components, engine.geometry)
            with engine.soft_knockout(head_amounts, lesion_amounts):
                summed, counts = transcript_log_likelihoods(engine, data["prompt_length"],
                                                            [data["sequences"][i] for i in batch])
            values.append((summed - data["clean"][batch]) / counts)
        r_values.append(torch.cat(values))
    loss = F.softplus(-r_values[0] / TAU).mean() + F.softplus(r_values[1] / TAU).mean()
    return {"loss": round(float(loss), 6), "mean_r_target": round(float(r_values[0].mean()), 6),
            "mean_r_contrast": round(float(r_values[1].mean()), 6)}


def train(engine, data: dict, components: list, objective: str, state_path: Path, seed: int, log) -> dict:
    """The lambda path for one objective and space; returns lambda_max and one checkpoint per stage."""
    split = data["split"][objective]
    generator = torch.Generator().manual_seed(seed)

    def sample_batch():
        # NOTE: equal numbers of target and contrast transcripts
        # per step, drawn with replacement, whatever the groups' sizes (Wernicke
        # has 58 Cinderella transcripts, Control 399).
        target = split["train_target"][torch.randint(len(split["train_target"]), (PER_SIDE,), generator=generator)]
        contrast = split["train_contrast"][torch.randint(len(split["train_contrast"]), (PER_SIDE,),
                                                         generator=generator)]
        return target.tolist(), contrast.tolist()

    # NOTE: N_STAGES is left out: a longer path may extend a finished one.
    config = {"theta_init": THETA_INIT, "theta_limit": THETA_LIMIT, "learning_rate": LEARNING_RATE,
              "steps_per_stage": STEPS_PER_STAGE, "per_side": PER_SIDE, "tau": TAU,
              "lambda_batches": LAMBDA_BATCHES, "micro_batch": MICRO_BATCH, "n_components": len(components)}
    theta = torch.full((len(components),), THETA_INIT, device=engine.device, requires_grad=True)
    saved = {"config": config, "lambda_max": None, "checkpoints": [], "seconds": 0.0}
    if state_path.exists():
        saved = torch.load(state_path, map_location=engine.device)
        # NOTE: a state trained with other settings must not
        # be resumed silently; delete it to start over.
        assert saved["config"] == config, (state_path, saved["config"], config)
        with torch.no_grad():
            theta.copy_(saved["theta"])
        generator.set_state(saved["generator"].cpu())
        log(f"    resumed after stage {len(saved['checkpoints'])}")
    start = time.time() - saved["seconds"]

    with tf32():
        if saved["lambda_max"] is None:
            # lambda_max: the largest average decrease of the data loss per unit
            # amount that any single component offers at the start.
            k = torch.sigmoid(theta.detach()).requires_grad_()
            for _ in range(LAMBDA_BATCHES):
                backward_data_loss(engine, data, components, lambda: k, *sample_batch())
            gradient = k.grad / LAMBDA_BATCHES
            # NOTE: if no component lowers the loss at the
            # start, lambda_max falls back to a small fraction of the largest
            # gradient, and the path still runs.
            saved["lambda_max"] = max(float((-gradient).max()), float(gradient.abs().max()) * 1e-3)
            log(f"    lambda_max {saved['lambda_max']:.3g}; components that lower the loss at the start: "
                f"{int((gradient < 0).sum())} of {len(components)}")

        for stage in range(len(saved["checkpoints"]) + 1, N_STAGES + 1):
            penalty = saved["lambda_max"] / 2 ** stage
            # NOTE: a fresh optimizer per stage. Adam divides
            # each step by the root mean square of all earlier gradients; kept
            # across stages, that memory is dominated by stage 1, where the
            # gradients through sigmoid'(theta) are large, so a component pushed
            # down to the clamp would rejoin at a fraction of the learning rate
            # when lambda halves, stages late or never. Fresh, every theta moves
            # about LEARNING_RATE per step in the direction its gradient says.
            optimizer = torch.optim.Adam([theta], lr=LEARNING_RATE)
            losses, r_target, r_contrast = [], [], []
            for _ in range(STEPS_PER_STAGE):
                target, contrast = sample_batch()
                optimizer.zero_grad()
                loss, r = backward_data_loss(engine, data, components, lambda: torch.sigmoid(theta), target, contrast)
                (penalty * torch.sigmoid(theta).sum()).backward()
                optimizer.step()
                # NOTE: the clamp keeps sigmoid'(theta) above
                # 3e-4, so a component's gradient stays far above Adam's epsilon
                # (1e-8) and it can rejoin when lambda falls.
                with torch.no_grad():
                    theta.clamp_(-THETA_LIMIT, THETA_LIMIT)
                losses.append(loss)
                r_target.append(float(r[:len(target)].mean()))
                r_contrast.append(float(r[len(target):].mean()))
            k = torch.sigmoid(theta.detach())
            checkpoint = {"stage": stage, "lambda": penalty, "amounts": k.cpu().numpy().round(4).tolist(),
                          "n_above_half": int((k > BINARY_THRESHOLD).sum()), "sum_k": round(float(k.sum()), 4),
                          "train_loss": round(float(np.mean(losses[-10:])), 6),
                          "train_mean_r_target": round(float(np.mean(r_target[-10:])), 6),
                          "train_mean_r_contrast": round(float(np.mean(r_contrast[-10:])), 6),
                          "validation": validation(engine, data, components, k, split["validation_target"],
                                                   split["validation_contrast"])}
            saved["checkpoints"].append(checkpoint)
            saved["seconds"] = time.time() - start
            log(f"    stage {stage}: lambda {penalty:.3g}, {checkpoint['n_above_half']} above 0.5, sum k "
                f"{checkpoint['sum_k']:.2f}, train loss {checkpoint['train_loss']:.4f} (r_T "
                f"{checkpoint['train_mean_r_target']:+.4f}, r_C {checkpoint['train_mean_r_contrast']:+.4f}), "
                f"validation {checkpoint['validation']}")
            # NOTE: written to a temporary file and renamed, so
            # a job preempted mid-write never leaves a cut-off state behind.
            temporary = state_path.with_suffix(".tmp")
            torch.save({**saved, "theta": theta.detach(), "generator": generator.get_state()}, temporary)
            temporary.replace(state_path)
    return {"lambda_max": saved["lambda_max"], "checkpoints": saved["checkpoints"], "seconds": saved["seconds"]}


def soft_intervention(components: list, amounts: list) -> Intervention:
    return Intervention.soft({c: a for (kind, c), a in zip(components, amounts) if kind == "head"},
                             {c: a for (kind, c), a in zip(components, amounts) if kind == "lesion"})


def binary_intervention(components: list, amounts: list) -> Intervention:
    return knockout_of([component for component, a in zip(components, amounts) if a > BINARY_THRESHOLD])


def drop_weakest(evaluator, objective, components: list, amounts: list, log) -> list:
    """Cheaply shrink a large working binary set before prune(): drop its lowest-amount components in halving chunks.

    NOTE: prune() tries every leave-one-out set in
    each round, about n^2 / 2 answers for a set of n. Dropping the weakest
    half, then quarter, ... of the components (by learned amount) while the
    set still works costs about 2 log2(n) answers and leaves prune() a
    smaller set; a component the set needs is kept, whatever its amount.
    """
    order = sorted(range(len(components)), key=lambda i: amounts[i])   # weakest first
    kept = [components[i] for i in order]
    chunk = len(kept) // 2
    while chunk >= 1 and len(kept) > 1:
        candidate = kept[chunk:]
        if objective.works(objective.value(evaluator.records([knockout_of(candidate)])[0])):
            kept = candidate
            chunk = min(chunk, len(kept) // 2)
        else:
            chunk //= 2
    log(f"    drop_weakest: {len(components)} -> {len(kept)} components")
    return kept


def choose(values: list, objective) -> int:
    """The first checkpoint that works, else the one with the highest value (the last one if none counts)."""
    for index, value in enumerate(values):
        if objective.works(value):
            return index
    scored = [(value, index) for index, value in enumerate(values) if value is not None]
    return max(scored)[1] if scored else len(values) - 1


def prepare_data(engine, task: str, log) -> dict:
    """Tokenized transcripts, their unablated log-likelihoods, and each objective's train/validation split."""
    rows = load_transcripts(task, None)
    _, kept = find_duplicates(rows)
    _, story_ids = task_prompt_ids(engine, task)
    sequences, groups, validation_side = [], [], []
    for row, keep in zip(rows, kept):
        transcript_ids = answer_token_ids(engine, row["text_response"])[:STORY_TOKENS]
        if not keep or not transcript_ids:
            continue
        # NOTE: the transcript is cut to the model's own answer
        # length (STORY_TOKENS), the stretch of text the classifier judges.
        sequences.append(story_ids + transcript_ids)
        groups.append(row["Group"])
        validation_side.append(zlib.crc32(speaker_of(row["File_DB"]).encode()) % 5 == 0)
    groups, validation_side = np.array(groups), np.array(validation_side)

    clean = torch.zeros(len(sequences), device=engine.device)
    order = np.argsort([len(sequence) for sequence in sequences], kind="stable")
    with torch.no_grad(), tf32():
        for start in range(0, len(order), LOG_BATCH):
            batch = order[start:start + LOG_BATCH].tolist()
            summed, _ = transcript_log_likelihoods(engine, len(story_ids), [sequences[i] for i in batch])
            clean[batch] = summed

    split = {}
    for objective, target_groups in TARGET_GROUPS.items():
        is_target = np.isin(groups, target_groups)
        part = {"train_target": np.flatnonzero(is_target & ~validation_side),
                "train_contrast": np.flatnonzero(~is_target & ~validation_side),
                # NOTE: a fixed, evenly spaced subset of the
                # held-out speakers keeps every stage's validation cheap and comparable.
                "validation_target": np.flatnonzero(is_target & validation_side),
                "validation_contrast": np.flatnonzero(~is_target & validation_side)}
        for name in ("validation_target", "validation_contrast"):
            indices = part[name]
            part[name] = indices[np.linspace(0, len(indices) - 1, min(VALIDATION_PER_SIDE, len(indices))).astype(int)]
        # Training indices are sampled from as tensors; validation ones are read in order, as lists.
        split[objective] = {name: torch.tensor(indices) if name.startswith("train") else indices.tolist()
                            for name, indices in part.items()}
    counts = {objective: {name: len(value) for name, value in part.items()} for objective, part in split.items()}
    log(f"{len(sequences)} transcripts ({dict(zip(*np.unique(groups, return_counts=True)))}), "
        f"{int(validation_side.sum())} from validation speakers; per objective {counts}")
    return {"sequences": sequences, "prompt_length": len(story_ids), "clean": clean, "split": split,
            "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--objectives", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--spaces", nargs="+", default=list(SPACES))
    parser.add_argument("--smoke", action="store_true", help="2 stages of 3 steps, results under a smoke name")
    args = parser.parse_args()
    global N_STAGES, STEPS_PER_STAGE
    if args.smoke:
        N_STAGES, STEPS_PER_STAGE = 2, 3
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
    engine = Engine(args.model, device)
    for parameter in engine.model.parameters():
        parameter.requires_grad_(False)
    scorer = FiveWayScorer(device)
    load_lesions(engine, mlp_matrix_names(engine))
    run_dir = RUNS / "fiveway" / args.model / (TAG + ("_smoke" if args.smoke else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(engine, scorer, args.task, DECODING, run_dir / f"cache_{args.task}.jsonl")
    # NOTE: a smoke run keeps its results under runs/, so
    # results/ (in git) only ever holds full runs.
    out_path = (run_dir / f"results_{args.task}.json" if args.smoke
                else RESULTS / "mask_search" / f"{args.model}_{args.task}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {
        "config": {**vars(args), "theta_init": THETA_INIT, "learning_rate": LEARNING_RATE, "n_stages": N_STAGES,
                   "steps_per_stage": STEPS_PER_STAGE, "per_side": PER_SIDE, "tau": TAU},
        "geometry": engine.geometry, "objectives": {}}
    log(f"{args.model} on {device_name}, task {args.task}: cache holds {len(evaluator.cache)} answers")

    baseline = evaluator.records([Intervention()])[0]
    evaluator.gate = baseline["answer"] is not False
    results["gate"] = evaluator.gate
    results["baseline"] = {"answer": baseline["answer"], "story": baseline["story"],
                           "valid": evaluator.valid(baseline), "probabilities": baseline["probabilities"],
                           "same_as_greedy_searches": same_baseline(args.model, args.task, baseline["story"])}
    log(f"unablated answer the same as the greedy searches': {results['baseline']['same_as_greedy_searches']}")
    data = prepare_data(engine, args.task, log)
    results["data"] = data["counts"]

    def save():
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    for name in args.objectives:
        objective = Objective(name, "classifier", evaluator)
        for space in args.spaces:
            if results["objectives"].get(name, {}).get(space, {}).get("done"):
                log(f"objective {name}, {space}: already done")
                continue
            components = space_components(engine, space)
            log(f"objective {name}, {space}: {len(components)} components")
            seed = zlib.crc32(f"{args.model} {args.task} {name} {space}".encode())
            trained = train(engine, data, components, name, run_dir / f"train_{args.task}_{name}_{space}.pt",
                            seed, log)

            evaluator.used = set()
            checkpoints = trained["checkpoints"]
            soft = [soft_intervention(components, c["amounts"]) for c in checkpoints]
            binary = [binary_intervention(components, c["amounts"]) for c in checkpoints]
            # One call, so both variants share generation batches.
            together = evaluator.records(soft + binary)
            records = {"soft": together[:len(soft)], "binary": together[len(soft):]}
            entry = {"n_components": len(components), "lambda_max": trained["lambda_max"], "checkpoints": []}
            for index, checkpoint in enumerate(checkpoints):
                row = {key: value for key, value in checkpoint.items() if key != "amounts"}
                for variant in ("soft", "binary"):
                    record = records[variant][index]
                    value = objective.value(record)
                    row[variant] = {"value": value, "works": objective.works(value),
                                    "predicted_class": None if record["probabilities"] is None
                                    else CLASSES[int(np.argmax(record["probabilities"]))],
                                    "n_knocked_out": len(record["heads"]) + len(record.get("lesions", []))
                                    + len(record.get("head_amounts", [])) + len(record.get("lesion_amounts", []))}
                entry["checkpoints"].append(row)
                log(f"    checkpoint {index + 1}: soft {row['soft']['value']} ({row['soft']['n_knocked_out']} "
                    f"with amount > 0), binary {row['binary']['value']} ({row['binary']['n_knocked_out']} components)")
            evaluate_keys = set(evaluator.used)

            evaluator.used = set()
            for variant in ("soft", "binary"):
                values = [objective.value(record) for record in records[variant]]
                chosen = choose(values, objective)
                entry[variant] = {"checkpoint": chosen + 1, "works": objective.works(values[chosen]),
                                  "amounts": checkpoints[chosen]["amounts"],
                                  "result": summary(records[variant][chosen], objective)}
                if variant == "binary" and entry[variant]["works"]:
                    found = [c for c, a in zip(components, checkpoints[chosen]["amounts"]) if a > BINARY_THRESHOLD]
                    if len(found) > DROP_WEAKEST_ABOVE:
                        weights = [a for a in checkpoints[chosen]["amounts"] if a > BINARY_THRESHOLD]
                        found = drop_weakest(evaluator, objective, found, weights, log)
                    start_set = summary(evaluator.records([knockout_of(found)])[0], objective)
                    entry[variant]["pruned"] = prune(evaluator, objective, start_set, log)
            prune_keys = set(evaluator.used) - evaluate_keys
            evaluator.used = None
            entry["n_evaluated_sets"] = len(evaluate_keys)
            entry["n_prune_sets"] = len(prune_keys)
            # The cost of the experiment: training, plus the generation and scoring time of every
            # answer it used, whichever job made it.
            entry["seconds"] = {"train": round(trained["seconds"], 1),
                                "evaluate": generation_seconds(evaluator, evaluate_keys),
                                "prune": generation_seconds(evaluator, prune_keys)}
            entry["device"] = device_name
            entry["done"] = True
            results["objectives"].setdefault(name, {})[space] = entry
            save()
            log(f"  objective {name}, {space} done: soft works {entry['soft']['works']} (checkpoint "
                f"{entry['soft']['checkpoint']}), binary works {entry['binary']['works']} (checkpoint "
                f"{entry['binary']['checkpoint']}); seconds {entry['seconds']}")

    # Held out: every reported set (soft mask; binary set, pruned if it works) on the other task and the TAB prompts.
    reported = {}
    for name, spaces in results["objectives"].items():
        for space, entry in spaces.items():
            components = space_components(engine, space)
            binary = entry["binary"].get("pruned", {}).get("circuit", entry["binary"]["result"])
            for variant, intervention in (("soft", soft_intervention(components, entry["soft"]["amounts"])),
                                          ("binary", knockout_of([("head", tuple(c)) for c in binary["heads"]]
                                                                 + [("lesion", c) for c in binary.get("lesions", [])]))):
                entry[variant]["reported_key"] = intervention.key()
                reported[intervention.key()] = intervention
    results["held_out"] = held_out_scores(engine, scorer, args.task, run_dir, results.get("held_out", {}),
                                          reported, log)
    save()
    log(f"finished; {evaluator.n_generated} new answers generated; wrote {out_path}")


if __name__ == "__main__":
    main()
