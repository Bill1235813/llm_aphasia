"""Record each human transcript's mean residual stream, and build the steering vectors.

The steering half of the five-way study adds difference-of-means directions to a
generator's residual stream (activation addition, Rimsky et al., 2024), as the
earlier transformer_lens study did (src/steering/collect_activations.py). This
script is its activation side. It runs forward passes only, with no generation:

1. Every human transcript of a task (the classifier's AphasiaBank corpus,
   CLASSIFIER_DIR/data/<task>/transcripts.json) is fed to the model as if it
   were the model's own answer to the task's story prompt (prompts.py):

       Question: Tell me as much of the story of Cinderella as you can.
       Answer: <transcript>                                         GPT-2 XL, Pythia
       [INST] Tell me as much of the story ... [/INST] <transcript>  Llama-2-chat

   The residual stream after every block (Hugging Face's block output, the
   raw residual stream) is averaged over the transcript's tokens, not the
   prompt's. Prompt plus transcript are cut to 2048 tokens (1024 for GPT-2 XL,
   its whole context).
     runs/fiveway/steering/<model>/acts_<task>.npz   "resid" (n_transcripts, n_layers, d_model), float16
     runs/fiveway/steering/<model>/meta_<task>.json  per transcript: File_DB, Corpus, Group, aq,
                                                     n_tokens, duplicate_of, kept
   Both stay out of git, because the meta names AphasiaBank files.

2. The steering vectors of every layer, in float32, from the transcripts that
   count ("kept": the first copy of each text, see find_duplicates). For the
   LayerNorm models (GPT-2, Pythia) every mean is first centred over d_model,
   which gives transformer_lens's hook_resid_post of the earlier study
   (see centre_for_layernorm):

       A  = mean(Anomic, Conduction, Broca, Wernicke) - mean(Control)
       B  = mean(Broca) - mean(Control)
       C  = mean(Wernicke) - mean(Control)
       AC = mean(Anomic, Conduction) - mean(Control)
       B_specific, C_specific = B and C with their component along A removed

   The file also holds the group means (control_mean, aphasia_mean), resid_norm
   (the mean norm of a transcript's mean residual vector, the yardstick for
   relative steering strengths) and the group counts.
     results/fiveway/steering/<model>/vectors_<task>.npz

3. The logit lens of A, B and C: at every layer, the 15 tokens each vector
   promotes and suppresses most when read out through the unembedding.
     results/fiveway/steering/<model>/logit_lens_<task>.json

A rerun finds the activations of a finished task on disk. It skips that task's
forward passes and only rebuilds the vectors and the lens. A --limit smoke
test keeps all its outputs in runs/fiveway/steering/<model>-limit<N>/.

Usage:
    bash slurm/fiveway_submit.sh steer-collect-p1b 32G fiveway/steer_collect.py --model pythia-1b
    bash slurm/fiveway_submit.sh steer-collect-l13b 64G fiveway/steer_collect.py --model llama2-13b-chat
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh steer-test 32G fiveway/steer_collect.py --model pythia-1b --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine  # noqa: E402
from fiveway.models import MODELS, TASKS, blocks  # noqa: E402
from fiveway.paths import CLASSIFIER_DIR, RESULTS, RUNS  # noqa: E402
from fiveway.prompts import task_prompt_ids  # noqa: E402

MAX_TOKENS = 2048   # prompt plus transcript, as in the earlier study (src/steering/common.py)
GROUPS = ["Control", "Anomic", "Conduction", "Broca", "Wernicke"]

# The aphasic side of each steering vector; the other side is always Control.
VECTOR_GROUPS = {
    "A": ["Anomic", "Conduction", "Broca", "Wernicke"],
    "B": ["Broca"],
    "C": ["Wernicke"],
    "AC": ["Anomic", "Conduction"],
}

LENS_TOKENS = 15     # tokens listed per vector, layer and side
LENS_MIN_COUNT = 20  # the "_in_corpus" lists rank only tokens used this often in the task's transcripts

# The final normalization, an attribute of engine.model.base_model, whose weight the logit lens reads through.
FINAL_NORM = {"llama": "norm", "gpt2": "ln_f", "neox": "final_layer_norm", "qwen3_5": "norm"}
# NOTE: Qwen3.5's RMSNorm scales by 1 + weight (its
# weights start at 0), Llama's by weight; the lens must use the actual scale.
FINAL_NORM_OFFSET = {"qwen3_5": 1.0}

# The families whose every block reads the residual stream through a LayerNorm; Llama and Qwen3.5 use RMSNorm.
LAYERNORM_FAMILIES = ("gpt2", "neox")


def load_transcripts(task: str, limit) -> list:
    """The task's human transcripts in file order; with ``limit``, only the first ``limit`` of each group."""
    rows = json.loads((CLASSIFIER_DIR / "data" / task / "transcripts.json").read_text())
    if limit is None:
        return rows
    selected, taken = [], {group: 0 for group in GROUPS}
    for row in rows:
        if taken[row["Group"]] < limit:
            selected.append(row)
            taken[row["Group"]] += 1
    return selected


def aphasia_quotient(row: dict):
    """The Western Aphasia Battery aphasia quotient (0 to 100, lower is more severe), or None.

    NOTE: AphasiaBank's Custom_field holds the AQ of aphasic
    speakers and "." where it is missing. For Control speakers it holds other
    test scores or nothing, so their AQ is None.
    """
    if row["Group"] == "Control" or row["Custom_field"] == ".":
        return None
    return float(row["Custom_field"])


def find_duplicates(rows: list) -> tuple:
    """(duplicate_of, kept) per transcript: the index of its text's first copy, and whether statistics count it.

    duplicate_of is None for a first copy. A transcript is kept when it is the
    first copy of its text and every copy is listed under the same group.

    NOTE: the sandwich corpus lists 557 texts twice under two
    file names (<file>.gem.cex and <file>.gem0.cex, or <file>.gem.cex
    and PWA-<file>.gem.cex), always in the same group; counting both copies
    would weigh those speakers double and put copies on both sides of a probe's
    train/test split. Four Cinderella texts are listed once as Anomic and once
    as Control, in two related sub-corpora. Their group is unknown, so neither
    copy counts.
    """
    first_copy, groups_of_text = {}, {}
    for index, row in enumerate(rows):
        first_copy.setdefault(row["text_response"], index)
        groups_of_text.setdefault(row["text_response"], set()).add(row["Group"])
    duplicate_of, kept = [], []
    for index, row in enumerate(rows):
        first = first_copy[row["text_response"]]
        duplicate_of.append(None if first == index else first)
        kept.append(first == index and len(groups_of_text[row["text_response"]]) == 1)
    return duplicate_of, kept


def answer_token_ids(engine, text: str) -> list:
    """A transcript's token ids as they follow the story prompt, the model's answer.

    NOTE: Llama's SentencePiece tokenizer marks the first word of
    a text as word-initial by itself. The byte-level BPE of GPT-2 and Pythia
    needs the space written out, or the first word after "Answer:" becomes a
    word-internal piece the model would never have produced there.
    NOTE: the transcript is tokenized on its own and appended
    to the prompt's ids, so every transcript follows exactly the same prompt
    tokens and its span is known exactly.
    """
    separator = "" if engine.spec.is_chat else " "
    return engine.tokenizer(separator + text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def mean_residuals(engine, prompt_ids: list, transcript_ids: list, batch_size: int, log) -> np.ndarray:
    """Each transcript's residual stream after every block, averaged over its own tokens.

    Returns float32 (n_transcripts, n_layers, d_model).
    """
    n_layers, d_model = engine.geometry["n_layers"], engine.geometry["d_model"]
    means = np.zeros((len(transcript_ids), n_layers, d_model), dtype=np.float32)
    current = {}   # what the hooks need about the batch in flight: its transcript mask, token counts, output

    def reader(layer):
        def hook(module, args, output):
            # NOTE: a block's output is the residual stream after
            # the block: its input plus what its attention and MLP wrote.
            # Hugging Face's output_hidden_states reports the last block only
            # after the final normalization, so a hook is the direct way to read
            # every block's output as it is.
            hidden = output[0] if isinstance(output, tuple) else output
            # NOTE: hidden: (batch, positions, d_model) times the mask
            # (batch, positions, 1) zeroes the prompt and the padding; the sum
            # over positions, (batch, d_model), divided by the transcript's
            # token count (batch, 1), is each transcript's mean.
            current["means"][:, layer] = (hidden * current["mask"]).sum(dim=1) / current["counts"]
        return hook

    handles = [block.register_forward_hook(reader(layer)) for layer, block in enumerate(blocks(engine.model))]
    # NOTE: sorting by length puts transcripts of
    # similar length in one batch, so little of each forward pass goes to
    # padding. A fixed token budget per batch, instead of a fixed number of
    # transcripts, would also fill the GPU on the short sandwich transcripts.
    order = np.argsort([len(ids) for ids in transcript_ids], kind="stable")
    start, n_tokens_done = time.time(), 0
    for batch_start in range(0, len(order), batch_size):
        members = order[batch_start:batch_start + batch_size]
        sequences = [prompt_ids + transcript_ids[index] for index in members]
        width = max(len(sequence) for sequence in sequences)
        # NOTE: right padding: every sequence starts at position
        # 0 and its padding comes after its last token. A causal model's token
        # attends only to earlier tokens, so padding never reaches a real token,
        # and positions count from 0 as in a run of the sequence alone. The
        # attention mask is the standard way to tell the model which positions
        # are padding; with right padding it changes only the padding positions,
        # which the transcript mask leaves out of the means anyway.
        input_ids = torch.full((len(members), width), engine.eos_id)
        attention_mask = torch.zeros((len(members), width), dtype=torch.long)
        transcript_mask = torch.zeros((len(members), width, 1))
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence)
            attention_mask[row, :len(sequence)] = 1
            transcript_mask[row, len(prompt_ids):len(sequence)] = 1
        current["mask"] = transcript_mask.to(engine.device, engine.dtype)
        current["counts"] = current["mask"].sum(dim=1)   # NOTE: (batch, 1): tokens per transcript
        current["means"] = torch.zeros(len(members), n_layers, d_model, device=engine.device, dtype=engine.dtype)
        # NOTE: base_model is the transformer without
        # its unembedding: the hooks read everything we need, so no logits are computed.
        engine.model.base_model(input_ids=input_ids.to(engine.device),
                                attention_mask=attention_mask.to(engine.device), use_cache=False)
        means[members] = current["means"].float().cpu().numpy()
        n_tokens_done += sum(len(sequence) for sequence in sequences)
        batch_number = batch_start // batch_size + 1
        if batch_number % 10 == 0 or batch_start + batch_size >= len(order):
            seconds = time.time() - start
            log(f"  {batch_start + len(members)}/{len(order)} transcripts, {n_tokens_done} tokens, "
                f"{seconds:.0f} s ({n_tokens_done / seconds:.0f} tokens/s)")
    for handle in handles:
        handle.remove()
    return means


def centre_for_layernorm(resid: np.ndarray) -> np.ndarray:
    """Every residual vector minus its own mean over d_model: the part a LayerNorm can see.

    NOTE: LayerNorm starts by subtracting a vector's mean over its
    d_model coordinates, so the residual stream's component along the all-ones
    direction (1, 1, ..., 1) never reaches any later layer of GPT-2 or Pythia.
    In GPT-2 XL that invisible component is large: it carries more than half of
    the raw |A|^2 in the last layers, where it would inflate the cosines between
    vectors, take over PC1, and make directional ablation (Engine's remove)
    subtract the wrong amount along A. transformer_lens, which the earlier study
    used, removes it from every write to the residual stream
    (center_writing_weights), so centred means reproduce its hook_resid_post.
    Llama's RMSNorm does not subtract the mean, so Llama keeps the raw stream.
    NOTE: centring commutes with averaging over tokens, so the
    raw per-transcript means stored in acts_<task>.npz can be centred afterwards.
    NOTE: resid: (..., d_model); the mean over the last axis, kept as
    (..., 1), broadcasts back over d_model.
    """
    return resid - resid.mean(axis=-1, keepdims=True)


def build_vectors(resid: np.ndarray, groups: np.ndarray, kept: np.ndarray, centre: bool) -> dict:
    """Steering vectors, group means and resid_norm from the kept transcripts, in float32.

    resid: (n_transcripts, n_layers, d_model); groups and kept: (n_transcripts,).
    centre: whether the model reads its residual stream through LayerNorm (centre_for_layernorm).
    """
    # NOTE: the stored means are float16; summing hundreds of
    # them in float16 would lose the small differences the vectors are made of,
    # so they are averaged in float32.
    resid = resid[kept].astype(np.float32)             # NOTE: (n_kept, n_layers, d_model)
    if centre:
        resid = centre_for_layernorm(resid)
    groups = groups[kept]
    control_mean = resid[groups == "Control"].mean(axis=0)
    vectors = {"control_mean": control_mean, "aphasia_mean": resid[groups != "Control"].mean(axis=0)}
    for name, members in VECTOR_GROUPS.items():
        # NOTE: a pooled mean over the members' transcripts,
        # not a mean of group means, so A = aphasia_mean - control_mean and each
        # subtype weighs as much as it has transcripts.
        vectors[name] = resid[np.isin(groups, members)].mean(axis=0) - control_mean
    unit_a = vectors["A"] / np.linalg.norm(vectors["A"], axis=-1, keepdims=True)
    for name in ("B", "C"):
        # NOTE: (vector * unit_a).sum(-1, keepdims=True) is each
        # layer's coordinate along A, (n_layers, 1); times unit_a (n_layers,
        # d_model) it is the component along A, which is subtracted.
        along_a = (vectors[name] * unit_a).sum(axis=-1, keepdims=True)
        vectors[f"{name}_specific"] = vectors[name] - along_a * unit_a
    # NOTE: the norm over d_model gives (n_kept, n_layers); the mean over transcripts gives (n_layers,).
    vectors["resid_norm"] = np.linalg.norm(resid, axis=-1).mean(axis=0)
    vectors["group_names"] = np.array(GROUPS)
    vectors["group_counts"] = np.array([int((groups == group).sum()) for group in GROUPS])
    return vectors


@torch.no_grad()
def logit_lens(engine, vectors: dict, in_corpus: np.ndarray) -> dict:
    """For A, B and C at every layer, the tokens whose logits the vector raises and lowers most.

    NOTE: the logit lens (nostalgebraist, 2020) reads a direction
    as if it reached the output unchanged. Adding v to the final residual
    stream changes the logits through the final normalization and the
    unembedding. The normalization divides by a positive scale, which does not
    change which tokens rank highest, and multiplies by its learned weight,
    which does. LayerNorm (GPT-2, Pythia) also subtracts the mean over d_model;
    RMSNorm (Llama) does not. So the scores are the unembedding applied to the
    centred (LayerNorm only) and weighted vector. This is the ranking of the
    earlier logit_lens.py, whose transformer_lens unembedding had the final
    weight folded in. The lens is exact for the last layer's vector only: the
    layers after L still transform whatever is added at L.

    Tokens are decoded one at a time with the model's tokenizer: GPT-2 and
    Pythia keep a word's leading space (" yeah"), Llama's tokenizer drops it.
    The full-vocabulary lists of base models are often led by junk tokens
    (runs of spaces, code); the "_in_corpus" lists, which rank only tokens the
    transcripts use at least LENS_MIN_COUNT times, are the readable ones.
    """
    unembedding = engine.model.get_output_embeddings().weight      # NOTE: (vocabulary rows, d_model)
    final_norm = getattr(engine.model.base_model, FINAL_NORM[engine.spec.family])
    # NOTE: Pythia's unembedding has 50304 rows for 50277
    # tokens (padded for speed); the extra rows are no tokens, so only the
    # tokenizer's ids are ranked.
    n_tokens = len(engine.tokenizer)

    def decode(token_ids):
        return [engine.tokenizer.decode([int(token_id)]) for token_id in token_ids[:LENS_TOKENS]]

    report = {"top_k": LENS_TOKENS, "min_count": LENS_MIN_COUNT, "n_corpus_tokens": int(in_corpus.sum())}
    for name in ("A", "B", "C"):
        # NOTE: direction: (n_layers, d_model), one vector per layer.
        direction = torch.from_numpy(vectors[name]).to(engine.device, engine.dtype)
        if isinstance(final_norm, torch.nn.LayerNorm):
            # build_vectors already centred these vectors; centring again changes nothing,
            # and keeps the lens right for any vector it is given.
            direction = direction - direction.mean(dim=-1, keepdim=True)
        # NOTE: direction * scale (the norm's per-coordinate scale,
        # (d_model,)) scales each of the d_model coordinates, (n_layers,
        # d_model); @ unembedding.T (d_model, vocabulary rows) gives one score
        # per layer and token.
        scale = final_norm.weight + FINAL_NORM_OFFSET.get(engine.spec.family, 0.0)
        scores = ((direction * scale) @ unembedding.T)[:, :n_tokens].float().cpu().numpy()
        report[name] = []
        for layer_scores in scores:
            order = np.argsort(layer_scores, kind="stable")     # token ids from the lowest score to the highest
            corpus_order = order[in_corpus[order]]              # the same order, frequent transcript tokens only
            report[name].append({"promoted": decode(order[::-1]), "suppressed": decode(order),
                                 "promoted_in_corpus": decode(corpus_order[::-1]),
                                 "suppressed_in_corpus": decode(corpus_order)})
    return report


def write_json(path: Path, payload) -> None:
    """Write through a temporary file, so a job killed mid-write never leaves half a file behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--batch-size", type=int, default=8, help="transcripts per forward pass")
    parser.add_argument("--limit", type=int, help="smoke test: only the first LIMIT transcripts of each group")
    args = parser.parse_args()

    # NOTE: a --limit run is a smoke test. It keeps all its
    # outputs, the small ones too, in a folder of its own under runs/, so a
    # full run never resumes from it and results/ holds full runs only.
    # steer_analyze.py reads from the same folders.
    name = args.model if args.limit is None else f"{args.model}-limit{args.limit}"
    runs_dir = RUNS / "fiveway" / "steering" / name
    results_dir = RESULTS / "steering" / name if args.limit is None else runs_dir / "results"
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    engine = Engine(args.model, device)
    log(f"{args.model} on {device} ({torch.get_num_threads()} CPU threads): {engine.geometry}")
    # NOTE: GPT-2 XL reads at most 1024 positions, so its
    # sequences are cut there; about 60 Cinderella retellings are longer.
    max_tokens = min(MAX_TOKENS, engine.model.config.max_position_embeddings)

    for task in args.tasks:
        rows = load_transcripts(task, args.limit)
        prompt_ids = task_prompt_ids(engine, task)[1]
        room = max_tokens - len(prompt_ids)              # transcript tokens that fit after the prompt
        full_ids = [answer_token_ids(engine, row["text_response"]) for row in rows]
        transcript_ids = [ids[:room] for ids in full_ids]
        acts_path, meta_path = runs_dir / f"acts_{task}.npz", runs_dir / f"meta_{task}.json"

        if meta_path.exists():
            log(f"{task}: activations of {len(rows)} transcripts already in {runs_dir}; no forward passes")
        else:
            log(f"{task}: {len(rows)} transcripts after a {len(prompt_ids)}-token prompt, "
                f"{sum(len(ids) for ids in transcript_ids)} transcript tokens, "
                f"{sum(len(ids) > room for ids in full_ids)} cut at {max_tokens} tokens in all")
            resid = mean_residuals(engine, prompt_ids, transcript_ids, args.batch_size, log)
            np.savez(acts_path, resid=resid.astype(np.float16))
            duplicate_of, kept = find_duplicates(rows)
            meta = [{"File_DB": row["File_DB"], "Corpus": row["Corpus"], "Group": row["Group"],
                     "aq": aphasia_quotient(row), "n_tokens": len(ids), "duplicate_of": duplicate, "kept": keep}
                    for row, ids, duplicate, keep in zip(rows, transcript_ids, duplicate_of, kept)]
            # NOTE: the meta file is written last, so its
            # existence means the task's activations are complete; a job killed
            # before this line collects the task again.
            write_json(meta_path, meta)

        # NOTE: the vectors are always built from the stored
        # float16 means, right after collecting too, so a rerun builds exactly
        # the same vectors and steer_analyze.py sees exactly the numbers the
        # vectors were made from.
        with np.load(acts_path) as stored:
            resid = stored["resid"]                      # NOTE: (n_transcripts, n_layers, d_model), float16
        meta = json.loads(meta_path.read_text())
        # NOTE: activations collected for another
        # transcripts.json, prompt or tokenizer would no longer line up with
        # these transcripts (the corpus counts below pair them row by row);
        # delete the task's acts and meta files and collect again.
        assert [(row["File_DB"], row["n_tokens"]) for row in meta] == \
            [(row["File_DB"], len(ids)) for row, ids in zip(rows, transcript_ids)], f"{meta_path} is out of date"
        groups = np.array([row["Group"] for row in meta])
        kept = np.array([row["kept"] for row in meta])
        # NOTE: float16 holds values up to 65504; a larger
        # mean would have been stored as inf. Pythia-1B's largest mean is
        # about 140 and GPT-2 XL's about 610, but models with massive activations come closer.
        assert np.isfinite(resid).all(), "a mean residual overflowed float16"
        vectors = build_vectors(resid, groups, kept, centre=engine.spec.family in LAYERNORM_FAMILIES)
        np.savez(results_dir / f"vectors_{task}.npz", **vectors)

        corpus_ids = np.concatenate([ids for ids, keep in zip(transcript_ids, kept) if keep])
        in_corpus = np.bincount(corpus_ids, minlength=len(engine.tokenizer)) >= LENS_MIN_COUNT
        lens = logit_lens(engine, vectors, in_corpus)
        write_json(results_dir / f"logit_lens_{task}.json", lens)

        log(f"{task}: {int(kept.sum())} of {len(meta)} transcripts kept: "
            f"{dict(zip(GROUPS, vectors['group_counts'].tolist()))}; wrote {results_dir}")
        ratio = np.linalg.norm(vectors["A"], axis=-1) / vectors["resid_norm"]
        log("  |A| / resid_norm by layer: " + " ".join(f"{value:.3f}" for value in ratio))
        last = lens["A"][-1]
        log(f"  logit lens of A, last layer, transcript tokens: up {last['promoted_in_corpus'][:10]}, "
            f"down {last['suppressed_in_corpus'][:10]}")
    log("finished")


if __name__ == "__main__":
    main()
