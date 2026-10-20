"""The emergent-modularity baseline: zero each submodule's most important weights, as Wang et al. (2025) do.

The study asks how the knockout search (search.py)
compares with the parameter lesioning of Wang, Fan, Han, Bi and Li (2025),
"Emergent modularity in large language models: Insights from aphasia
simulations" (bioRxiv 10.1101/2025.02.22.639416; Methods, "Lesioning
individual parameters"). Their method, run here on any of the seven
generators and on both tasks:

  1. Importance. Every weight w of the attention and MLP weight matrices gets
     the first-order Taylor estimate of how much the loss changes when w is
     set to zero, I(w) = |w * dL/dw|. L is the paper's corpus loss, the summed
     negative log-likelihood of the Control speakers' narratives: here the
     Control transcripts of both tasks, each fed as the model's answer to its
     task's story prompt, with the loss on the answer tokens only (float32,
     one transcript per backward pass, gradients summed). Weights that read
     an input channel with massive activations (Sun et al., 2024) are left out.
  2. Selection. Within each submodule, the top 1% of its weights by importance.
  3. Lesions. One submodule at a time, its top 1% is set to zero, the model
     answers the task as search.py makes it answer (rep1.25 decoding; for
     Cinderella the yes/no question first; a base model's answer cut at the
     first question it writes itself), and the weights are restored exactly.
  4. Subtypes. Each answer is compared with each human group's transcripts of
     the task (Control, Anomic, Broca, Conduction, Wernicke) by BLEU-1 and by
     BERTScore F1. Each metric is standardized per group across the lesions'
     answers, the two standardized metrics are averaged, and the submodule is
     assigned the group where that average is highest. The five-way
     classifier also scores every answer.
  5. Clusters. For each aphasia subtype, the submodules assigned to it are
     lesioned together and the answer is assessed again. While it is not
     assigned the subtype, the next best submodule for the subtype is added
     (at most 32 in all). The paper reports clusters of 16 (Broca), 5
     (Wernicke), 4 (Conduction), 15 (Anomic) and 3 (Global) submodules.
  6. Comparison with the knockout search: the share of each cluster's lesioned
     weights that lie in the circuit the search found, and what the five-way
     classifier says of every answer.

The submodules are the paper's "4 attention blocks and 3 feedforward blocks" per layer:

    family    submodules of one layer                                                       per layer
    Llama-2   q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj                     7
    GPT-2     the q, k and v thirds of attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj           6
    Pythia    the q, k and v rows of attention.query_key_value, attention.dense,
              mlp.dense_h_to_4h, mlp.dense_4h_to_h                                              6

and are named L<layer>.<kind>, with kind q, k, v, o, gate, up or down. GPT-2
and Pythia have no gate matrix. Embeddings, norms, the unembedding and all
biases are not scored.

Outputs:
    runs/fiveway/em_baseline/<model>/top1pct.pt            each submodule's top 1%: positions in its module's
                                                           flattened weight, most important first; a summary
    runs/fiveway/em_baseline/<model>/generations_<task>.jsonl, scores_<task>.jsonl   caches, for resuming
    results/fiveway/em_baseline/<model>/lesions.json       the unlesioned model and every single-submodule
                                                           lesion: gate, answer, BLEU-1 and BERTScore per
                                                           group, standardized scores, assigned subtype,
                                                           five-way probabilities and verdict
    results/fiveway/em_baseline/<model>/clusters.json      every cluster's steps (the same fields) and the
                                                           comparison with the search's circuits
A run with a --max-... test option keeps all of its files in
runs/fiveway/em_baseline/<model>-<options>/. A rerun resumes from these files.
--compare-only redoes step 6 alone, for search results that came in later.

NOTE: run a model's two tasks in one job (the default):
they share top1pct.pt, lesions.json and clusters.json, and two jobs of one
model running at once would write the same files over each other.

Usage:
    bash slurm/fiveway_submit.sh em-p1b 48G fiveway/em_baseline.py --model pythia-1b
    GRES=gpu:1 bash slurm/fiveway_submit.sh em-l13b 96G fiveway/em_baseline.py --model llama2-13b-chat
    GPUS=0 bash slurm/fiveway_submit.sh em-compare-p1b 16G fiveway/em_baseline.py --model pythia-1b --compare-only
  CPU smoke test:
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh em-test 32G fiveway/em_baseline.py --model pythia-1b \\
        --tasks sandwich --max-transcripts 3 --max-submodules 4 --max-references 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import AutoConfig, AutoTokenizer
from transformers.pytorch_utils import Conv1D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import MLP_BLOCK_SIZE, MODELS, TASKS, blocks, geometry  # noqa: E402
from fiveway.paths import CLASSIFIER_DIR, RESULTS, RUNS  # noqa: E402
from fiveway.prompts import (STORY_TOKENS, YESNO_TOKENS, cut_at_new_question, long_enough,  # noqa: E402
                             parse_yes_no, task_prompt_ids)
from fiveway.scorer import CLASSES, OBJECTIVES, SUCCESS_THRESHOLD, FiveWayScorer, objective_values  # noqa: E402

DECODING = "rep1.25"
SEARCH_TAG = "rep1.25_classifier"   # the search results whose circuits step 6 compares with
TOP_FRACTION = 0.01                 # the paper's "top 1% of parameters" of each submodule
MASSIVE_RATIO = 100.0               # an input channel is massive above this multiple of the median channel
MAX_TOKENS = 1024                   # prompt plus transcript in an importance pass (GPT-2 XL's whole context)
MAX_CLUSTER_SIZE = 32
GROUPS = ["Control", "Anomic", "Broca", "Conduction", "Wernicke"]   # AphasiaBank's groups; it has no Global group
APHASIA_GROUPS = GROUPS[1:]


# --- the human transcripts ----------------------------------------------------

def human_transcripts(task: str) -> dict:
    """The task's AphasiaBank transcripts by group, each text once: {group: [texts]}, in file order.

    NOTE: the sandwich corpus lists about 557 texts twice
    under two file names, and four Cinderella texts are listed once as Control
    and once as Anomic. As everywhere in the study (steer_collect.py), the
    first copy of a text counts, and a text listed under two groups counts for
    neither, since its group is unknown.
    """
    rows = json.loads((CLASSIFIER_DIR / "data" / task / "transcripts.json").read_text())
    groups_of_text = {}
    for row in rows:
        groups_of_text.setdefault(row["text_response"], set()).add(row["Group"])
    transcripts = {group: [] for group in GROUPS}
    seen = set()
    for row in rows:
        text = row["text_response"]
        if text not in seen and len(groups_of_text[text]) == 1:
            transcripts[row["Group"]].append(text)
        seen.add(text)
    return transcripts


# --- the submodules -------------------------------------------------------------

@dataclass
class Submodule:
    """One of the paper's submodules: one weight matrix of one layer."""
    name: str                  # "L3.q": layer 3's query projection
    layer: int
    kind: str                  # q, k, v, o, gate, up or down
    module: torch.nn.Module    # the nn.Linear (a Conv1D in GPT-2) whose weight holds it
    rows: list | None          # its outputs among the module's outputs; None when it is the whole module


def find_submodules(model) -> list:
    """Every layer's attention and MLP weight matrices, layer by layer."""
    family = model.config.model_type
    shape = geometry(model.config)
    d_model, n_heads, d_head = shape["d_model"], shape["n_heads"], shape["d_head"]
    submodules = []
    for layer, block in enumerate(blocks(model)):
        if family == "llama":
            attention, mlp = block.self_attn, block.mlp
            parts = [("q", attention.q_proj, None), ("k", attention.k_proj, None), ("v", attention.v_proj, None),
                     ("o", attention.o_proj, None), ("gate", mlp.gate_proj, None), ("up", mlp.up_proj, None),
                     ("down", mlp.down_proj, None)]
        elif family == "gpt2":
            # NOTE: GPT-2 computes q, k and v with one matrix,
            # c_attn, and splits its 3 * d_model outputs into thirds: q, k, v.
            thirds = [list(range(part * d_model, (part + 1) * d_model)) for part in range(3)]
            fused = block.attn.c_attn
            parts = [("q", fused, thirds[0]), ("k", fused, thirds[1]), ("v", fused, thirds[2]),
                     ("o", block.attn.c_proj, None), ("up", block.mlp.c_fc, None), ("down", block.mlp.c_proj, None)]
        else:
            # NOTE: GPT-NeoX (Pythia) also fuses q, k and v into one
            # matrix, query_key_value, but orders its outputs head by head: it
            # views them as (n_heads, 3 * d_head) and cuts the last axis in three,
            # so head h's q, k and v are the d_head outputs starting at
            # h*3*d_head, h*3*d_head + d_head and h*3*d_head + 2*d_head.
            interleaved = [[head * 3 * d_head + part * d_head + index
                            for head in range(n_heads) for index in range(d_head)] for part in range(3)]
            fused = block.attention.query_key_value
            parts = [("q", fused, interleaved[0]), ("k", fused, interleaved[1]), ("v", fused, interleaved[2]),
                     ("o", block.attention.dense, None), ("up", block.mlp.dense_h_to_4h, None),
                     ("down", block.mlp.dense_4h_to_h, None)]
        for kind, module, rows in parts:
            submodules.append(Submodule(f"L{layer}.{kind}", layer, kind, module, rows))
    return submodules


def weight_positions(submodule: Submodule) -> torch.Tensor:
    """Where each of the submodule's weights sits in its module's flattened weight: (outputs, inputs).

    NOTE: torch.arange(n).view(shape) numbers the entries of a
    tensor by their position in memory. Transposing this numbering and taking
    rows of it, as one would of the weight itself, says for each weight of the
    submodule where it lives, so weight.view(-1)[positions] reads or writes
    exactly the submodule's weights, in (outputs, inputs) layout.
    """
    weight = submodule.module.weight
    positions = torch.arange(weight.numel(), device=weight.device).view(weight.shape)
    if isinstance(submodule.module, Conv1D):
        # NOTE: GPT-2's Conv1D stores its weight as (inputs, outputs),
        # the transpose of nn.Linear's (outputs, inputs).
        positions = positions.T
    if submodule.rows is not None:
        positions = positions[submodule.rows]
    return positions


# --- steps 1 and 2: importance and the top 1% -----------------------------------

def answer_token_ids(engine, text: str) -> list:
    """A transcript's token ids as they follow the story prompt, the model's answer.

    NOTE: Llama's SentencePiece tokenizer marks the first word of
    a text as word-initial by itself; the byte-level BPE of GPT-2 and Pythia
    needs the space after "Answer:" written out (as in steer_collect.py).
    """
    separator = "" if engine.spec.is_chat else " "
    return engine.tokenizer(separator + text, add_special_tokens=False)["input_ids"]


def select_top_weights(importance: torch.Tensor, positions: torch.Tensor, channel_mean: torch.Tensor) -> tuple:
    """A submodule's top 1% weights: (their positions, most important first; a few numbers about the selection).

    importance and positions: (outputs, inputs); channel_mean: (inputs,), the
    mean |activation| of each input channel over the corpus.
    """
    # NOTE: Sun et al. (2024) found a few hidden channels whose
    # activations are thousands of times larger than the rest, at a few tokens
    # such as the first one. They act as fixed biases rather than carrying
    # content, and a weight that reads such a channel gets a large |w * dL/dw|
    # for that reason alone. Following the paper, those weights are left out.
    median = channel_mean.median()
    massive = channel_mean > MASSIVE_RATIO * median
    # NOTE: massive (inputs,) picks whole columns of importance
    # (outputs, inputs); importance is never negative, so -1 is never in the top 1%.
    importance[:, massive] = -1.0
    n_selected = int(TOP_FRACTION * importance.numel())
    values, order = torch.topk(importance.flatten(), n_selected)
    details = {"n_massive_channels": int(massive.sum()), "max_channel_ratio": float(channel_mean.max() / median),
               "importance_threshold": float(values[-1])}
    return positions.flatten()[order], details


def importance_top_weights(model, submodules: list, sequences: list, log) -> tuple:
    """Steps 1 and 2: each submodule's top 1% weights by first-order importance.

    sequences: (prompt ids, answer ids) pairs, the Control transcripts as answers.
    Returns ({submodule name: int32 positions of its top 1% in its module's
    flattened weight, most important first}, [a summary row per submodule]).
    """
    modules = list(dict.fromkeys(submodule.module for submodule in submodules))
    # NOTE: only the scored weight matrices need
    # gradients. Leaving them off everywhere else (embeddings, norms, the
    # unembedding, biases) saves those gradients' memory: a float32 13B model
    # and the gradients of its matrices fill about 105 GB of a 141 GB H200.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules:
        module.weight.requires_grad_(True)
        module.weight.grad = None

    # The massive-activation filter needs each matrix's mean |input| per input
    # channel over the corpus; a hook on the matrix reads its input.
    channel_sums, token_counts = {}, {}

    def reader(module):
        def hook(_module, args):
            inputs = args[0].detach()
            # NOTE: inputs: (1, positions, n_inputs); summing |inputs|
            # over the batch and position axes leaves one total per input channel.
            channel_sums[module] = channel_sums.get(module, 0) + inputs.abs().sum(dim=(0, 1), dtype=torch.float64)
            token_counts[module] = token_counts.get(module, 0) + inputs.shape[0] * inputs.shape[1]
        return hook

    handles = [module.register_forward_pre_hook(reader(module)) for module in modules]
    device = next(model.parameters()).device
    start, loss_sum, n_answer_tokens = time.time(), 0.0, 0
    for index, (prompt_ids, answer_ids) in enumerate(sequences):
        input_ids = torch.tensor([prompt_ids + answer_ids], device=device)
        # NOTE: logits: (positions, vocabulary) of the one sequence.
        logits = model(input_ids, use_cache=False).logits[0]
        # NOTE: the logits at position t predict token t + 1, so
        # the answer's tokens (positions len(prompt) onwards) are predicted by
        # the logits one position earlier. The prompt's own tokens get no loss.
        # NOTE: predictions: (answer tokens, vocabulary), one row per
        # target in input_ids[0, len(prompt_ids):], which is (answer tokens,).
        predictions = logits[len(prompt_ids) - 1:-1]
        # NOTE: the paper's L sums the log-likelihood over
        # every token of every sequence, so a long narrative weighs more than
        # a short one; the usual mean loss would weigh every transcript equally.
        loss = F.cross_entropy(predictions, input_ids[0, len(prompt_ids):], reduction="sum")
        # NOTE: backward() adds this transcript's gradient to each
        # weight's .grad, so after the loop .grad is dL/dw of the whole corpus.
        loss.backward()
        loss_sum += loss.item()
        n_answer_tokens += len(answer_ids)
        if (index + 1) % 50 == 0 or index + 1 == len(sequences):
            log(f"  importance: {index + 1}/{len(sequences)} transcripts, {n_answer_tokens} answer tokens, "
                f"loss {loss_sum / n_answer_tokens:.3f} per token, {(time.time() - start) / (index + 1):.2f} s each")
    for handle in handles:
        handle.remove()

    top_positions, summaries = {}, []
    for submodule in submodules:
        weight = submodule.module.weight
        positions = weight_positions(submodule)
        # NOTE: weight * grad has the module's weight shape; flattening
        # it and indexing with positions gathers the submodule's (outputs, inputs) matrix.
        importance = (weight.detach() * weight.grad).abs().flatten()[positions]
        channel_mean = channel_sums[submodule.module] / token_counts[submodule.module]
        selected, details = select_top_weights(importance, positions, channel_mean)
        top_positions[submodule.name] = selected.to(torch.int32).cpu()
        # max_channel_ratio: the largest input channel's mean |activation| over
        # the median channel's, which the filter compares with MASSIVE_RATIO.
        summaries.append({"name": submodule.name, "layer": submodule.layer, "kind": submodule.kind,
                          "shape": list(weight.shape), "transposed": isinstance(submodule.module, Conv1D),
                          "n_weights": importance.numel(), "n_inputs": importance.shape[1],
                          "n_lesioned": len(selected), **details})
    for module in modules:
        module.weight.grad = None
        module.weight.requires_grad_(False)
    return top_positions, summaries


# --- step 3: lesions ------------------------------------------------------------

@contextmanager
def lesioned(modules_by_name: dict, top_positions: dict, names: list):
    """Inside the with-block, the top 1% weights of the named submodules are zero; afterwards they are restored."""
    saved = []
    with torch.no_grad():
        for name in names:
            # NOTE: view(-1) shares the weight's memory, so writing
            # into it changes the weight the model computes with.
            flat_weight = modules_by_name[name].weight.view(-1)
            positions = top_positions[name].to(flat_weight.device).long()
            saved.append((flat_weight, positions, flat_weight[positions].clone()))
            flat_weight[positions] = 0.0
    try:
        yield
    finally:
        with torch.no_grad():
            # NOTE: writing the saved values back restores the
            # weights bit for bit, with no arithmetic that could round.
            for flat_weight, positions, values in reversed(saved):
                flat_weight[positions] = values


def load_lesions(engine, names: list) -> None:
    """Let the engine's interventions lesion these weight matrices (Intervention.lesions), with step 2's top 1%.

    Reads runs/fiveway/em_baseline/<model>/top1pct.pt, so this script must
    have run on the model first (em_search.py, patching.py, benchmarks.py).
    """
    positions = torch.load(RUNS / "fiveway" / "em_baseline" / engine.model_key / "top1pct.pt")["positions"]
    modules = {submodule.name: submodule.module for submodule in find_submodules(engine.model)}
    engine.add_lesions({name: (modules[name], positions[name]) for name in names})


@torch.no_grad()
def weights_fingerprint(modules_by_name: dict) -> list:
    """The float64 sum of every scored weight matrix; a lesion left in place would change it."""
    modules = dict.fromkeys(modules_by_name.values())
    return [float(module.weight.sum(dtype=torch.float64)) for module in modules]


def generate_answer(engine, yesno_ids, story_ids) -> dict:
    """The model's gate text and answer as its weights are now, made as search.py makes them."""
    yesno, answer = "", None
    # NOTE: the weights change between calls, so every call
    # holds one intervention (the empty one) and is not padded to the engine's
    # batch size (engine.generate's pad_batches).
    if yesno_ids is not None:
        yesno = engine.generate(yesno_ids, [Intervention()], YESNO_TOKENS, DECODING, pad_batches=False)[0]
        answer = parse_yes_no(yesno)[0]
    full_text = engine.generate(story_ids, [Intervention()], STORY_TOKENS, DECODING, pad_batches=False)[0]
    # NOTE: a base model goes on to write question and answer
    # turns of its own; only its answer, before the first new question, counts.
    story = full_text if engine.spec.is_chat else cut_at_new_question(full_text)
    return {"yesno": yesno, "answer": answer, "full_text": full_text, "story": story}


# --- step 4: similarity to each human group, and the classifier -----------------

def word_tokens(text: str) -> list:
    """The lower-cased words of a text, the units BLEU-1 counts.

    NOTE: the transcripts are punctuated by CHAT
    transcription conventions (for example " , " between words) and the models
    by written English, so BLEU-1 matches words only, ignoring case. A word is
    a run of letters and digits, joined by apostrophes inside it ("don't",
    "cinderella's"); an apostrophe at a word's edge is a quotation mark
    ('over' is the word over). The transcripts write every apostrophe as ',
    while Pythia writes ’ in 10 to 50% of its answers, so ’ and ‘ count as ';
    otherwise its "don’t" would be the two words "don" and "t", which no
    transcript contains.
    """
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")   # ’ and ‘ become '
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)


class SubtypeScorer:
    """BLEU-1 and BERTScore F1 of answers against each human group of one task, and the five-way probabilities."""

    def __init__(self, references: dict, bert_scorer: BERTScorer, classifier: FiveWayScorer):
        self.references = references   # group -> transcripts
        self.reference_words = {group: [word_tokens(text) for text in texts] for group, texts in references.items()}
        self.bert_scorer = bert_scorer
        self.classifier = classifier
        self.smoothing = SmoothingFunction().method1

    def score(self, stories: list) -> list:
        """One record per story: {story, bleu1: {group: value}, bertscore: {group: value}, probabilities}."""
        # NOTE: with several references, BLEU-1 counts each word
        # of the answer up to the most times any one reference uses it (clipped
        # unigram precision), and its brevity penalty compares the answer's
        # length with the reference closest in length.
        bleu = {group: [sentence_bleu(self.reference_words[group], word_tokens(story), weights=(1, 0, 0, 0),
                                      smoothing_function=self.smoothing) for story in stories]
                for group in GROUPS}
        # NOTE: bert-score gives an empty answer an F1 of 0,
        # but under transformers 5 its encoding of the empty string fails, so
        # empty answers are left out of the call and get their 0 here.
        nonempty = [story for story in stories if story.strip()]
        bertscore = {group: {} for group in GROUPS}
        for group in GROUPS:
            if nonempty:
                # NOTE: given a list of references per answer,
                # bert-score scores the answer against each one and keeps the
                # best: the maximum over the group's transcripts.
                f1 = self.bert_scorer.score(nonempty, [self.references[group]] * len(nonempty))[2]
                bertscore[group] = dict(zip(nonempty, f1.tolist()))
        # NOTE: each call embeds the group's
        # transcripts again, a few seconds per call on a GPU. Embedding them
        # once per task (bert_score.utils.get_bert_embedding) and matching
        # answers against the stored embeddings (bert_score.utils.greedy_cos_idf)
        # would save that.
        scored = [story for story in stories if long_enough(story)]
        probabilities = dict(zip(scored, self.classifier.probabilities(scored).round(6).tolist()))
        return [{"story": story,
                 "bleu1": {group: round(bleu[group][index], 6) for group in GROUPS},
                 "bertscore": {group: round(bertscore[group].get(story, 0.0), 6) for group in GROUPS},
                 "probabilities": probabilities.get(story)}
                for index, story in enumerate(stories)]


def standardization(records: list) -> dict:
    """Each metric's mean and standard deviation for each group, across the single-submodule lesions' answers."""
    statistics = {}
    for metric in ("bleu1", "bertscore"):
        statistics[metric] = {}
        for group in GROUPS:
            values = np.array([record[metric][group] for record in records])   # NOTE: (n_lesions,)
            statistics[metric][group] = [float(values.mean()), float(values.std())]
    return statistics


def assign_subtype(record: dict, statistics: dict) -> dict:
    """The mean of the two standardized metrics for each group, and the group where it is highest.

    NOTE: standardizing per group removes what every answer
    shares, such as a larger group's references matching everything better.
    What is left says for which group an answer stands out from the other
    lesions' answers. Many lesions do not change the answer at all, and they
    all get the same scores, hence the same group.
    """
    combined = {}
    for group in GROUPS:
        z_scores = []
        for metric in ("bleu1", "bertscore"):
            mean, deviation = statistics[metric][group]
            # NOTE: when every lesion's answer scores the
            # same (deviation 0, possible in a small test), the tiny constant
            # keeps the z-score of an answer with that score at 0 instead of 0/0.
            z_scores.append((record[metric][group] - mean) / (deviation + 1e-12))
        combined[group] = float(np.mean(z_scores))
    # NOTE: on a tie, the first group in GROUPS wins.
    assigned = max(GROUPS, key=lambda group: combined[group])
    return {"combined_z": {group: round(value, 6) for group, value in combined.items()}, "assigned": assigned}


def classifier_verdict(record: dict, gate: bool) -> dict:
    """What the five-way classifier says of an answer, read as search.py reads it.

    An answer counts ("valid") when it has at least MIN_STORY_CHARS characters
    (prompts.long_enough: invisible format characters do not count)
    and, while the gate applies, the model did not say no to the yes/no
    question; an objective "works" on a valid answer whose value exceeds 0.5.

    NOTE: as in search.py, the gate applies only when the
    unlesioned model passes it. A model that already says it does not know
    Cinderella (Pythia-6.9B) would otherwise have every lesion rejected, and
    the verdicts would only say which lesions flip its "no" to "yes".
    """
    valid = long_enough(record["story"]) and (not gate or record["answer"] is not False)
    if record["probabilities"] is None:
        return {"valid": valid, "predicted_class": None, "objectives": None, "works": None}
    probabilities = np.array([record["probabilities"]])   # NOTE: (1 answer, 5 classes)
    objectives = {name: round(float(values[0]), 6) for name, values in objective_values(probabilities).items()}
    return {"valid": valid, "predicted_class": CLASSES[int(probabilities[0].argmax())], "objectives": objectives,
            "works": {name: valid and value > SUCCESS_THRESHOLD for name, value in objectives.items()}}


def describe(record: dict, statistics: dict, gate: bool) -> dict:
    """What the results keep about one lesion: its texts, both metrics, the assigned group, the classifier's verdict."""
    return {"submodules": record["submodules"], "yesno": record["yesno"], "answer": record["answer"],
            "story": record["story"], "full_text": record["full_text"],
            "bleu1": record["bleu1"], "bertscore": record["bertscore"], **assign_subtype(record, statistics),
            "probabilities": record["probabilities"], **classifier_verdict(record, gate)}


# --- caches ---------------------------------------------------------------------

def read_cache(path: Path) -> list:
    """The records of a JSON-lines cache (none when it does not exist yet).

    NOTE: a job preempted while appending can leave a
    cut-off last line, with no newline at its end. It is cut off the file here,
    before anything is appended: the next record would otherwise be written
    onto the end of it, and the next read would lose both. Its record is made
    again.
    """
    if not path.exists():
        return []
    text = path.read_text()
    if not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
        path.write_text(text)
    return [json.loads(line) for line in text.splitlines()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def write_json(path: Path, payload) -> None:
    """Write through a temporary file, so a job killed mid-write never leaves half a file behind."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1))
    temporary.replace(path)


class LesionEvaluator:
    """Generates the model's answers under lesions and scores them, caching both on disk.

    A lesion is a list of submodule names; [] is the unlesioned model. Answers
    are cached by lesion (generations_<task>.jsonl) and scores by answer text
    (scores_<task>.jsonl), so a requeued job only makes what is missing, and
    every decision it takes again (standardization, assigned groups, cluster
    growth) comes out exactly as before.
    """

    def __init__(self, engine, modules_by_name: dict, top_positions: dict, task: str,
                 subtype_scorer: SubtypeScorer, run_dir: Path, log):
        self.engine, self.modules_by_name, self.top_positions = engine, modules_by_name, top_positions
        self.subtype_scorer, self.log = subtype_scorer, log
        self.order = {name: index for index, name in enumerate(top_positions)}   # the layer-by-layer order
        self.yesno_ids, self.story_ids = task_prompt_ids(engine, task)
        self.generations_path = run_dir / f"generations_{task}.jsonl"
        self.scores_path = run_dir / f"scores_{task}.jsonl"
        self.generations = {record["key"]: record for record in read_cache(self.generations_path)}
        self.scores = {record["story"]: record for record in read_cache(self.scores_path)}

    def canonical(self, names: list) -> list:
        return sorted(names, key=self.order.get)

    def key(self, names: list) -> str:
        return ",".join(self.canonical(names)) or "unlesioned"

    def answers(self, lesions: list) -> list:
        """The answer and scores of the model under each lesion, making what the caches lack."""
        missing = [names for names in lesions if self.key(names) not in self.generations]
        start = time.time()
        for count, names in enumerate(missing, 1):
            lesion_start = time.time()
            with lesioned(self.modules_by_name, self.top_positions, names):
                record = generate_answer(self.engine, self.yesno_ids, self.story_ids)
            record = {"key": self.key(names), "submodules": self.canonical(names), **record,
                      "seconds": round(time.time() - lesion_start, 1)}
            append_jsonl(self.generations_path, record)
            self.generations[record["key"]] = record
            if count % 20 == 0 or count == len(missing):
                self.log(f"  generated {count}/{len(missing)} lesions, {(time.time() - start) / count:.1f} s each")
        records = [self.generations[self.key(names)] for names in lesions]
        new_stories = list(dict.fromkeys(record["story"] for record in records
                                         if record["story"] not in self.scores))
        if new_stories:
            score_start = time.time()
            for score in self.subtype_scorer.score(new_stories):
                append_jsonl(self.scores_path, score)
                self.scores[score["story"]] = score
            self.log(f"  scored {len(new_stories)} new answers in {time.time() - score_start:.0f} s")
        return [{**record, **self.scores[record["story"]]} for record in records]


# --- step 5: clusters -----------------------------------------------------------

def grow_cluster(subtype: str, assigned: list, ranking: list, assess, max_size: int) -> dict:
    """Lesion a subtype's submodules together, adding the next best one until the answer is assigned the subtype.

    assigned: the submodules assigned the subtype, best first; ranking: every
    lesioned submodule, best first, by its standardized score for the subtype;
    assess(names) -> the described answer of the joint lesion.

    NOTE: the paper grows a cluster "until lesioning a
    sufficient number of parameters successfully reproduced the targeted
    aphasic behavior"; here that is the joint lesion's answer being assigned
    the subtype by the same rule as the single lesions. A subtype with more
    than max_size assigned submodules starts from its max_size best, which is
    already the largest cluster allowed.
    """
    members = assigned[:max_size]
    candidates = [name for name in ranking if name not in members]
    steps = []
    while True:
        step = assess(members)
        steps.append({"size": len(members), **step})
        if step["assigned"] == subtype or len(members) >= max_size or not candidates:
            break
        members = members + [candidates.pop(0)]
    return {"n_assigned": len(assigned), "assigned_submodules": assigned, "success": steps[-1]["assigned"] == subtype,
            "size": len(members), "members": members, "steps": steps}


# --- step 6: comparison with the knockout search ----------------------------------

def input_channels(positions: torch.Tensor, summary: dict) -> torch.Tensor:
    """The input channel that the weight at each flat position of a module's weight reads."""
    # NOTE: flat position p of a (rows, columns) matrix is row
    # p // columns, column p % columns.
    rows, columns = positions // summary["shape"][1], positions % summary["shape"][1]
    # NOTE: nn.Linear stores (outputs, inputs), so the input is the
    # column; GPT-2's Conv1D stores (inputs, outputs), so it is the row.
    return rows if summary["transposed"] else columns


def circuit_overlap(circuit: dict, clusters: dict, top_positions: dict, summaries: dict, d_head: int) -> dict:
    """How many of each cluster's lesioned weights sit in the circuit's components.

    NOTE: a component's weights are where the knockout
    search zeroes it: a head is its d_head input columns of its layer's
    attention output projection (o), an MLP block its 256 input columns of
    the MLP output projection (down). A head also owns rows of q, k and v,
    and a block rows of up (and gate); those are not counted.
    """
    components = [(f"L{layer}H{head}", f"L{layer}.o", head * d_head, d_head) for layer, head in circuit["heads"]]
    components += [(f"L{layer}MLP{block}", f"L{layer}.down", block * MLP_BLOCK_SIZE, MLP_BLOCK_SIZE)
                   for layer, block in circuit["mlp_blocks"]]
    rows = []
    for component, submodule, first, width in components:
        channels = input_channels(top_positions[submodule].long(), summaries[submodule])
        n_top = int(((channels >= first) & (channels < first + width)).sum())
        n_outputs = summaries[submodule]["n_weights"] // summaries[submodule]["n_inputs"]
        # top1pct_share: the part of the component's weights in its submodule's
        # top 1%; 0.01 is what a component of average importance would get.
        rows.append({"component": component, "submodule": submodule, "n_weights": n_outputs * width,
                     "n_top1pct": n_top, "top1pct_share": round(n_top / (n_outputs * width), 6)})
    overlaps = {}
    for subtype, cluster in clusters.items():
        if cluster is None:
            continue
        n_lesioned = sum(summaries[name]["n_lesioned"] for name in cluster["members"])
        hit = [row for row in rows if row["submodule"] in cluster["members"]]
        n_in_circuit = sum(row["n_top1pct"] for row in hit)
        overlaps[subtype] = {"n_lesioned_weights": n_lesioned, "n_in_circuit": n_in_circuit,
                             "fraction_in_circuit": round(n_in_circuit / n_lesioned, 8),
                             "components_hit": [row["component"] for row in hit]}
    return {"components": rows, "clusters": overlaps}


def verdict_summary(task_lesions: dict, task_clusters: dict) -> dict:
    """The five-way classifier's verdicts on the task's answers, gathered from lesions.json and clusters.json."""
    lesions = task_lesions["lesions"]
    finals = {subtype: cluster["steps"][-1] for subtype, cluster in task_clusters.items() if cluster is not None}
    return {
        "unlesioned": {key: task_lesions["unlesioned"][key] for key in ("predicted_class", "objectives", "works")},
        "single_lesions": {
            "n": len(lesions), "n_valid": sum(lesion["valid"] for lesion in lesions),
            "predicted_class_counts": dict(Counter(str(lesion["predicted_class"]) for lesion in lesions)),
            "works": {name: [lesion["submodules"][0] for lesion in lesions if lesion["works"] and lesion["works"][name]]
                      for name in OBJECTIVES},
        },
        "clusters": {subtype: {"size": task_clusters[subtype]["size"], "success": task_clusters[subtype]["success"],
                               **{key: final[key] for key in ("predicted_class", "objectives", "works")}}
                     for subtype, final in finals.items()},
    }


def compare_with_search(model_key: str, task: str, task_lesions: dict, task_clusters: dict,
                        top_positions: dict, summaries: dict, d_head: int) -> dict:
    """Step 6 for one task: the overlap with each circuit the search found, and the classifier's verdicts."""
    comparison = {"classifier": verdict_summary(task_lesions, task_clusters), "circuits": {}}
    search_path = RESULTS / "search" / f"{model_key}_{task}_{SEARCH_TAG}.json"
    if not search_path.exists():
        return comparison
    search = json.loads(search_path.read_text())
    for objective in OBJECTIVES:
        # NOTE: an objective is missing while its search is
        # still running, and has no circuit when nothing worked.
        entry = search["objectives"].get(objective, {})
        if entry.get("found_at_stage") is None:
            continue
        circuit = entry["pruned"]["circuit"]
        comparison["circuits"][objective] = {
            "heads": circuit["heads"], "mlp_blocks": circuit["mlp_blocks"], "value": circuit["value"],
            "found_at_stage": entry["found_at_stage"],
            **circuit_overlap(circuit, task_clusters, top_positions, summaries, d_head)}
    return comparison


def print_summary(model_key: str, lesions_doc: dict, clusters_doc: dict) -> None:
    summaries = lesions_doc["submodules"]
    largest = max(summaries, key=lambda row: row["max_channel_ratio"])
    print(f"\n=== {model_key}: {len(summaries)} submodules, {sum(row['n_lesioned'] for row in summaries):,} weights "
          f"in their top 1%; {sum(row['n_massive_channels'] > 0 for row in summaries)} submodules read massive "
          f"channels ({sum(row['n_massive_channels'] for row in summaries)} channels; the largest channel is "
          f"{largest['max_channel_ratio']:.1f} times the median, in {largest['name']}); importance corpus "
          f"{lesions_doc['config']['importance_corpus']}", flush=True)
    for task, task_clusters in clusters_doc["tasks"].items():
        task_lesions = lesions_doc["tasks"][task]
        unlesioned = task_lesions["unlesioned"]
        print(f"\n{task}: unlesioned answer assigned {unlesioned['assigned']}; classifier "
              f"{unlesioned['predicted_class']}, objectives {unlesioned['objectives']}; yes/no gate "
              f"{'on' if task_lesions['gate'] else 'off (the unlesioned model answers no)'}; checks "
              f"{task_lesions['checks']}\n  {unlesioned['story'][:160]!r}", flush=True)
        print(f"  single lesions: {task_lesions['n_answers_changed']} of {len(task_lesions['lesions'])} change the "
              f"answer; assigned {task_lesions['assigned_counts']}", flush=True)
        verdicts = task_clusters["comparison"]["classifier"]["single_lesions"]
        print(f"  classifier on single lesions: {verdicts['predicted_class_counts']}; works "
              + ", ".join(f"{name}: {len(names)} {names[:6]}" for name, names in verdicts["works"].items()), flush=True)
        for subtype, cluster in task_clusters["clusters"].items():
            if cluster is None:
                print(f"  cluster {subtype}: no submodule assigned", flush=True)
                continue
            final = cluster["steps"][-1]
            print(f"  cluster {subtype}: {cluster['n_assigned']} assigned, {len(cluster['steps'])} steps, size "
                  f"{cluster['size']} ({cluster['n_members_unchanged']} change nothing alone), success "
                  f"{cluster['success']} (assigned {final['assigned']}); classifier "
                  f"{final['predicted_class']} {final['objectives']}\n    {final['story'][:120]!r}", flush=True)
        for objective, circuit in task_clusters["comparison"]["circuits"].items():
            shares = ", ".join(f"{row['component']} {row['top1pct_share']:.4f}" for row in circuit["components"])
            fractions = ", ".join(f"{subtype} {overlap['fraction_in_circuit']:.2e}"
                                  for subtype, overlap in circuit["clusters"].items())
            print(f"  search circuit {objective} (value {circuit['value']:.3f}): top-1% share of its components: "
                  f"{shares}\n    clusters' lesioned weights in it: {fractions}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--max-transcripts", type=int, help="test: only the first N Control transcripts of each task")
    parser.add_argument("--max-submodules", type=int, help="test: lesion only the first N submodules")
    parser.add_argument("--max-references", type=int,
                        help="test: only the first N transcripts of each group as references")
    parser.add_argument("--compare-only", action="store_true", help="only redo step 6 from the saved files")
    parser.add_argument("--importance-only", action="store_true",
                        help="only steps 1 and 2 (top1pct.pt), for em_search.py on a model without the baseline")
    args = parser.parse_args()

    # NOTE: a test run keeps all its files, the small ones
    # too, in a folder of its own under runs/, so a full run never resumes from
    # a test's files and results/ holds full runs only.
    test_options = [("t", args.max_transcripts), ("s", args.max_submodules), ("r", args.max_references)]
    is_test = any(value is not None for _, value in test_options)
    run_name = args.model + "".join(f"-{letter}{value}" for letter, value in test_options if value is not None)
    run_dir = RUNS / "fiveway" / "em_baseline" / run_name
    results_dir = run_dir / "results" if is_test else RESULTS / "em_baseline" / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    top_path = run_dir / "top1pct.pt"
    lesions_path, clusters_path = results_dir / "lesions.json", results_dir / "clusters.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    if not args.compare_only:
        engine = Engine(args.model, device)
        submodules = find_submodules(engine.model)
        log(f"{args.model} on {device} ({torch.get_num_threads()} CPU threads): {engine.geometry}; "
            f"{len(submodules)} submodules; run folder {run_dir}")

        # Steps 1 and 2: the importance of every weight, and each submodule's top 1%.
        if top_path.exists():
            log(f"top 1% already selected: {top_path}")
        else:
            sequences, corpus = [], {}
            for task in TASKS:
                prompt_ids = task_prompt_ids(engine, task)[1]
                texts = human_transcripts(task)["Control"][:args.max_transcripts]
                answers = [answer_token_ids(engine, text) for text in texts]
                # NOTE: 12 to 16% of the Cinderella Control
                # narratives (by tokenizer) are longer than MAX_TOKENS with their
                # prompt, and are cut there; of the sandwich ones at most one is.
                room = MAX_TOKENS - len(prompt_ids)
                sequences += [(prompt_ids, answer_ids[:room]) for answer_ids in answers]
                corpus[task] = {"n_transcripts": len(texts), "n_cut": sum(len(ids) > room for ids in answers)}
            log(f"importance corpus: {corpus}")
            top_positions, summaries = importance_top_weights(engine.model, submodules, sequences, log)
            temporary = top_path.with_name(top_path.name + ".tmp")
            torch.save({"positions": top_positions, "summaries": summaries, "corpus": corpus}, temporary)
            temporary.replace(top_path)
            log(f"wrote {top_path}: {sum(len(positions) for positions in top_positions.values()):,} weights; "
                f"massive channels in {sum(row['n_massive_channels'] > 0 for row in summaries)} submodules")
            del top_positions
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if args.importance_only:
            log("--importance-only: stopping after steps 1 and 2")
            return
        saved = torch.load(top_path)
        lesion_names = [row["name"] for row in saved["summaries"]][:args.max_submodules]
        modules_by_name = {submodule.name: submodule.module for submodule in submodules}
        fingerprint = weights_fingerprint(modules_by_name)

        classifier = FiveWayScorer(device)
        # NOTE: imported here, not at the top: bert_score
        # imports matplotlib, whose first import scans the node's fonts, and
        # the scripts that only borrow find_submodules() and load_lesions()
        # (em_search.py, mask_search.py, cem_search.py, ...) need neither.
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(lang="en", device=device.type)
        # NOTE: bert-score 0.3.13 asks its RoBERTa tokenizer
        # for add_prefix_space=True per call, which transformers 5 ignores: the
        # first word of every text would be encoded without its leading space,
        # unlike in bert-score's own setting. A tokenizer built with the option
        # restores the package's encoding.
        bert_scorer._tokenizer = AutoTokenizer.from_pretrained(bert_scorer.model_type, add_prefix_space=True)
        log(f"scorers loaded; BERTScore model {bert_scorer.model_type}, layer {bert_scorer.num_layers}")

        config = {"model": args.model, "decoding": DECODING, "story_tokens": STORY_TOKENS, "yesno_tokens": YESNO_TOKENS,
                  "top_fraction": TOP_FRACTION, "massive_ratio": MASSIVE_RATIO, "max_tokens": MAX_TOKENS,
                  "max_cluster_size": MAX_CLUSTER_SIZE, "importance_corpus": saved["corpus"],
                  "bertscore": f"{bert_scorer.model_type} layer {bert_scorer.num_layers}, F1, max over references",
                  "bleu": "nltk sentence_bleu, weights (1, 0, 0, 0), smoothing method1, lower-cased words",
                  "max_transcripts": args.max_transcripts, "max_submodules": args.max_submodules,
                  "max_references": args.max_references, "device": str(device),
                  "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
        lesions_doc = json.loads(lesions_path.read_text()) if lesions_path.exists() else {"tasks": {}}
        clusters_doc = json.loads(clusters_path.read_text()) if clusters_path.exists() else {"tasks": {}}
        lesions_doc.update(config=config, submodules=saved["summaries"])
        clusters_doc.update(config=config)

        for task in args.tasks:
            references = {group: texts[:args.max_references] for group, texts in human_transcripts(task).items()}
            n_references = {group: len(texts) for group, texts in references.items()}
            log(f"{task}: references per group {n_references}")
            subtype_scorer = SubtypeScorer(references, bert_scorer, classifier)
            evaluator = LesionEvaluator(engine, modules_by_name, saved["positions"], task, subtype_scorer, run_dir, log)
            log(f"{task}: {len(evaluator.generations)} answers and {len(evaluator.scores)} scores already cached")

            # Steps 3 and 4: the unlesioned model and every single-submodule lesion.
            records = evaluator.answers([[]] + [[name] for name in lesion_names])
            assert weights_fingerprint(modules_by_name) == fingerprint, "a lesion was not restored exactly"
            # The Cinderella gate applies when the unlesioned model says it knows the story, as in search.py.
            gate = records[0]["answer"] is not False
            statistics = standardization(records[1:])
            unlesioned = describe(records[0], statistics, gate)
            lesions = [describe(record, statistics, gate) for record in records[1:]]
            search_path = RESULTS / "search" / f"{args.model}_{task}_{SEARCH_TAG}.json"
            # NOTE: the search generated its baseline in a
            # padded batch, this run one answer at a time; other matrix shapes
            # can move a near-tie in the logits and change the greedy answer.
            same_as_search = (json.loads(search_path.read_text())["baseline"]["story"] == unlesioned["story"]
                              if search_path.exists() else None)
            lesions_doc["tasks"][task] = {
                "gate": gate,
                "n_references": n_references,
                "standardization": statistics,
                "unlesioned": unlesioned,
                "lesions": lesions,
                "assigned_counts": dict(Counter(lesion["assigned"] for lesion in lesions)),
                "n_answers_changed": sum(lesion["story"] != unlesioned["story"] for lesion in lesions),
                "checks": {"weights_restored": True, "unlesioned_equals_search_baseline": same_as_search},
            }
            write_json(lesions_path, lesions_doc)
            log(f"{task}: {len(lesions)} lesions; assigned {lesions_doc['tasks'][task]['assigned_counts']}; "
                f"{lesions_doc['tasks'][task]['n_answers_changed']} change the answer; wrote {lesions_path}")

            # Step 5: a cluster per aphasia subtype.
            def assess(names):
                return describe(evaluator.answers([names])[0], statistics, gate)

            # NOTE: every lesion that leaves the answer as it
            # was gets the unlesioned answer's group: its subtype says nothing
            # about the submodule, only about the intact model. When that group
            # is an aphasia subtype, its cluster would start from submodules
            # that change nothing, and a single one would "reproduce" the
            # subtype at once. So only lesions that change the answer are
            # assigned to clusters or added to them.
            unchanged = {lesion["submodules"][0] for lesion in lesions if lesion["story"] == unlesioned["story"]}
            changed = [lesion for lesion in lesions if lesion["submodules"][0] not in unchanged]
            clusters = {}
            for subtype in APHASIA_GROUPS:
                # NOTE: a stable sort keeps the layer order among
                # submodules with equal scores.
                ranked = sorted(changed, key=lambda lesion: -lesion["combined_z"][subtype])
                ranking = [lesion["submodules"][0] for lesion in ranked]
                assigned = [lesion["submodules"][0] for lesion in ranked if lesion["assigned"] == subtype]
                if not assigned:
                    clusters[subtype] = None
                    log(f"{task}: no submodule assigned {subtype}")
                    continue
                clusters[subtype] = grow_cluster(subtype, assigned, ranking, assess, MAX_CLUSTER_SIZE)
                clusters[subtype]["n_members_unchanged"] = sum(name in unchanged
                                                               for name in clusters[subtype]["members"])
                final = clusters[subtype]["steps"][-1]
                log(f"{task}: cluster {subtype}: {len(assigned)} assigned -> size {clusters[subtype]['size']} "
                    f"({clusters[subtype]['n_members_unchanged']} of them change nothing alone), success "
                    f"{clusters[subtype]['success']} (answer assigned {final['assigned']}, classifier "
                    f"{final['predicted_class']})")
            assert weights_fingerprint(modules_by_name) == fingerprint, "a lesion was not restored exactly"
            clusters_doc["tasks"][task] = {
                "clusters": clusters,
                "sizes": {subtype: cluster["size"] if cluster is not None and cluster["success"] else None
                          for subtype, cluster in clusters.items()},
            }
            write_json(clusters_path, clusters_doc)
            log(f"{task}: wrote {clusters_path}")

    # Step 6: the comparison with the knockout search, from the saved files.
    saved = torch.load(top_path)
    summaries = {row["name"]: row for row in saved["summaries"]}
    d_head = geometry(AutoConfig.from_pretrained(MODELS[args.model].hf_id))["d_head"]
    lesions_doc = json.loads(lesions_path.read_text())
    clusters_doc = json.loads(clusters_path.read_text())
    for task, task_entry in clusters_doc["tasks"].items():
        task_entry["comparison"] = compare_with_search(args.model, task, lesions_doc["tasks"][task],
                                                       task_entry["clusters"], saved["positions"], summaries, d_head)
    write_json(clusters_path, clusters_doc)
    log(f"compared with the search's circuits; wrote {clusters_path}")
    print_summary(args.model, lesions_doc, clusters_doc)
    log("finished")


if __name__ == "__main__":
    main()
