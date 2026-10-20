"""Text generation with optional attention-head and MLP-neuron knockout.

Wraps ``transformer_lens.HookedTransformer`` for the three generators used in
the project, so every interpretability experiment can:

  * build the elicitation for a task. ``cinderella`` asks a yes/no gate
    question ("Do you know the story of Cinderella?") and, separately, for the
    story; ``sandwich`` asks one question ("Tell me how you would make a
    peanut butter and jelly sandwich.") with no gate;
  * zero any subset of attention heads (per layer) and/or MLP neurons (per
    layer) during the whole forward pass;
  * parse the yes/no answer and return the story generated under the same
    ablation.

Supported models (``--model`` values; see SUPPORTED_MODELS). The geometry is
read from the loaded model's config, so the experiments are model-agnostic:

    model             layers  heads/layer  MLP neurons/layer
    gpt2-xl               48           25               6400
    llama2-7b-chat        32           32              11008
    llama2-13b-chat       40           40              13824
    pythia-1b             16            8               8192
    pythia-1.4b           24           16               8192
    pythia-2.8b           32           32              10240

GPT-2 XL and Pythia are base LMs and get bare ``Question: ... Answer:``
prompts; the Llama-2-chat models get the same questions wrapped in their chat
template.

Decoding (``--decoding`` values; see DECODING_PRESETS). Every preset is
deterministic: ``greedy`` is plain argmax decoding, and the others take the
argmax of logits that a repetition control has changed first: a repetition
penalty of 1.25 (``rep1.25``), DRY (``dry``), or no-repeat-4-gram blocking
(``ngram4``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer
from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache
from transformers.generation.logits_process import (
    LogitsProcessor,
    LogitsProcessorList,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
)


MODEL_ID = "gpt2-xl"

# Friendly aliases the CLI layer (and sbatch scripts) can pass in.
# transformer_lens.HookedTransformer.from_pretrained accepts both the
# canonical HuggingFace id and these short names.
SUPPORTED_MODELS = {
    "gpt2-xl": "gpt2-xl",
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    # Base (non-instruction-tuned) Llama-2: no chat template — uses the same
    # bare Question:/Answer: prompts as GPT-2 XL (_is_llama2_chat is False).
    "llama2-7b-base": "meta-llama/Llama-2-7b-hf",
    "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    # 13B chat: same chat-template path as 7B chat (_is_llama2_chat is True).
    # NOTE: fp32 weights are ~52 GB — needs an 80 GB A100,
    # not the 48 GB RTX A6000s that fit the 7B.
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    # Pythia (EleutherAI): base LMs trained on the Pile, same bare prompts as
    # GPT-2 XL. Their parallel attention + MLP blocks are handled by
    # transformer_lens, and the hook names (hook_z, mlp.hook_post) are the same.
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}

# NOTE: GPT-2 XL is a base language model, not instruction-
# tuned. A single combined prompt ("First answer yes or no. If yes, tell me
# the story.") gets continued as *more instructions* instead of an answer —
# the baseline run produced 0/10 parseable yes/no prefixes. Splitting the
# elicitation into two independent Question/Answer pairs is far easier for
# a base LM to complete: each prompt ends with "Answer:" so the next token
# naturally starts the reply.
#
# Q1 is used as a yes/no gate (answerable in ~5 tokens). Q2 is generated
# separately under the *same* ablation and supplies the story we score.
# We still require answer=yes on Q1 before trusting Q2's story, which is
# the original design's gate — just split across two completions.
CINDERELLA_YESNO_PROMPT = (
    "Question: Do you know the story of Cinderella?\nAnswer:"
)

CINDERELLA_STORY_PROMPT = (
    "Question: Tell me as much of the story of Cinderella as you can.\nAnswer:"
)

# The same questions, content-wise, for Llama-2-chat — used as the *user*
# message that gets wrapped in the chat template at build time.
LLAMA2_YESNO_USER = "Do you know the story of Cinderella?"
LLAMA2_STORY_USER = "Tell me as much of the story of Cinderella as you can."

# --- sandwich task -------------------------------------------------------
# NOTE: the sandwich elicitation (AphasiaBank's procedural
# discourse task) uses a SINGLE prompt with no yes/no gate — everyone "knows"
# how to make a PB&J, so the Cinderella-style familiarity check is pointless.
# GeneratorContext.build(task="sandwich") therefore sets yesno_prompt=None
# and generate_and_parse skips Q1 entirely (answer=None, which every
# experiment's gate treats as not-explicitly-no).
SANDWICH_USER = "Tell me how you would make a peanut butter and jelly sandwich."
SANDWICH_STORY_PROMPT = f"Question: {SANDWICH_USER}\nAnswer:"

# Display-only alias so log dumps and JSON payloads can reference a single
# human-readable string. Not passed into the generator directly.
CINDERELLA_PROMPT = (
    f"[Q1] {CINDERELLA_YESNO_PROMPT}  ||  [Q2] {CINDERELLA_STORY_PROMPT}"
)


def _is_llama2_chat(model_name: str) -> bool:
    return "llama" in model_name.lower() and "chat" in model_name.lower()


def _build_llama2_chat_prompt(tokenizer, user_message: str) -> str:
    """Apply Llama-2-chat's template, returning a string ready for HookedTransformer.

    NOTE: Llama-2-chat's tokenizer.apply_chat_template renders to
    ``<s>[INST] {user_message} [/INST]``. transformer_lens's generate()
    pipeline prepends BOS itself when ``prepend_bos=True``, so we strip the
    leading literal "<s>" — otherwise the tokenizer would encode those 3
    chars as plain BPE pieces in addition to the BOS, double-counting the
    sentence start and shifting every downstream position by ~3 tokens.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )
    if rendered.startswith("<s>"):
        rendered = rendered[len("<s>"):]
    return rendered


# --- yes/no parsing ------------------------------------------------------

# NOTE: We want to be permissive about the affirmative /
# negative marker at the very start of the generation, but strict about
# stopping as soon as we leave it. So each pattern matches:
#
#   ^<prefix>                 — the keyword (case-insensitive, word boundary)
#   [,.!;: \-—\n"']*          — optional punctuation/whitespace trailer
#
# and everything AFTER that trailer is treated as the story remainder.

_YES_PREFIXES = [
    r"yes",
    r"yeah",
    r"yep",
    r"yup",
    r"sure",
    r"of course",
    r"certainly",
    r"absolutely",
    r"definitely",
    r"i do remember",
    r"i remember",
    r"i do",
]

_NO_PREFIXES = [
    r"no",
    r"nope",
    r"nah",
    r"sorry",
    r"i don'?t",
    r"i do not",
    r"i cannot",
    r"i can'?t",
    r"not really",
    r"unfortunately",
]


def _build_prefix_regex(prefixes: List[str]) -> re.Pattern:
    """One big alternation: pick the longest match first."""
    # NOTE: alternations in Python's `re` are tried
    # left-to-right, so we sort by length descending to ensure that
    # "i don't remember" wins over bare "i".
    sorted_by_len = sorted(prefixes, key=len, reverse=True)
    joined = "|".join(sorted_by_len)
    # \b anchors to a word boundary so "yesterday" doesn't match "yes".
    return re.compile(rf"^\s*(?:{joined})\b[\s,.!;:\"'\-—]*", re.IGNORECASE)


_YES_RE = _build_prefix_regex(_YES_PREFIXES)
_NO_RE = _build_prefix_regex(_NO_PREFIXES)


def parse_yes_no(continuation: str) -> Tuple[Optional[bool], str]:
    """Split a GPT-2 continuation into ``(answer, trailing_text)``.

    ``answer`` is ``True`` for yes-variants, ``False`` for no-variants,
    ``None`` if no prefix matched. ``trailing_text`` is whatever remains
    after the prefix (with leading whitespace/punctuation stripped); if
    no prefix matched, ``trailing_text`` is the full continuation.
    """
    yes_match = _YES_RE.match(continuation)
    no_match = _NO_RE.match(continuation)

    # NOTE: both regexes might match empty-ish
    # continuations (rare). Prefer the longer match; on tie, yes wins.
    if yes_match and (not no_match or yes_match.end() >= no_match.end()):
        return True, continuation[yes_match.end():].lstrip(" ,.!;:\"'-—\n")
    if no_match:
        return False, continuation[no_match.end():].lstrip(" ,.!;:\"'-—\n")
    return None, continuation


# --- decoding constraints against repetition -----------------------------

# NOTE: greedy decoding of the small base LMs (Pythia) loops
# before any ablation ("I don't know. I don't know. ..."), and the classifier
# reads the loop as aphasic perseveration, so a knockout search that starts
# there has nothing left to find. The presets below keep decoding
# deterministic (argmax at every step, so one sample per condition is still
# enough) but change the logits before the argmax so the model cannot settle
# into a verbatim loop:
#
#   rep1.25  multiplicative repetition penalty (Keskar et al., 2019): every
#            token already in the context, prompt included, has its logit
#            divided by 1.25 if positive and multiplied by 1.25 if negative.
#            It hits every reused token, function words too.
#   dry      DRY, "Don't Repeat Yourself" (Weidmann et al., arXiv:2608.22761),
#            with the paper's defaults: only a token that would extend a
#            verbatim copy of an earlier span of 2 or more tokens is penalized.
#   ngram4   no-repeat n-gram blocking (Paulus et al., 2017) with n = 4, the
#            DRY paper's setting: a 4-gram can never occur twice.
DECODING_PRESETS = {
    "greedy": {},
    "rep1.25": {"repetition_penalty": 1.25},
    "dry": {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2},
    "ngram4": {"no_repeat_ngram_size": 4},
}

# DRY's default sequence breakers: newline, colon, quotation mark, asterisk.
DRY_SEQUENCE_BREAKERS = ["\n", ":", '"', "*"]


class DRYLogitsProcessor(LogitsProcessor):
    """DRY penalty, as defined in Appendix A of Weidmann et al. (arXiv:2608.22761).

    For a candidate next token ``v``, ``n(v)`` is the length of the longest
    suffix of the context that also occurs earlier in the context and was
    followed there by ``v``: generating ``v`` would continue that earlier span
    verbatim. If ``n(v) >= allowed_length`` and ``v`` is not a sequence
    breaker, DRY subtracts ``multiplier * base ** (n(v) - allowed_length)``
    from ``v``'s logit. A suffix never extends across a breaker token.
    """

    def __init__(self, multiplier: float, base: float, allowed_length: int,
                 breaker_ids: set, max_match_length: int = 50):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.breaker_ids = breaker_ids
        # NOTE: the cap of 50 comes from the reference
        # implementation (text-generation-webui PR 5677). It keeps base ** n
        # finite; at n = 50 the penalty is already about 4e11, a hard ban.
        self.max_match_length = max_match_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # NOTE: input_ids: batch x context_len (prompt and generated
        # tokens), scores: batch x d_vocab, the logits of the next token.
        for context_ids, row_scores in zip(input_ids, scores):
            context = context_ids.tolist()
            last = len(context) - 1
            if context[last] in self.breaker_ids:
                continue
            # Longest match n(v) for every candidate v that followed an
            # earlier copy of the last token.
            match_lengths = {}
            for i in range(last):
                if context[i] != context[last]:
                    continue
                candidate = context[i + 1]
                if candidate in self.breaker_ids:
                    continue
                # The spans ending at i and at last match for `length` tokens;
                # walk both back one token at a time while they still agree.
                length = 1
                while length < self.max_match_length:
                    earlier, current = i - length, last - length
                    if earlier < 0 or context[earlier] != context[current]:
                        break
                    if context[current] in self.breaker_ids:
                        break
                    length += 1
                match_lengths[candidate] = max(length, match_lengths.get(candidate, 0))
            for candidate, length in match_lengths.items():
                if length >= self.allowed_length:
                    row_scores[candidate] -= self.multiplier * self.base ** (length - self.allowed_length)
        # NOTE: this scan is O(context_len x match
        # length) in Python per step, a few ms for our ~250-token contexts.
        # llama.cpp runs the same search in C++ with a Z-algorithm on the
        # reversed context, which is linear and stops at the first breaker.
        return scores


def dry_breaker_ids(tokenizer) -> set:
    """Every token whose text contains a DRY sequence breaker.

    NOTE: the breakers are characters, but tokenizers merge
    them with their neighbours: GPT-2 and Pythia have single tokens for
    "\\n\\n", for '."' and for ':"'. Following llama.cpp's DRY sampler, a token
    breaks a match if its text contains any breaker character, so a paragraph
    break stops matching the same way a single newline does.
    """
    breaker_ids = set()
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        if any(breaker in text for breaker in DRY_SEQUENCE_BREAKERS):
            breaker_ids.add(token_id)
    return breaker_ids


def build_logits_processors(decoding: str, breaker_ids: set) -> LogitsProcessorList:
    """The logit adjustments of one ``DECODING_PRESETS`` entry, in a list.

    NOTE: each Hugging Face ``LogitsProcessor`` maps
    ``(input_ids, scores)`` to new scores, so the three controls compose by
    simply applying them in turn; a preset that combines two of them would
    only need both keys in its dict.
    """
    preset = DECODING_PRESETS[decoding]
    processors = LogitsProcessorList()
    if "repetition_penalty" in preset:
        processors.append(RepetitionPenaltyLogitsProcessor(preset["repetition_penalty"]))
    if "dry_multiplier" in preset:
        processors.append(DRYLogitsProcessor(
            preset["dry_multiplier"], preset["dry_base"], preset["dry_allowed_length"], breaker_ids,
        ))
    if "no_repeat_ngram_size" in preset:
        processors.append(NoRepeatNGramLogitsProcessor(preset["no_repeat_ngram_size"]))
    return processors


def greedy_with_processors(model: HookedTransformer, tokens: torch.Tensor, max_new_tokens: int,
                           processors: LogitsProcessorList) -> torch.Tensor:
    """Greedy decoding where ``processors`` adjust the logits before each argmax.

    ``HookedTransformer.generate`` cannot take logits processors, so this is
    its greedy path written out: the same key-value cache and the same stop at
    EOS. Takes the prompt's tokens (``model.to_tokens(prompt)``, BOS
    prepended) and returns them followed by the generated tokens, like
    ``generate(tokens, do_sample=False)``; with an empty ``processors`` the
    two agree exactly (``diag_decoding_smoke.py`` checks this).
    """
    # NOTE: tokens: 1 x context_len, the prompt followed by every
    # token generated so far.
    kv_cache = TransformerLensKeyValueCache.init_cache(model.cfg, model.cfg.device, 1)
    # NOTE: with a key-value cache, the first forward pass reads
    # the whole prompt and every later pass reads only the newest token; the
    # cache holds the keys and values of all earlier positions.
    new_tokens = tokens
    for _ in range(max_new_tokens):
        logits = model(new_tokens, past_kv_cache=kv_cache)
        # NOTE: logits: 1 x new_len x d_vocab; the last position
        # predicts the next token. clone() because processors edit in place.
        scores = processors(tokens, logits[:, -1, :].clone())
        next_token = scores.argmax(dim=-1, keepdim=True)  # 1 x 1
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == model.tokenizer.eos_token_id:
            break
        new_tokens = next_token
    return tokens


# --- generator wrapper ---------------------------------------------------


def task_prompts(tokenizer, is_chat: bool, task: str) -> Tuple[Optional[str], str]:
    """The rendered ``(yesno_prompt, story_prompt)`` of a task; no yes/no prompt for sandwich."""
    if task == "sandwich":
        # Single-prompt task: no yes/no gate at all.
        if is_chat:
            return None, _build_llama2_chat_prompt(tokenizer, SANDWICH_USER)
        return None, SANDWICH_STORY_PROMPT
    if is_chat:
        return (_build_llama2_chat_prompt(tokenizer, LLAMA2_YESNO_USER),
                _build_llama2_chat_prompt(tokenizer, LLAMA2_STORY_USER))
    return CINDERELLA_YESNO_PROMPT, CINDERELLA_STORY_PROMPT


@dataclass
class GeneratorContext:
    """Shared state for a batch of generations with knockouts.

    ``yesno_prompt`` and ``story_prompt`` are the rendered Q1/Q2 strings
    fed directly to ``model.generate``. For GPT-2 XL these are the bare
    ``Question:/Answer:`` templates; for Llama-2-chat they're produced by
    ``tokenizer.apply_chat_template`` so the model sees its expected
    ``[INST] ... [/INST]`` framing.
    """

    model: HookedTransformer
    device: torch.device
    model_name: str
    yesno_prompt: Optional[str]  # None for tasks without a yes/no gate (sandwich)
    story_prompt: str
    is_chat: bool  # True for Llama-2-chat (uses chat template + Llama-2 hook names)
    decoding: str = "greedy"  # a DECODING_PRESETS key
    breaker_ids: frozenset = frozenset()  # DRY sequence-breaker token ids

    @classmethod
    def build(
        cls,
        device: Optional[torch.device] = None,
        model_name: str = MODEL_ID,
        task: str = "cinderella",
        decoding: str = "greedy",
    ) -> "GeneratorContext":
        """Load ``model_name`` via transformer_lens; pre-render Q1/Q2 prompts.

        Accepts either a HuggingFace id (``meta-llama/Llama-2-7b-chat-hf``)
        or a friendly alias from ``SUPPORTED_MODELS`` (``llama2-7b-chat``).
        ``decoding`` picks a ``DECODING_PRESETS`` entry for every generation.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Resolve aliases to the canonical id transformer_lens expects.
        resolved = SUPPORTED_MODELS.get(model_name, model_name)
        if _is_llama2_chat(resolved):
            # NOTE: Llama-2-7B in fp32 is ~28 GB on the GPU
            # — fits on the A100s we run on (40-80 GB) with margin for
            # activations + classifier + KV cache. We tried fp16 and bf16
            # via ``from_pretrained_no_processing`` first, but both ran into
            # the same mixed-precision RoPE bug inside transformer_lens
            # ("expected scalar type Float but found Half/BFloat16" at the
            # q @ k matmul): the cached rotary cos/sin buffers stay fp32
            # while the model weights drop to reduced precision, and the
            # attention path can't reconcile the two dtypes. Loading in fp32
            # sidesteps the dtype dance entirely and the foldable
            # ``from_pretrained`` path now has the 90 GB CPU-RAM headroom
            # it needs (the job must request it).
            model = HookedTransformer.from_pretrained(resolved, device=device)
        else:
            model = HookedTransformer.from_pretrained(resolved, device=device)
        model.eval()

        is_chat = _is_llama2_chat(resolved)
        yesno_prompt, story_prompt = task_prompts(model.tokenizer, is_chat, task)

        return cls(
            model=model, device=device, model_name=resolved,
            yesno_prompt=yesno_prompt, story_prompt=story_prompt, is_chat=is_chat,
            decoding=decoding, breaker_ids=frozenset(dry_breaker_ids(model.tokenizer)),
        )

    @property
    def n_layers(self) -> int:
        return self.model.cfg.n_layers

    @property
    def n_heads(self) -> int:
        return self.model.cfg.n_heads

    @property
    def d_mlp(self) -> int:
        return self.model.cfg.d_mlp


def _make_head_hook(layer: int, heads_to_zero: List[int]):
    """Return a hook that zeros the listed heads at ``hook_z`` of ``layer``.

    NOTE: the activation at ``blocks.<l>.attn.hook_z`` is
    ``(batch, pos, n_heads, d_head)``. Zeroing along the ``n_heads`` axis
    kills those heads' contribution to the residual stream before W_O.
    """
    heads_tensor = torch.tensor(heads_to_zero, dtype=torch.long)

    def hook(z, hook):
        # NOTE: heads_to_zero may be empty — treat as no-op.
        if heads_tensor.numel() == 0:
            return z
        z[:, :, heads_tensor, :] = 0.0
        return z

    name = f"blocks.{layer}.attn.hook_z"
    return name, hook


def _make_mlp_hook(layer: int, neurons_to_zero: List[int]):
    """Return a hook that zeros the listed MLP neurons at ``hook_post``.

    NOTE: ``blocks.<l>.mlp.hook_post`` is ``(batch, pos, d_mlp)``;
    zeroing along the last dim removes those neurons' activations before
    W_out projects them back to the residual stream.
    """
    neurons_tensor = torch.tensor(neurons_to_zero, dtype=torch.long)

    def hook(act, hook):
        if neurons_tensor.numel() == 0:
            return act
        act[:, :, neurons_tensor] = 0.0
        return act

    name = f"blocks.{layer}.mlp.hook_post"
    return name, hook


def _assemble_hooks(
    head_mask: Optional[Dict[int, List[int]]],
    mlp_mask: Optional[Dict[int, List[int]]],
) -> List[Tuple[str, Callable]]:
    fwd_hooks = []
    if head_mask:
        for layer, heads in head_mask.items():
            fwd_hooks.append(_make_head_hook(layer, heads))
    if mlp_mask:
        for layer, neurons in mlp_mask.items():
            fwd_hooks.append(_make_mlp_hook(layer, neurons))
    return fwd_hooks


def generate(
    ctx: GeneratorContext,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_k: int = 50,
    seed: Optional[int] = None,
    head_mask: Optional[Dict[int, List[int]]] = None,
    mlp_mask: Optional[Dict[int, List[int]]] = None,
) -> str:
    """Generate a continuation of ``prompt``, optionally with knockouts.

    The hooks installed here stay active across every forward pass
    internal to ``HookedTransformer.generate`` (one pass per token during
    KV-cache-free decoding), so *every* generated token sees the same
    ablated circuit.
    """
    if seed is not None:
        # NOTE: reproducibility matters for the
        # ablation comparisons — same seed + same ablation = same output.
        torch.manual_seed(seed)

    fwd_hooks = _assemble_hooks(head_mask, mlp_mask)

    # NOTE: temperature=0 flips to greedy argmax decoding.
    # Naively dividing logits by 0 would blow up, so we route zero (and
    # negatives) through ``do_sample=False``, which ignores temperature
    # and top_k entirely and returns argmax at every step. Under greedy,
    # the output is a deterministic function of the prompt + ablation, so
    # the seed no longer matters — we still pass it so any RNG call that
    # slips in is reproducible, but the caller should expect identical
    # outputs for different seeds when temperature == 0.
    greedy = temperature <= 0.0
    # NOTE: the repetition presets are written for greedy
    # decoding only, the setting of every search; sampling under them would
    # need the processors inside a sampling loop.
    assert greedy or ctx.decoding == "greedy", "repetition presets need temperature 0"

    # NOTE: ``ctx.model.hooks(fwd_hooks=...)`` is a context
    # manager that registers the hooks on entry and removes them on exit,
    # so we don't leak hook state into later calls.
    with torch.no_grad():
        with ctx.model.hooks(fwd_hooks=fwd_hooks):
            if ctx.decoding != "greedy":
                # NOTE: plain greedy keeps transformer_lens's
                # own generate(), so every earlier result stays reproducible
                # bit for bit; only the repetition presets use our loop.
                processors = build_logits_processors(ctx.decoding, ctx.breaker_ids)
                tokens = greedy_with_processors(ctx.model, ctx.model.to_tokens(prompt), max_new_tokens, processors)
                # Decoded as generate(..., return_type="str") decodes: prompt and continuation together.
                out = ctx.model.tokenizer.decode(tokens[0], skip_special_tokens=True)
            elif greedy:
                out = ctx.model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_type="str",
                    verbose=False,
                )
            else:
                out = ctx.model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    do_sample=True,
                    return_type="str",
                    verbose=False,
                )
    # ``out`` contains the prompt prepended; strip it off.
    continuation = out[len(prompt):]
    return continuation


def generate_and_parse(
    ctx: GeneratorContext,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_k: int = 50,
    seed: Optional[int] = None,
    head_mask: Optional[Dict[int, List[int]]] = None,
    mlp_mask: Optional[Dict[int, List[int]]] = None,
    yesno_max_new_tokens: int = 12,
) -> Dict:
    """Issue both Q1 (yes/no) and Q2 (story) under the same ablation.

    The same ``head_mask`` / ``mlp_mask`` is applied to both completions,
    so the yes/no decision and the story both see the same ablated
    circuit — the gate is consistent with what produced the story.

    NOTE: the two completions are issued
    sequentially here for code clarity. They could be batched in a single
    forward-pass stack (batch dim = 2) to cut the sampling loop's
    per-token overhead roughly in half; the ablation hooks already
    broadcast across the batch axis. Skipped for simplicity — each
    generation is already dominated by the KV-cache-free decoding cost.
    """
    if ctx.yesno_prompt is None:
        # Gate-free task (sandwich): answer=None passes every experiment's
        # "answer is not False" gate, so the story alone decides scoring.
        yesno_continuation = ""
        answer = None
    else:
        yesno_continuation = generate(
            ctx,
            prompt=ctx.yesno_prompt,
            max_new_tokens=yesno_max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            head_mask=head_mask,
            mlp_mask=mlp_mask,
        )
        answer, _ = parse_yes_no(yesno_continuation)

    # NOTE: same seed for both prompts — they're
    # conditioned on different strings so the continuations are not
    # correlated, and reusing the seed makes the whole trial bit-exact
    # reproducible from a single integer.
    story_continuation = generate(
        ctx,
        prompt=ctx.story_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        head_mask=head_mask,
        mlp_mask=mlp_mask,
    )

    return {
        "yesno_prompt": ctx.yesno_prompt,
        "yesno_continuation": yesno_continuation,
        "answer": answer,
        "story_prompt": ctx.story_prompt,
        "story_continuation": story_continuation,
        "story": story_continuation.strip(),
    }


if __name__ == "__main__":
    # Smoke test: one unablated pair of generations.
    ctx = GeneratorContext.build()
    print(f"Model: {MODEL_ID} | n_layers={ctx.n_layers} n_heads={ctx.n_heads} d_mlp={ctx.d_mlp}")
    out = generate_and_parse(ctx, seed=0, max_new_tokens=150)
    print(f"\n--- yes/no continuation ---\n{out['yesno_continuation']!r}")
    print(f"answer = {out['answer']}")
    print(f"\n--- story continuation ---\n{out['story_continuation'][:200]}")
