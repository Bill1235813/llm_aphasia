"""Batched generation under interventions: a different intervention for every batch element.

The earlier experiments (``src/interp/generate.py``) generated one text at a
time with transformer_lens, so a search over a thousand head knockouts took
most of a day. Here every element of a batch carries its own intervention,
and one ``generate`` call produces up to ``batch_size`` texts. The model is
the plain Hugging Face one; ``models.py`` explains why zeroing the inputs of
the output projections is the same knockout as transformer_lens's ``hook_z``
and ``hook_post``.

An ``Intervention`` can combine:

    heads        (layer, head) pairs zeroed at every position
    mlp_blocks   (layer, block) pairs: 256-neuron MLP blocks zeroed at every position
    steer        (layer, vector name, alpha): alpha * vector[layer] added to the
                 residual stream after that block, at every position
    remove       vector names whose direction is projected out of the residual
                 stream after every block (directional ablation)
    patch_heads  (layer, head) pairs whose clean activations are written back
                 at every prompt position, after the knockout (activation patching)
    patch_blocks (layer, block) pairs, the same for 256-neuron MLP blocks
    lesions      weight-matrix names ("L3.up") whose most important weights
                 (em_baseline.py's top 1%) are zeroed; add_lesions() registers them
    patch_lesions lesioned matrices whose clean outputs are written back at every
                 prompt position (whole matrices only: o, gate, up, down)
    head_amounts   (layer, head, amount): the head's output scaled by 1 - amount, a
                   knockout by an amount between 0 (none) and 1 (zeroed), for the
                   learned masks of mask_search.py
    lesion_amounts (name, amount): the lesion by an amount, x @ W - amount * x @ D

soft_knockout() applies such amounts as tensors that carry gradients, to train them.

NOTE: every batch is padded to the same ``batch_size`` with
copies of its first intervention. Matrix kernels are then always called with
the same shapes, so an element's numbers do not depend on what else is in its
batch, and an intervention gives the same text whether it is generated in a
search batch or alone. ``test_engine.py`` checks this.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.pytorch_utils import Conv1D

from fiveway.models import (MLP_BLOCK_SIZE, MODELS, attention_output_projection, blocks, geometry,
                            mlp_output_projection)

# Greedy decoding, plain or with a repetition penalty (Keskar et al., 2019) of
# 1.25, the setting under which the earlier study found every unablated
# generator fluent (src/interp/generate.py, DECODING_PRESETS).
REPETITION_PENALTY = {"greedy": 1.0, "rep1.25": 1.25}
LESION_KINDS = ["q", "k", "v", "o", "gate", "up", "down"]   # a layer's weight matrices, in em_baseline.py's order


def lesion_order(name: str) -> tuple:
    """Sort key of a weight-matrix name: "L3.up" -> (3, 5), layer first, then the matrix's place in the layer."""
    layer, kind = name[1:].split(".")
    return int(layer), LESION_KINDS.index(kind)


@dataclass(frozen=True)
class Intervention:
    heads: tuple = ()
    mlp_blocks: tuple = ()
    steer: tuple = ()
    remove: tuple = ()
    patch_heads: tuple = ()
    patch_blocks: tuple = ()
    lesions: tuple = ()
    patch_lesions: tuple = ()
    head_amounts: tuple = ()
    lesion_amounts: tuple = ()

    @classmethod
    def soft(cls, head_amounts: dict, lesion_amounts: dict) -> "Intervention":
        """Knockouts by amounts, {(layer, head): amount} and {name: amount}, rounded to 3 decimals.

        NOTE: rounding keeps cache keys short and drops the
        amounts a learned mask leaves near zero (below 0.0005), which change
        nothing visible; the rounded amounts are the ones applied.
        """
        heads = tuple(sorted((int(layer), int(head), round(float(amount), 3))
                             for (layer, head), amount in head_amounts.items() if round(float(amount), 3) > 0))
        lesions = tuple(sorted(((name, round(float(amount), 3)) for name, amount in lesion_amounts.items()
                                if round(float(amount), 3) > 0), key=lambda pair: lesion_order(pair[0])))
        return cls(head_amounts=heads, lesion_amounts=lesions)

    @classmethod
    def knockout(cls, heads=(), mlp_blocks=(), lesions=()) -> "Intervention":
        """A knockout of heads, MLP blocks and weight lesions, with its components in a canonical order."""
        return cls(heads=tuple(sorted(set(map(tuple, heads)))),
                   mlp_blocks=tuple(sorted(set(map(tuple, mlp_blocks)))),
                   lesions=tuple(sorted(set(lesions), key=lesion_order)))

    def key(self) -> str:
        """A string naming the intervention, used as its cache key."""
        parts = []
        if self.heads:
            parts.append("H:" + ",".join(f"L{layer}H{head}" for layer, head in self.heads))
        if self.mlp_blocks:
            parts.append("M:" + ",".join(f"L{layer}B{block}" for layer, block in self.mlp_blocks))
        if self.steer:
            parts.append("S:" + ",".join(f"L{layer}:{name}:{alpha:.6g}" for layer, name, alpha in self.steer))
        if self.remove:
            parts.append("R:" + ",".join(self.remove))
        if self.patch_heads:
            parts.append("P:" + ",".join(f"L{layer}H{head}" for layer, head in self.patch_heads))
        if self.patch_blocks:
            parts.append("PB:" + ",".join(f"L{layer}B{block}" for layer, block in self.patch_blocks))
        if self.lesions:
            parts.append("W:" + ",".join(self.lesions))
        if self.patch_lesions:
            parts.append("PW:" + ",".join(self.patch_lesions))
        if self.head_amounts:
            parts.append("SH:" + ",".join(f"L{layer}H{head}={amount:.3f}" for layer, head, amount in self.head_amounts))
        if self.lesion_amounts:
            parts.append("SW:" + ",".join(f"{name}={amount:.3f}" for name, amount in self.lesion_amounts))
        return "|".join(parts) or "clean"


@dataclass
class BatchState:
    """The per-element tensors the hooks read while one batch generates."""
    prompt_len: int
    head_scale: dict    # layer -> (batch, 1, heads x d_head): 0 on zeroed heads' columns, 1 elsewhere
    mlp_scale: dict     # layer -> (batch, 1, d_mlp)
    resid_add: dict     # layer -> (batch, 1, d_model)
    remove: list        # [(unit vectors (n_layers, d_model), flags (batch, 1, 1))]
    head_patch: dict    # layer -> (keep (batch, 1, heads x d_head), clean (1, prompt_len, heads x d_head))
    mlp_patch: dict     # layer -> (keep (batch, 1, d_mlp), clean (1, prompt_len, d_mlp))
    lesions: dict       # module -> [(delta (inputs, outputs), flags (batch, 1, 1))]
    lesion_patch: dict  # module -> (keep (batch, 1, 1), clean (1, prompt_len, outputs))


class Engine:
    """A generator model with hooks that apply one intervention per batch element."""

    def __init__(self, model_key: str, device: torch.device, dtype: torch.dtype = torch.float32):
        spec = MODELS[model_key]
        self.model_key = model_key
        self.spec = spec
        self.device = device
        self.batch_size = spec.batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        # NOTE: float32, as in every earlier experiment:
        # greedy decoding follows the argmax, and half precision moves enough
        # logits to change some texts. Sharing a GPU with the classifier is
        # no problem on an 80 or 141 GB card.
        # NOTE: device_map loads the weights straight
        # onto the GPU, so a float32 13B model never needs 52 GB of host memory.
        # NOTE: Qwen3.5's checkpoint is an image-text model
        # whose language model has a config of its own (get_text_config(); a
        # text-only config returns itself). Loaded with the checkpoint's config,
        # the language model ignores dtype and comes out in bfloat16, rounding
        # the float32 norm weights of its linear-attention layers; loaded with
        # its own config, every weight is the checkpoint's, in float32.
        config = AutoConfig.from_pretrained(spec.hf_id).get_text_config()
        self.model = AutoModelForCausalLM.from_pretrained(spec.hf_id, config=config, dtype=dtype,
                                                          attn_implementation="sdpa", device_map=device)
        self.model.eval()
        self.dtype = dtype
        self.geometry = geometry(self.model.config)
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.stop_ids = [self.eos_id] + self.tokenizer.convert_tokens_to_ids(list(spec.stop_tokens))
        # NOTE: Llama-2-chat's own generation config turns
        # sampling on (temperature 0.6, top-p 0.9), and generate() falls back
        # to the model's config for every option a call leaves at its default.
        # A blank config here makes greedy decoding really greedy.
        self.model.generation_config = GenerationConfig(
            bos_token_id=self.bos_id, eos_token_id=self.stop_ids, pad_token_id=self.eos_id)
        self.vectors = {}   # name -> (n_layers, d_model) tensor, for steer and remove
        self.lesion_deltas = {}   # weight-matrix name -> (module, delta): see add_lesions
        self._state = None
        self._capture = None
        self._blocks = blocks(self.model)
        for layer, block in enumerate(self._blocks):
            attention_output_projection(block).register_forward_pre_hook(self._attention_hook(layer))
            mlp_output_projection(block).register_forward_pre_hook(self._mlp_hook(layer))
            block.register_forward_hook(self._block_hook(layer))

    # --- prompts ----------------------------------------------------------

    def encode(self, prompt: str) -> list:
        """Token ids of ``prompt``, with the beginning-of-sequence token in front for GPT-2 and Llama.

        NOTE: this reproduces transformer_lens's tokenization,
        which the earlier experiments used: Llama's tokenizer adds BOS on its
        own, transformer_lens adds "<|endoftext|>" in front of GPT-2 prompts,
        and it adds nothing in front of Pythia prompts (models.py). Encoding
        without special tokens and adding BOS by hand gives the same ids for
        all three families. Qwen3.5 has no BOS token: its chat template's
        <|im_start|> markers are encoded as the special tokens they are.
        """
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        return [self.bos_id] + ids if self.spec.prepend_bos else ids

    @torch.no_grad()
    def add_lesions(self, lesions: dict) -> None:
        """Make weight lesions available to interventions: {name: (module, positions in its flattened weight)}.

        NOTE: zeroing some weights of W changes the output x @ W
        by -x @ D, where D holds those weights and zeros elsewhere. Subtracting
        x @ D in a forward hook lesions the matrix for some batch elements and
        leaves it whole for the others, so differently lesioned models share a
        batch, as different head knockouts do. The result equals zeroing the
        weights up to float32 rounding (test_em_search.py).
        NOTE: D is stored dense, 1% filled: 4 GB for
        GPT-2 XL's 96 MLP matrices. A sparse D (positions and values, with
        index_add_) would take 1% of that, for larger models.
        """
        for name, (module, positions) in lesions.items():
            weight = module.weight.detach()
            delta = torch.zeros(weight.numel(), device=weight.device, dtype=weight.dtype)
            delta[positions.to(weight.device).long()] = weight.reshape(-1)[positions.to(weight.device).long()]
            delta = delta.view(weight.shape)
            # NOTE: GPT-2's Conv1D computes x @ W with W (inputs, outputs);
            # nn.Linear computes x @ W.T with W (outputs, inputs). D is kept as
            # (inputs, outputs) so that x @ D is the lesioned weights' part either way.
            if not isinstance(module, Conv1D):
                delta = delta.T
            if not any(other is module for other, _ in self.lesion_deltas.values()):
                module.register_forward_hook(self._lesion_hook(module))
            self.lesion_deltas[name] = (module, delta)

    # --- hooks ------------------------------------------------------------

    def _attention_hook(self, layer: int):
        def hook(module, args):
            if self._capture is not None and ("attention", layer) in self._capture:
                self._capture[("attention", layer)] = args[0].detach().clone()
            state = self._state
            if state is None or (layer not in state.head_scale and layer not in state.head_patch):
                return None
            # NOTE: x: (batch, positions, n_heads * d_head), head h in
            # columns h*d_head to (h+1)*d_head (this layer's own n_heads and
            # d_head); scale (batch, 1, n_heads * d_head) broadcasts over positions.
            x = args[0]
            if layer in state.head_scale:
                x = x * state.head_scale[layer]
            # NOTE: with a key-value cache, generate() reads the
            # whole prompt in one forward pass and then one new token per pass,
            # so a pass as long as the prompt is the prompt pass. Patching
            # happens only there: generated positions have no clean
            # counterpart, so a patched head stays knocked out on them.
            if layer in state.head_patch and x.shape[1] == state.prompt_len:
                keep, clean = state.head_patch[layer]
                x = x * keep + clean * (1 - keep)
            return (x,) + tuple(args[1:])
        return hook

    def _mlp_hook(self, layer: int):
        def hook(module, args):
            if self._capture is not None and ("mlp", layer) in self._capture:
                self._capture[("mlp", layer)] = args[0].detach().clone()
            state = self._state
            if state is None or (layer not in state.mlp_scale and layer not in state.mlp_patch):
                return None
            # NOTE: (batch, positions, d_mlp) * (batch, 1, d_mlp).
            x = args[0]
            if layer in state.mlp_scale:
                x = x * state.mlp_scale[layer]
            if layer in state.mlp_patch and x.shape[1] == state.prompt_len:
                keep, clean = state.mlp_patch[layer]
                x = x * keep + clean * (1 - keep)
            return (x,) + tuple(args[1:])
        return hook

    def _lesion_hook(self, module):
        def hook(module_, args, output):
            if self._capture is not None and ("module", module) in self._capture:
                self._capture[("module", module)] = output.detach().clone()
            state = self._state
            if state is None or (module not in state.lesions and module not in state.lesion_patch):
                return None
            # NOTE: args[0] (batch, positions, inputs) @ delta (inputs,
            # outputs) -> (batch, positions, outputs), times flags (batch, 1, 1):
            # 1 for the elements that lesion this matrix, 0 for the others.
            for delta, flags in state.lesions.get(module, []):
                output = output - flags * (args[0] @ delta)
            # NOTE: the product is computed for the
            # whole batch even when few elements lesion the matrix; taking only
            # the flagged rows would save the rest.
            if module in state.lesion_patch and output.shape[1] == state.prompt_len:
                keep, clean = state.lesion_patch[module]
                output = output * keep + clean * (1 - keep)
            return output
        return hook

    def _block_hook(self, layer: int):
        def hook(module, args, output):
            state = self._state
            if state is None or (layer not in state.resid_add and not state.remove):
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if layer in state.resid_add:
                hidden = hidden + state.resid_add[layer]
            for units, flags in state.remove:
                unit = units[layer]
                # NOTE: hidden @ unit: (batch, positions), the coordinate
                # along the direction; [..., None] * unit turns it back into
                # (batch, positions, d_model). flags (batch, 1, 1) is 1 for the
                # elements that remove this direction.
                hidden = hidden - flags * (hidden @ unit)[..., None] * unit
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        return hook

    # --- per-batch state ----------------------------------------------------

    @torch.no_grad()
    def clean_projection_inputs(self, prompt_ids: list, sites: set) -> dict:
        """Activations on the clean prompt, for sites ("attention" or "mlp", layer) and ("module", module).

        Returns {site: (1, prompt_len, width)}: the concatenated head outputs
        (width: the layer's heads times their size), the MLP neurons (width
        d_mlp) or a lesionable weight matrix's output of an unablated run.
        """
        self._capture = {site: None for site in sites}
        try:
            self.model(torch.tensor([prompt_ids], device=self.device))
            return dict(self._capture)
        finally:
            self._capture = None

    def _build_state(self, interventions: list, prompt_ids: list) -> BatchState:
        size = len(interventions)
        d_model, d_mlp = self.geometry["d_model"], self.geometry["d_mlp"]
        heads_per_layer, head_dims = self.geometry["heads_per_layer"], self.geometry["head_dims"]

        def ones(width):
            return torch.ones(size, 1, width, device=self.device, dtype=self.dtype)

        def head_ones(layer):
            return ones(heads_per_layer[layer] * head_dims[layer])

        def columns(layer, head):
            """The columns of head ``head`` in its layer's concatenated head outputs."""
            # NOTE: Qwen3.5's softmax layers have half as
            # many heads as its linear-attention layers. A head number past
            # the layer's last head would slice no columns, a knockout that
            # silently does nothing, so it stops here instead.
            assert head < heads_per_layer[layer], f"layer {layer} has {heads_per_layer[layer]} heads, no head {head}"
            return slice(head * head_dims[layer], (head + 1) * head_dims[layer])

        head_scale, mlp_scale, resid_add, head_patch, mlp_patch = {}, {}, {}, {}, {}
        lesioned, lesion_patch = {}, {}   # name -> flags (batch, 1, 1); module -> keep (batch, 1, 1)
        for element, intervention in enumerate(interventions):
            for layer, head in intervention.heads:
                head_scale.setdefault(layer, head_ones(layer))[element, 0, columns(layer, head)] = 0
            for layer, block in intervention.mlp_blocks:
                start = block * MLP_BLOCK_SIZE
                mlp_scale.setdefault(layer, ones(d_mlp))[element, 0, start:start + MLP_BLOCK_SIZE] = 0
            for layer, name, alpha in intervention.steer:
                if layer not in resid_add:
                    resid_add[layer] = torch.zeros(size, 1, d_model, device=self.device, dtype=self.dtype)
                resid_add[layer][element, 0] += alpha * self.vectors[name][layer].to(self.dtype)
            for layer, head in intervention.patch_heads:
                head_patch.setdefault(layer, head_ones(layer))[element, 0, columns(layer, head)] = 0
            for layer, block in intervention.patch_blocks:
                start = block * MLP_BLOCK_SIZE
                mlp_patch.setdefault(layer, ones(d_mlp))[element, 0, start:start + MLP_BLOCK_SIZE] = 0
            for name in intervention.lesions:
                lesioned.setdefault(name, torch.zeros(size, 1, 1, device=self.device, dtype=self.dtype))[element] = 1
            for layer, head, amount in intervention.head_amounts:
                head_scale.setdefault(layer, head_ones(layer))[element, 0, columns(layer, head)] = 1 - amount
            for name, amount in intervention.lesion_amounts:
                lesioned.setdefault(name, torch.zeros(size, 1, 1, device=self.device, dtype=self.dtype))[element] = amount
            for name in intervention.patch_lesions:
                assert name.split(".")[1] not in ("q", "k", "v"), "q, k and v share a module; patch whole matrices only"
                lesion_patch.setdefault(self.lesion_deltas[name][0], ones(1))[element] = 0

        lesions = {}
        for name, flags in lesioned.items():
            module, delta = self.lesion_deltas[name]
            lesions.setdefault(module, []).append((delta, flags))

        if head_patch or mlp_patch or lesion_patch:
            clean = self.clean_projection_inputs(
                prompt_ids, {("attention", layer) for layer in head_patch} | {("mlp", layer) for layer in mlp_patch}
                | {("module", module) for module in lesion_patch})
            head_patch = {layer: (keep, clean[("attention", layer)]) for layer, keep in head_patch.items()}
            mlp_patch = {layer: (keep, clean[("mlp", layer)]) for layer, keep in mlp_patch.items()}
            lesion_patch = {module: (keep, clean[("module", module)]) for module, keep in lesion_patch.items()}

        remove = []
        for name in sorted({name for intervention in interventions for name in intervention.remove}):
            vectors = self.vectors[name].to(self.dtype)
            units = vectors / vectors.norm(dim=-1, keepdim=True)
            flags = torch.tensor([float(name in intervention.remove) for intervention in interventions],
                                 device=self.device, dtype=self.dtype).view(size, 1, 1)
            remove.append((units, flags))
        return BatchState(len(prompt_ids), head_scale, mlp_scale, resid_add, remove, head_patch, mlp_patch,
                          lesions, lesion_patch)

    @contextmanager
    def soft_knockout(self, head_amounts: torch.Tensor | None, lesion_amounts: dict):
        """Inside the block, every forward pass knocks out heads and lesions by differentiable amounts.

        ``head_amounts``: (n_layers, n_heads) tensor, or None for no heads (a
        layer with fewer heads than n_heads, as in Qwen3.5, uses its first ones);
        ``lesion_amounts``: {name: 0-dimensional tensor}. All elements of the
        batch get the same amounts (the batch holds transcripts, not
        interventions). Gradients flow from the model's output to the amounts.
        """
        heads_per_layer, head_dims = self.geometry["heads_per_layer"], self.geometry["head_dims"]
        head_scale = {}
        if head_amounts is not None:
            for layer in range(head_amounts.shape[0]):
                # NOTE: head_amounts[layer, :n]: (n,) for the layer's n
                # heads; repeat_interleave repeats each head's amount over its
                # d_head columns, giving (n * d_head,), viewed as (1, 1, n * d_head)
                # to broadcast over the batch and the positions of the hook's input.
                amounts = head_amounts[layer, :heads_per_layer[layer]]
                head_scale[layer] = (1 - amounts).repeat_interleave(head_dims[layer]).view(1, 1, -1)
        lesions = {}
        for name, amount in lesion_amounts.items():
            module, delta = self.lesion_deltas[name]
            lesions.setdefault(module, []).append((delta, amount.view(1, 1, 1)))
        self._state = BatchState(-1, head_scale, {}, {}, [], {}, {}, lesions, {})
        try:
            yield
        finally:
            self._state = None

    # --- generation -------------------------------------------------------

    def decode(self, token_ids) -> str:
        """The text of generated token ids, up to the first end-of-sequence (or other stop) token."""
        token_ids = token_ids.tolist()
        stops = [position for position, token_id in enumerate(token_ids) if token_id in self.stop_ids]
        if stops:
            token_ids = token_ids[:stops[0]]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate(self, prompt_ids: list, interventions: list, max_new_tokens: int,
                 decoding: str = "greedy", temperature: float = 0.0, seed: int = 0,
                 pad_batches: bool = True) -> list:
        """One continuation of the prompt per intervention, as text.

        ``decoding`` is "greedy" or "rep1.25"; ``temperature > 0`` samples
        instead (top-k 50, seeded per batch), for the steering runs that ask for it.
        ``pad_batches=False`` generates a short request as it is, without the
        padding copies: for runs that edit the weights between calls (the
        lesion baseline), where every call holds a single intervention.
        """
        texts = []
        prompt = torch.tensor([prompt_ids], device=self.device)
        options = {"max_new_tokens": max_new_tokens, "repetition_penalty": REPETITION_PENALTY[decoding]}
        if temperature > 0:
            options.update(do_sample=True, temperature=temperature, top_k=50)
        else:
            options.update(do_sample=False)
        for start in range(0, len(interventions), self.batch_size):
            chunk = interventions[start:start + self.batch_size]
            padded = chunk + [chunk[0]] * (self.batch_size - len(chunk)) if pad_batches else chunk
            self._state = self._build_state(padded, prompt_ids)
            try:
                input_ids = prompt.repeat(len(padded), 1)
                if temperature > 0:
                    torch.manual_seed(seed + start)
                output = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                             **options)
            finally:
                self._state = None
            # NOTE: output: (batch_size, prompt_len + n_generated);
            # the padding copies at the end of the batch are dropped.
            texts.extend(self.decode(row) for row in output[:len(chunk), prompt.shape[1]:])
        return texts

    @torch.no_grad()
    def next_token_logits(self, prompt_ids: list, interventions: list, pad_batches: bool = True) -> torch.Tensor:
        """Logits of the token after the prompt under each intervention: (n_interventions, vocab).

        Used by the tests and by the MMLU readout, which needs no generation.
        ``pad_batches=False`` reads short requests without the padding copies:
        a single readout has no chain of greedy choices for a last-digit
        difference to change, and padding ten conditions to a batch of 32
        would spend two thirds of every forward pass on copies.
        """
        rows = []
        prompt = torch.tensor([prompt_ids], device=self.device)
        for start in range(0, len(interventions), self.batch_size):
            chunk = interventions[start:start + self.batch_size]
            padded = chunk + [chunk[0]] * (self.batch_size - len(chunk)) if pad_batches else chunk
            self._state = self._build_state(padded, prompt_ids)
            try:
                # NOTE: logits_to_keep=1 computes the
                # unembedding for the last position only, and use_cache=False
                # stops the forward pass from keeping a key-value cache that a
                # single readout never uses; for a 2,000-token MMLU prompt
                # either would otherwise take tens of gigabytes.
                logits = self.model(prompt.repeat(len(padded), 1), use_cache=False, logits_to_keep=1).logits
            finally:
                self._state = None
            rows.append(logits[:len(chunk), -1].float().cpu())
        return torch.cat(rows)
