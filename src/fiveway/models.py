"""The generators of the five-way study, and where their hook points are.

Four model families:

    key               Hugging Face id                    layers  heads    d_head     MLP neurons
    gpt2-xl           gpt2-xl                                48     25        64         6400
    llama2-7b-chat    meta-llama/Llama-2-7b-chat-hf          32     32       128        11008
    llama2-13b-chat   meta-llama/Llama-2-13b-chat-hf         40     40       128        13824
    pythia-1b         EleutherAI/pythia-1b                   16      8       256         8192
    pythia-1.4b       EleutherAI/pythia-1.4b                 24     16       128         8192
    pythia-2.8b       EleutherAI/pythia-2.8b                 32     32        80        10240
    pythia-6.9b       EleutherAI/pythia-6.9b                 32     32       128        16384
    qwen3.5-0.8b      Qwen/Qwen3.5-0.8B                      24   8 / 16   256 / 128     3584
    qwen3.5-2b        Qwen/Qwen3.5-2B                        24   8 / 16   256 / 128     6144
    qwen3.5-4b        Qwen/Qwen3.5-4B                        32  16 / 32   256 / 128     9216
    qwen3.5-9b        Qwen/Qwen3.5-9B                        32  16 / 32   256 / 128    12288

Qwen3.5 (the post-trained checkpoints, not "-Base") mixes two kinds of layer:
every fourth layer is softmax attention (the first number of a pair), the
other three Gated DeltaNet linear attention (the second number). A
linear-attention layer has value heads, each with its own recurrent state; a
gated RMSNorm normalises each head's output on its own, and the heads are then
concatenated into the output projection, as softmax heads are. So a "head" of
a linear-attention layer is a value head, and zeroing its columns removes
everything it writes, as for a softmax head. The two kinds differ in number
and size, so geometry() lists the heads layer by layer.

Every intervention of the study acts at one of three places in each block:

  * the input of the attention output projection (``o_proj`` in Llama and in
    Qwen3.5's softmax layers, ``out_proj`` in its linear-attention layers,
    ``c_proj`` in GPT-2, ``dense`` in Pythia), which is every head's output
    concatenated, head h at columns h*d_head to (h+1)*d_head. Zeroing a slice
    there removes everything the head writes. For Llama, which has no biases,
    that is exactly transformer_lens's zeroing of ``hook_z``. For GPT-2 and
    Pythia it differs by a constant: transformer_lens folds each head's value
    bias into the shared output bias, so its zeroed ``hook_z`` still lets the
    head write W_O b_V, which this zeroing removes. Qwen3.5's softmax heads are
    gated (each output scaled by a sigmoid of the layer's input) before
    ``o_proj``; zeroing after the gate removes the gated output.
  * the input of the MLP output projection (``down_proj``, ``c_proj``,
    ``dense_4h_to_h``), the post-activation neurons: transformer_lens's
    ``hook_post``.
  * the output of the block, the residual stream after the block:
    transformer_lens's ``hook_resid_post``, where steering adds its vectors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    family: str         # "llama", "gpt2", "neox" or "qwen3_5"
    is_chat: bool       # chat models get their chat template, the base models Question/Answer prompts
    prepend_bos: bool   # whether prompts start with the beginning-of-sequence token
    batch_size: int     # interventions per generation batch (the number of heads per layer, at least 25)
    stop_tokens: tuple = ()   # tokens that also end an answer, besides the tokenizer's end-of-sequence token


# NOTE: batch_size is fixed per model for every experiment:
# matrix kernels then always see the same shapes, so an intervention's answer
# never depends on how it was batched (engine.py). The number of heads per
# layer fills one batch per step of the layer-by-layer searches; for Qwen3.5,
# 32 is the most heads a layer has (4B, 9B), or two layers' worth (0.8B, 2B).
#
# NOTE: prepend_bos follows transformer_lens, which the
# earlier experiments used: it puts BOS in front of GPT-2 and Llama prompts but
# not in front of GPT-NeoX (Pythia) prompts (its default_prepend_bos is False
# for that architecture). Pythia was trained on documents separated by
# <|endoftext|> with no BOS, and test_engine.py checks that these choices
# reproduce the earlier study's stored answers. Qwen3.5's tokenizer has no BOS
# token: its chat template starts every prompt with <|im_start|>.
#
# NOTE: a Qwen3.5 chat turn ends with <|im_end|>, its
# tokenizer's end-of-sequence token, but the model's config names
# <|endoftext|> as its end of text; an answer ends at whichever comes first.
MODELS = {
    "gpt2-xl": ModelSpec("gpt2-xl", "gpt2", False, True, 25),
    "llama2-7b-chat": ModelSpec("meta-llama/Llama-2-7b-chat-hf", "llama", True, True, 32),
    "llama2-13b-chat": ModelSpec("meta-llama/Llama-2-13b-chat-hf", "llama", True, True, 40),
    "pythia-1b": ModelSpec("EleutherAI/pythia-1b", "neox", False, False, 32),
    "pythia-1.4b": ModelSpec("EleutherAI/pythia-1.4b", "neox", False, False, 32),
    "pythia-2.8b": ModelSpec("EleutherAI/pythia-2.8b", "neox", False, False, 32),
    "pythia-6.9b": ModelSpec("EleutherAI/pythia-6.9b", "neox", False, False, 32),
    "qwen3.5-0.8b": ModelSpec("Qwen/Qwen3.5-0.8B", "qwen3_5", True, False, 32, ("<|endoftext|>",)),
    "qwen3.5-2b": ModelSpec("Qwen/Qwen3.5-2B", "qwen3_5", True, False, 32, ("<|endoftext|>",)),
    "qwen3.5-4b": ModelSpec("Qwen/Qwen3.5-4B", "qwen3_5", True, False, 32, ("<|endoftext|>",)),
    "qwen3.5-9b": ModelSpec("Qwen/Qwen3.5-9B", "qwen3_5", True, False, 32, ("<|endoftext|>",)),
}

TASKS = ["cinderella", "sandwich"]

# The search splits each MLP into contiguous blocks of this many neurons, as
# the earlier experiments did; every model's MLP width is a multiple of it.
MLP_BLOCK_SIZE = 256


def blocks(model):
    """The transformer blocks of a loaded Hugging Face model, in order."""
    family = model.config.model_type
    # NOTE: AutoModelForCausalLM loads only the language model of
    # Qwen3.5's image-text checkpoint (Qwen3_5ForCausalLM), whose config is the
    # text config, model_type "qwen3_5_text"; its blocks sit where Llama's do.
    if family in ("llama", "qwen3_5_text"):
        return list(model.model.layers)
    if family == "gpt2":
        return list(model.transformer.h)
    if family == "gpt_neox":
        return list(model.gpt_neox.layers)
    raise ValueError(f"unsupported model type {family}")


def attention_output_projection(block):
    """The linear map whose input is the concatenated head outputs (of softmax or linear-attention heads)."""
    for path in ("self_attn.o_proj", "attn.c_proj", "attention.dense", "linear_attn.out_proj"):
        module = block
        try:
            for name in path.split("."):
                module = getattr(module, name)
            return module
        except AttributeError:
            continue
    raise ValueError("no attention output projection found")


def mlp_output_projection(block):
    """The linear map whose input is the MLP's post-activation neurons."""
    for path in ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"):
        module = block
        try:
            for name in path.split("."):
                module = getattr(module, name)
            return module
        except AttributeError:
            continue
    raise ValueError("no MLP output projection found")


def geometry(config) -> dict:
    """The model's shape, read from its config.

    n_layers, d_model, d_mlp and n_blocks (256-neuron MLP blocks); then the
    heads layer by layer: heads_per_layer, head_dims and layer_types
    ("attention", or "linear_attention" for Qwen3.5's Gated DeltaNet layers).
    n_heads is the most heads any layer has, and d_head the size of every head
    when all heads have one size (None for Qwen3.5, whose two kinds differ).
    """
    # NOTE: a multimodal config (Qwen3.5's checkpoint config)
    # keeps the language model's settings in a text config of its own;
    # get_text_config() returns it, and returns any text-only config itself.
    config = config.get_text_config()
    if config.model_type == "gpt2":
        n_layers, n_heads, d_model = config.n_layer, config.n_head, config.n_embd
        d_mlp = config.n_inner or 4 * d_model
    else:
        n_layers, n_heads, d_model = config.num_hidden_layers, config.num_attention_heads, config.hidden_size
        d_mlp = config.intermediate_size
    if config.model_type == "qwen3_5_text":
        layer_types = ["attention" if kind == "full_attention" else "linear_attention" for kind in config.layer_types]
        heads_per_layer = [config.num_attention_heads if kind == "attention" else config.linear_num_value_heads
                           for kind in layer_types]
        head_dims = [config.head_dim if kind == "attention" else config.linear_value_head_dim for kind in layer_types]
    else:
        layer_types = ["attention"] * n_layers
        heads_per_layer = [n_heads] * n_layers
        head_dims = [d_model // n_heads] * n_layers
    return {"n_layers": n_layers, "n_heads": max(heads_per_layer),
            "d_head": head_dims[0] if len(set(head_dims)) == 1 else None,
            "d_model": d_model, "d_mlp": d_mlp, "n_blocks": d_mlp // MLP_BLOCK_SIZE,
            "heads_per_layer": heads_per_layer, "head_dims": head_dims, "layer_types": layer_types}
