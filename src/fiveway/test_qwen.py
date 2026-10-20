"""CPU checks of the engine on Qwen3.5, whose layers hold two kinds of head (Qwen3.5-0.8B by default).

1. Geometry. Every fourth layer is softmax attention (8 heads of 256 in the
   0.8B model), the others Gated DeltaNet linear attention (16 value heads of
   128); the engine hooks o_proj in the first and out_proj in the second.
2. Knockouts. A head knocked out through the engine gives exactly the logits
   of a hook written from scratch that zeroes the head's columns of its
   layer's output-projection input: one softmax head, one linear-attention
   head, and both together. A head number past a layer's last head is refused.
3. Batch invariance. An intervention gives bit-identical logits and the same
   text alone (padded with copies of itself) and among other interventions.
4. Prompts. The chat template is rendered with thinking off (an empty
   "<think>" block closes the prompt), with no beginning-of-sequence token
   and the special tokens encoded as such; Llama-2-chat's TAB prompts render
   exactly as before chat_prompt() existed.
5. Stops. An answer ends at <|im_end|> or <|endoftext|>.
6. MMLU. Each answer letter is one token after "Answer:".
7. The unablated answers to both tasks, printed for reading.
8. With --score: those answers scored by the five-way classifier and the
   cos-sim objective and, on a GPU, the time of a batch of story generations
   and of a batch of single-head knockouts, written to
   RUNS/fiveway/qwen_check/<model>.json.

Usage (CPU, in the shell or a CPU job; about ten minutes for the 0.8B model):
    CUDA_VISIBLE_DEVICES= .venv/bin/python src/fiveway/test_qwen.py
  every size on one GPU, with scores and timing:
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh \\
        qwen-check 96G fiveway/test_qwen.py --models qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b --score
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.benchmarks import TAB_ITEMS, answer_letter_ids  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import MODELS, attention_output_projection, blocks  # noqa: E402
from fiveway.paths import RUNS  # noqa: E402
from fiveway.prompts import STORY_TOKENS, YESNO_TOKENS, chat_prompt, parse_yes_no, task_prompt_ids  # noqa: E402
from fiveway.scorer import OBJECTIVES, CosineObjective, FiveWayScorer  # noqa: E402

THINKING_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def scratch_logits(engine, prompt_ids: list, heads: list) -> torch.Tensor:
    """Next-token logits with heads zeroed by hooks written here, at the engine's batch size (no engine state)."""
    def zeroer(columns):
        def hook(module, args):
            x = args[0].clone()
            for start, stop in columns:
                x[..., start:stop] = 0
            return (x,) + tuple(args[1:])
        return hook

    by_layer = {}
    for layer, head in heads:
        size = engine.geometry["head_dims"][layer]
        by_layer.setdefault(layer, []).append((head * size, (head + 1) * size))
    handles = [attention_output_projection(blocks(engine.model)[layer]).register_forward_pre_hook(zeroer(columns))
               for layer, columns in by_layer.items()]
    try:
        # NOTE: the engine pads every batch to batch_size,
        # so the reference runs at that size too: matrix kernels then see the
        # same shapes, and the logits can be compared bit for bit.
        prompt = torch.tensor([prompt_ids] * engine.batch_size, device=engine.device)
        with torch.no_grad():
            return engine.model(prompt, use_cache=False, logits_to_keep=1).logits[:1, -1].float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def check(model_key: str, device: torch.device, scorer) -> dict:
    """Checks 1-7 on one model, and check 8 when a scorer is given; returns what check 8 measured."""
    torch.manual_seed(0)
    engine = Engine(model_key, device)
    geometry = engine.geometry
    config = engine.model.config

    # --- 1. geometry --------------------------------------------------------
    assert geometry["layer_types"] == ["attention" if kind == "full_attention" else "linear_attention"
                                       for kind in config.layer_types]
    softmax_layers = [layer for layer, kind in enumerate(geometry["layer_types"]) if kind == "attention"]
    linear_layers = [layer for layer, kind in enumerate(geometry["layer_types"]) if kind == "linear_attention"]
    assert softmax_layers == list(range(3, geometry["n_layers"], 4))
    for layer, block in enumerate(blocks(engine.model)):
        projection = attention_output_projection(block)
        expected = block.self_attn.o_proj if layer in softmax_layers else block.linear_attn.out_proj
        assert projection is expected
        assert projection.in_features == geometry["heads_per_layer"][layer] * geometry["head_dims"][layer]
    assert geometry["d_head"] is None and geometry["n_heads"] == max(geometry["heads_per_layer"])
    print(f"1. geometry: {len(softmax_layers)} softmax layers of {geometry['heads_per_layer'][softmax_layers[0]]} heads "
          f"of {geometry['head_dims'][softmax_layers[0]]}, {len(linear_layers)} linear-attention layers of "
          f"{geometry['heads_per_layer'][linear_layers[0]]} heads of {geometry['head_dims'][linear_layers[0]]}; "
          f"hooks on o_proj and out_proj", flush=True)

    # --- 2. knockouts against hooks written from scratch -----------------------
    _, story_ids = task_prompt_ids(engine, "sandwich")
    softmax_head = (softmax_layers[1], geometry["heads_per_layer"][softmax_layers[1]] - 1)
    linear_head = (linear_layers[2], 5)
    clean = engine.next_token_logits(story_ids, [Intervention()])
    for heads in ([softmax_head], [linear_head], [softmax_head, linear_head]):
        through_engine = engine.next_token_logits(story_ids, [Intervention.knockout(heads=heads)])
        from_scratch = scratch_logits(engine, story_ids, heads)
        assert torch.equal(through_engine, from_scratch), (heads, (through_engine - from_scratch).abs().max())
        assert not torch.equal(through_engine, clean), f"knocking out {heads} changed nothing"
        print(f"2. knockout of {heads}: engine == scratch hook, bit for bit; max logit change from clean "
              f"{(through_engine - clean).abs().max():.4f}", flush=True)
    try:
        engine.next_token_logits(story_ids, [Intervention.knockout(heads=[(softmax_layers[0], geometry["n_heads"] - 1)])])
        raise RuntimeError("a head past its layer's last head was accepted")
    except AssertionError as error:
        print(f"2. a head past its layer's last head is refused: {error}", flush=True)

    # --- 3. batch invariance ----------------------------------------------------
    target = Intervention.knockout(heads=[softmax_head, (linear_layers[0], 3)])
    others = [Intervention.knockout(heads=[(layer, 0)]) for layer in linear_layers[:5]] + [Intervention()]
    alone = engine.next_token_logits(story_ids, [target])
    mixed = engine.next_token_logits(story_ids, others[:3] + [target] + others[3:])
    assert torch.equal(alone, mixed[3:4])
    text_alone = engine.generate(story_ids, [target], 24, "rep1.25")[0]
    text_mixed = engine.generate(story_ids, others[:3] + [target] + others[3:], 24, "rep1.25")[3]
    assert text_alone == text_mixed, (text_alone, text_mixed)
    print(f"3. batch invariance: logits bit-identical, text identical ({text_alone[:60]!r}...)", flush=True)

    # --- 4. prompts ---------------------------------------------------------------
    for task in ("cinderella", "sandwich"):
        yesno_ids, story_ids_task = task_prompt_ids(engine, task)
        for ids in [story_ids_task] + ([yesno_ids] if yesno_ids is not None else []):
            text = engine.tokenizer.decode(ids)
            assert text.startswith("<|im_start|>user\n") and text.endswith(THINKING_OFF), text
            assert ids.count(engine.tokenizer.convert_tokens_to_ids("<|im_start|>")) == 2
    tab_text = engine.tokenizer.decode(engine.encode(chat_prompt(engine.tokenizer, [
        {"role": "system", "content": TAB_ITEMS[0]["system"]}, {"role": "user", "content": TAB_ITEMS[0]["prompt"]}])))
    assert tab_text.startswith("<|im_start|>system\n") and tab_text.endswith(THINKING_OFF), tab_text
    llama = AutoTokenizer.from_pretrained(MODELS["llama2-7b-chat"].hf_id)
    for item in TAB_ITEMS:
        messages = [{"role": "system", "content": item["system"]}, {"role": "user", "content": item["prompt"]}]
        before = llama.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).removeprefix("<s>")
        assert chat_prompt(llama, messages) == before
    print(f"4. prompts: thinking off, no BOS, <|im_start|> encoded as one token; Llama-2's {len(TAB_ITEMS)} TAB "
          f"prompts unchanged", flush=True)

    # --- 5. stops --------------------------------------------------------------------
    im_end, end_of_text = (engine.tokenizer.convert_tokens_to_ids(token) for token in ("<|im_end|>", "<|endoftext|>"))
    assert engine.stop_ids == [im_end, end_of_text], engine.stop_ids
    words = engine.tokenizer("Once upon a time", add_special_tokens=False)["input_ids"]
    for stop in (im_end, end_of_text):
        assert engine.decode(torch.tensor(words + [stop] + words)) == "Once upon a time"
    print(f"5. stops: {engine.stop_ids} end an answer", flush=True)

    # --- 6. MMLU answer letters ----------------------------------------------------
    letters = answer_letter_ids(engine.tokenizer)
    print(f"6. MMLU letters: {[engine.tokenizer.decode([i]) for i in letters]}", flush=True)

    # --- 7. the unablated answers -----------------------------------------------------
    report = {"model": model_key, "device": str(device), "geometry": geometry, "answers": {}, "timing": {}}
    for task in ("cinderella", "sandwich"):
        yesno_ids, story_ids_task = task_prompt_ids(engine, task)
        gate = None
        if yesno_ids is not None:
            gate = engine.generate(yesno_ids, [Intervention()], YESNO_TOKENS, "rep1.25")[0]
            print(f"7. {task} gate: {gate!r} -> {parse_yes_no(gate)[0]}", flush=True)
        story = engine.generate(story_ids_task, [Intervention()], STORY_TOKENS, "rep1.25")[0]
        print(f"7. {task} answer: {story!r}", flush=True)
        report["answers"][task] = {"gate": gate, "gate_answer": None if gate is None else parse_yes_no(gate)[0],
                                   "story": story}
    if scorer is None:
        del engine
        return report

    # --- 8. scores and timing ------------------------------------------------------------
    cosine = CosineObjective(scorer)
    for task, entry in report["answers"].items():
        probabilities = scorer.probabilities([entry["story"]])[0]
        entry["probabilities"] = [round(float(p), 6) for p in probabilities]
        entry["classifier"] = {name: round(float(sum(probabilities[c] for c in columns)), 6)
                               for name, columns in OBJECTIVES.items()}
        entry["cosine"] = {name: round(float(values[0]), 6) for name, values in cosine.values([entry["story"]]).items()}
        print(f"8. {task}: classifier {entry['classifier']}, cos-sim {entry['cosine']}", flush=True)
    if device.type != "cuda":
        del engine
        return report
    _, story_ids = task_prompt_ids(engine, "sandwich")
    singles = [Intervention.knockout(heads=[(layer, head)]) for layer, n in enumerate(geometry["heads_per_layer"])
               for head in range(n)][:engine.batch_size]
    for name, batch in (("clean batch", [Intervention()] * engine.batch_size), ("single-head batch", singles)):
        torch.cuda.synchronize()
        start = time.time()
        texts = engine.generate(story_ids, batch, STORY_TOKENS, "rep1.25")
        torch.cuda.synchronize()
        seconds = time.time() - start
        tokens = sum(len(engine.tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts)
        report["timing"][name] = {"seconds": round(seconds, 2), "batch": len(batch), "answer_tokens": tokens,
                                  "tokens_per_second": round(tokens / seconds, 1)}
        print(f"8. {name} of {len(batch)} sandwich answers: {seconds:.1f} s, {tokens / seconds:.0f} answer tokens/s",
              flush=True)
    report["timing"]["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    # NOTE: the engine's hooks are closures over the
    # engine, registered on the model: a reference cycle that only the
    # garbage collector frees, before the next model is loaded.
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3.5-0.8b"],
                        choices=[key for key in MODELS if key.startswith("qwen")])
    parser.add_argument("--score", action="store_true", help="classifier and cos-sim scores, and on a GPU timing")
    args = parser.parse_args()
    device = torch.device("cuda" if args.score and torch.cuda.is_available() else "cpu")
    scorer = FiveWayScorer(device) if args.score else None
    out_folder = RUNS / "fiveway" / "qwen_check"
    for model_key in args.models:
        report = check(model_key, device, scorer)
        if args.score:
            out_folder.mkdir(parents=True, exist_ok=True)
            (out_folder / f"{model_key}.json").write_text(json.dumps(report, indent=1))
        print(f"{model_key}: all checks passed", flush=True)


if __name__ == "__main__":
    main()
