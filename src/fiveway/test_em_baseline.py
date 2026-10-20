"""Unit checks of em_baseline.py on tiny random models of the three families (CPU, seconds).

What each check guards against:

  layout      the submodules of a fused q/k/v matrix (GPT-2's thirds, GPT-NeoX's
              head-by-head order) cover the matrix exactly once.
  attention   the rows called q and v are the model's queries and values: with
              a submodule's q weights (and biases) zeroed, every head attends
              uniformly, so the attention output projection's input must be the
              running mean of the values computed from the v rows; with v
              zeroed, that input must be zero.
  lesion      lesioned() zeroes exactly the given weights and restores every
              weight bit for bit.
  importance  importance_top_weights() selects the top 1% of |w * dL/dw| with L
              the summed loss of the answer tokens only, checked against
              gradients of Hugging Face's own label-masked loss.
  filter      select_top_weights() never selects weights of a massive channel.
  channels    input_channels() reads the input channel of a lesioned weight
              back, for nn.Linear and for GPT-2's transposed Conv1D.
  clusters    grow_cluster() adds the next best submodule until the subtype
              is reached, and stops at the size limit or when candidates run out.
  scoring     standardization() and assign_subtype() on a small worked example;
              word_tokens(), with typographic apostrophes and quoted words; the
              de-duplicated transcript counts.
  verdict     classifier_verdict() applies the yes/no gate only while it is on,
              as search.py does.
  overlap     circuit_overlap() counts a cluster's lesioned weights in a head's
              columns of o and an MLP block's inputs of down, for nn.Linear and
              GPT-2's transposed Conv1D, on a hand-counted example.
  cache       read_cache() drops the cut-off last line a preempted job leaves,
              so the record appended after it survives the next read.

Usage:
    CUDA_VISIBLE_DEVICES= .venv/bin/python src/fiveway/test_em_baseline.py
"""

import copy
import sys
import tempfile
from pathlib import Path

import torch
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import (GROUPS, append_jsonl, assign_subtype, circuit_overlap,  # noqa: E402
                                 classifier_verdict, find_submodules, grow_cluster, human_transcripts,
                                 importance_top_weights, input_channels, lesioned, read_cache, select_top_weights,
                                 standardization, weight_positions, word_tokens)
from fiveway.models import attention_output_projection, blocks  # noqa: E402


def tiny_models() -> dict:
    torch.manual_seed(0)
    models = {
        "llama": LlamaForCausalLM(LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
                                              num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                                              max_position_embeddings=64)),
        "gpt2": GPT2LMHeadModel(GPT2Config(vocab_size=97, n_embd=32, n_layer=2, n_head=4, n_positions=64)),
        "neox": GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
                                                 num_hidden_layers=2, num_attention_heads=4,
                                                 max_position_embeddings=64, rotary_pct=0.25)),
    }
    for model in models.values():
        model.eval()
    return models


def check_layout(models):
    for family, model in models.items():
        submodules = find_submodules(model)
        kinds = [submodule.kind for submodule in submodules if submodule.layer == 0]
        expected = ["q", "k", "v", "o", "up", "down"]
        if family == "llama":
            expected = ["q", "k", "v", "o", "gate", "up", "down"]
        assert kinds == expected, (family, kinds)
        by_module = {}
        for submodule in submodules:
            by_module.setdefault(submodule.module, []).append(weight_positions(submodule).flatten())
        for module, position_lists in by_module.items():
            positions = torch.cat(position_lists)
            # Every weight of the module belongs to exactly one submodule.
            assert torch.equal(positions.sort().values, torch.arange(module.weight.numel())), family
    print("layout: ok (submodules partition every weight matrix)", flush=True)


def zero_submodule(submodule):
    """Zero all of a submodule's weights and, when its module has a bias, its bias entries."""
    with torch.no_grad():
        submodule.module.weight.view(-1)[weight_positions(submodule).flatten()] = 0.0
        if submodule.module.bias is not None:
            rows = submodule.rows if submodule.rows is not None else slice(None)
            submodule.module.bias[rows] = 0.0


def run_capturing(model, layer, input_ids):
    """(input of the q/k/v matrix holding v, input of o) of one layer, for one forward pass."""
    submodules = {submodule.kind: submodule for submodule in find_submodules(model) if submodule.layer == layer}
    captured = {}
    handles = [submodules["v"].module.register_forward_pre_hook(
                   lambda module, args: captured.__setitem__("x", args[0])),
               attention_output_projection(blocks(model)[layer]).register_forward_pre_hook(
                   lambda module, args: captured.__setitem__("o", args[0]))]
    with torch.no_grad():
        model(input_ids)
    for handle in handles:
        handle.remove()
    return captured["x"][0], captured["o"][0], submodules


def check_attention(models):
    input_ids = torch.randint(0, 97, (1, 9))
    for family, original in models.items():
        for layer in range(2):
            for scoring_part in ("q", "k"):
                model = copy.deepcopy(original)
                _, clean_o, submodules = run_capturing(model, layer, input_ids)
                zero_submodule(submodules[scoring_part])
                x, o, submodules = run_capturing(model, layer, input_ids)
                v_submodule = submodules["v"]
                # NOTE: weight: (outputs, inputs), the v submodule's own matrix.
                weight = v_submodule.module.weight.detach().flatten()[weight_positions(v_submodule)]
                bias = 0.0
                if v_submodule.module.bias is not None:
                    rows = v_submodule.rows if v_submodule.rows is not None else slice(None)
                    bias = v_submodule.module.bias.detach()[rows]
                # NOTE: x: (positions, d_model) @ weight.T: (d_model, d_model) gives
                # the values (positions, d_model), head h in columns h*d_head to (h+1)*d_head.
                values = x @ weight.T + bias
                # With q (or k) zero every attention score is zero, so each head
                # averages the values of all positions up to its own.
                counts = torch.arange(1, values.shape[0] + 1, dtype=values.dtype)[:, None]
                expected = values.cumsum(dim=0) / counts
                assert torch.allclose(o, expected, atol=1e-5), (family, layer, scoring_part, (o - expected).abs().max())
                assert not torch.allclose(clean_o, expected, atol=1e-3), (family, layer, "the check cannot fail")
            model = copy.deepcopy(original)
            submodules = {submodule.kind: submodule for submodule in find_submodules(model) if submodule.layer == layer}
            zero_submodule(submodules["v"])
            _, o, _ = run_capturing(model, layer, input_ids)
            assert torch.equal(o, torch.zeros_like(o)), (family, layer, "zero values must give zero head outputs")
    print("attention: ok (q, k and v rows are the model's queries, keys and values in every family)", flush=True)


def check_lesion(models):
    for family, model in models.items():
        submodules = find_submodules(model)
        names = ["L0.k", "L1.down", "L1.o"]
        modules_by_name = {submodule.name: submodule.module for submodule in submodules}
        by_name = {submodule.name: submodule for submodule in submodules}
        generator = torch.Generator().manual_seed(1)
        top_positions = {}
        for name in names:
            positions = weight_positions(by_name[name]).flatten()
            top_positions[name] = positions[torch.randperm(len(positions), generator=generator)[:7]].to(torch.int32)
        before = copy.deepcopy(model.state_dict())
        module_names = {module: module_name for module_name, module in model.named_modules()}
        with lesioned(modules_by_name, top_positions, names):
            changed = {key for key, value in model.state_dict().items() if not torch.equal(value, before[key])}
            for name in names:
                flat = modules_by_name[name].weight.view(-1)
                original = before[f"{module_names[modules_by_name[name]]}.weight"].view(-1)
                positions = top_positions[name].long()
                untouched = torch.ones(flat.numel(), dtype=torch.bool)
                untouched[positions] = False
                assert torch.equal(flat[positions], torch.zeros(7)), (family, name)
                assert torch.equal(flat[untouched], original[untouched]), (family, name)
        assert changed == {f"{module_names[modules_by_name[name]]}.weight" for name in names}, (family, changed)
        for key, value in model.state_dict().items():
            assert torch.equal(value, before[key]), (family, key, "not restored")
    print("lesion: ok (exactly the given weights are zeroed, and everything is restored bit for bit)", flush=True)


def check_importance(models):
    torch.manual_seed(2)
    sequences = [([5, 6, 7, 8], [11, 12, 13, 14, 15, 16]), ([5, 6, 7], [21, 22, 23, 24, 25, 26, 27, 28, 29])]
    for family, original in models.items():
        model = copy.deepcopy(original)
        submodules = find_submodules(model)
        top_positions, summaries = importance_top_weights(model, submodules, sequences, log=lambda message: None)
        assert all(row["n_massive_channels"] == 0 for row in summaries), family
        # The same gradient, from Hugging Face's loss with the prompt tokens
        # masked out (-100): its mean over answer tokens times their number is the sum.
        modules = list(dict.fromkeys(submodule.module for submodule in submodules))
        for module in modules:
            module.weight.requires_grad_(True)
        total = 0.0
        for prompt_ids, answer_ids in sequences:
            input_ids = torch.tensor([prompt_ids + answer_ids])
            labels = input_ids.clone()
            labels[0, :len(prompt_ids)] = -100
            total = total + model(input_ids, labels=labels).loss * len(answer_ids)
        gradients = dict(zip(modules, torch.autograd.grad(total, [module.weight for module in modules])))
        for submodule in submodules:
            positions = weight_positions(submodule)
            importance = (submodule.module.weight.detach() * gradients[submodule.module]).abs().flatten()[positions]
            n_selected = int(0.01 * importance.numel())
            expected = positions.flatten()[torch.topk(importance.flatten(), n_selected).indices]
            assert torch.equal(top_positions[submodule.name].long(), expected), (family, submodule.name)
    print("importance: ok (top 1% of |w * dL/dw|, L summed over answer tokens only)", flush=True)


def check_filter():
    importance = torch.rand(10, 20)
    importance[:, 3] = 100.0                 # the most important weights all read channel 3 ...
    channel_mean = torch.ones(20, dtype=torch.float64)
    channel_mean[3] = 1000.0                 # ... which is massive
    positions = torch.arange(200).view(10, 20)
    selected, details = select_top_weights(importance.clone(), positions, channel_mean)
    assert details["n_massive_channels"] == 1 and details["max_channel_ratio"] == 1000.0
    assert len(selected) == 2 and not (selected % 20 == 3).any()
    print("filter: ok (weights reading a massive channel are never selected)", flush=True)


def check_channels(models):
    for family in ("gpt2", "neox"):
        model = models[family]
        for submodule in find_submodules(model):
            if submodule.kind not in ("o", "down"):
                continue
            weight = submodule.module.weight
            summary = {"shape": list(weight.shape), "transposed": family == "gpt2"}
            positions = weight_positions(submodule)          # (outputs, inputs)
            channel = positions.shape[1] - 2
            assert torch.equal(input_channels(positions[:, channel], summary),
                               torch.full((positions.shape[0],), channel)), (family, submodule.name)
    print("channels: ok (input channels are read back for Linear and Conv1D)", flush=True)


def check_clusters():
    def assess(names):
        return {"assigned": "Broca" if "c" in names else "Control", "submodules": list(names)}
    grown = grow_cluster("Broca", ["a"], ["b", "a", "c", "d"], assess, 32)
    assert grown["members"] == ["a", "b", "c"] and grown["success"] and len(grown["steps"]) == 3, grown
    capped = grow_cluster("Broca", ["a"], ["b", "a", "c", "d"], assess, 2)
    assert capped["members"] == ["a", "b"] and not capped["success"] and len(capped["steps"]) == 2, capped
    over = grow_cluster("Broca", ["a", "b", "d"], ["b", "a", "c", "d"], assess, 2)
    assert over["members"] == ["a", "b"] and len(over["steps"]) == 1 and over["n_assigned"] == 3, over
    exhausted = grow_cluster("Broca", ["a"], ["a", "b"], assess, 32)
    assert exhausted["members"] == ["a", "b"] and not exhausted["success"], exhausted
    print("clusters: ok (growth order, success, size limit, running out of candidates)", flush=True)


def check_scoring():
    def record(bleu, bert):
        return {"bleu1": dict(zip(GROUPS, bleu)), "bertscore": dict(zip(GROUPS, bert))}
    records = [record([0.1, 0.2, 0.3, 0.4, 0.5], [0.8] * 5), record([0.3, 0.2, 0.3, 0.4, 0.5], [0.8] * 5),
               record([0.2, 0.2, 0.6, 0.4, 0.5], [0.8, 0.8, 0.9, 0.8, 0.8])]
    stats = standardization(records)
    # (The standard deviation of three equal values is 0 up to float rounding of their mean.)
    assert abs(stats["bleu1"]["Control"][0] - 0.2) < 1e-12 and stats["bertscore"]["Anomic"][1] < 1e-12
    # Control BLEU z = (0.3 - 0.2) / 0.0816 = 1.22; Broca's BLEU and BERTScore are below their means.
    assert assign_subtype(records[1], stats)["assigned"] == "Control"
    assert assign_subtype(records[2], stats)["assigned"] == "Broca"
    words = word_tokens("Cinderella's shoe , fell. Then-she ran!")
    assert words == ["cinderella's", "shoe", "fell", "then", "she", "ran"], words
    # A typographic apostrophe is an apostrophe, and quotation marks are not part of a word.
    words = word_tokens("I\u2019d spread \u2018over\u2019 it, 'and' don't stop. The stepsisters' gowns")
    assert words == ["i'd", "spread", "over", "it", "and", "don't", "stop", "the", "stepsisters", "gowns"], words
    counts = {task: {group: len(texts) for group, texts in human_transcripts(task).items()}
              for task in ("cinderella", "sandwich")}
    assert counts["cinderella"] == {"Control": 395, "Anomic": 272, "Broca": 273, "Conduction": 141,
                                    "Wernicke": 58}, counts
    assert counts["sandwich"] == {"Control": 472, "Anomic": 187, "Broca": 111, "Conduction": 81,
                                  "Wernicke": 29}, counts
    print(f"scoring: ok; de-duplicated transcripts per group {counts}", flush=True)


def check_verdict():
    probabilities = [0.1, 0.3, 0.3, 0.2, 0.1]          # objective A = 0.8, B = 0.3, C = 0.2
    story = "The prince looked for the girl everywhere."
    said_no = {"answer": False, "story": story, "probabilities": probabilities}
    with_gate, without_gate = classifier_verdict(said_no, gate=True), classifier_verdict(said_no, gate=False)
    assert not with_gate["valid"] and with_gate["works"] == {"A": False, "B": False, "C": False}, with_gate
    assert without_gate["valid"] and without_gate["works"] == {"A": True, "B": False, "C": False}, without_gate
    no_gate_question = {"answer": None, "story": story, "probabilities": probabilities}
    assert classifier_verdict(no_gate_question, gate=True)["valid"]
    too_short = {"answer": True, "story": "No.", "probabilities": None}
    assert classifier_verdict(too_short, gate=False) == {"valid": False, "predicted_class": None,
                                                         "objectives": None, "works": None}
    print("verdict: ok (the yes/no gate rejects a 'no' only while it is on)", flush=True)


def check_overlap():
    # o: nn.Linear (8 outputs, 8 inputs), 4 heads of d_head 2. Its top 1% here: four weights in
    # head 1's columns 2 and 3, one in head 3's column 6. Flat position = row * 8 + column.
    # down: GPT-2 Conv1D (512 inputs, 4 outputs), 2 blocks of 256 neurons. Flat position =
    # input * 4 + output; inputs 300 (block 1) and 10 (block 0).
    top_positions = {"L0.o": torch.tensor([2, 3, 10, 11, 6]), "L0.down": torch.tensor([300 * 4 + 1, 10 * 4 + 2]),
                     "L0.q": torch.tensor([0, 1, 2])}
    summaries = {"L0.o": {"shape": [8, 8], "transposed": False, "n_weights": 64, "n_inputs": 8, "n_lesioned": 5},
                 "L0.down": {"shape": [512, 4], "transposed": True, "n_weights": 2048, "n_inputs": 512,
                             "n_lesioned": 2},
                 "L0.q": {"shape": [8, 8], "transposed": False, "n_weights": 64, "n_inputs": 8, "n_lesioned": 3}}
    clusters = {"Broca": {"members": ["L0.o", "L0.down"]}, "Anomic": {"members": ["L0.q"]}, "Wernicke": None}
    circuit = {"heads": [[0, 1]], "mlp_blocks": [[0, 1]]}
    overlap = circuit_overlap(circuit, clusters, top_positions, summaries, d_head=2)
    head, block = overlap["components"]
    assert (head["component"], head["n_top1pct"], head["n_weights"], head["top1pct_share"]) == ("L0H1", 4, 16, 0.25)
    assert (block["component"], block["n_top1pct"], block["n_weights"]) == ("L0MLP1", 1, 1024), block
    assert overlap["clusters"]["Broca"]["fraction_in_circuit"] == round(5 / 7, 8), overlap["clusters"]["Broca"]
    assert overlap["clusters"]["Anomic"]["n_in_circuit"] == 0 and "Wernicke" not in overlap["clusters"]
    print("overlap: ok (a cluster's lesioned weights in the circuit's heads and blocks, hand-counted)", flush=True)


def check_cache():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generations.jsonl"
        assert read_cache(path) == []
        append_jsonl(path, {"key": "a"})
        append_jsonl(path, {"key": "b"})
        path.write_text(path.read_text() + '{"key": "c", "sto')     # a job killed while appending "c"
        assert read_cache(path) == [{"key": "a"}, {"key": "b"}]
        append_jsonl(path, {"key": "c"})                              # the restarted job makes "c" again
        assert read_cache(path) == [{"key": "a"}, {"key": "b"}, {"key": "c"}], path.read_text()
    print("cache: ok (a cut-off last line is dropped, and the next record survives)", flush=True)


def main():
    models = tiny_models()
    check_layout(models)
    check_attention(models)
    check_lesion(models)
    check_importance(models)
    check_filter()
    check_channels(models)
    check_clusters()
    check_scoring()
    check_verdict()
    check_overlap()
    check_cache()
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
