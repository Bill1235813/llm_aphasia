"""Checks of the learned-mask search (mask_search.py) and the cross-entropy search (cem_search.py).

  1. The cross-entropy loop, on a made-up objective that only a pair of
     components together satisfies (no model): it finds the pair within its
     rounds, and a second run with the same seed makes the same draws.
  2. speaker_of() on made-up file names shaped like AphasiaBank's.
  2b. The lambda path (no model): with a made-up loss whose gradient in
     amount i is a constant c_i, component i must cross k = 0.5 at exactly the
     stage j where lambda_max / 2^j first falls below -c_i, and stay there.
  3. Knockout amounts in generation (Intervention.soft): amount 1 gives the
     binary knockout's logits bit for bit; soft, binary and clean knockouts
     in one batch give each one's logits alone, bit for bit; amounts under
     0.0005 are dropped.
  4. Knockout amounts in training (Engine.soft_knockout): a forward pass with
     amount tensors gives the logits of the same amounts as an Intervention
     (up to the rounding of a batch of one against a padded batch).
  5. transcript_log_likelihoods(): the same values for a sequence alone, in a
     padded batch, and summed by hand from full logits.
  6. The gradient: in float64, autograd's derivative of a transcript's
     log-likelihood with respect to head and lesion amounts matches central
     finite differences; the model's weights get no gradient and do not change.
  7. Training: two stages of three steps on the real transcripts give finite
     losses and save a state; a second call resumes from it without training
     again; a run stopped after stage 1 and resumed matches an uninterrupted
     run; the model's weights get no gradient and do not change.

Usage (GPU; gpt2-xl checks GPT-2's Conv1D lesions, pythia-1b nn.Linear):
    PARTITION=<partition> GRES=gpu:1 bash slurm/fiveway_submit.sh learned-check 40G \\
        fiveway/test_learned_searches.py --model pythia-1b
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import mask_search  # noqa: E402
from fiveway.cem_search import cross_entropy_search  # noqa: E402
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.knockout_spaces import mlp_matrix_names, space_components  # noqa: E402
from fiveway.mask_search import amounts_of, speaker_of, transcript_log_likelihoods  # noqa: E402
from fiveway.prompts import task_prompt_ids  # noqa: E402
from fiveway.search import knockout_of  # noqa: E402

MAX_BATCH_DIFFERENCE = 1e-4   # float32 rounding of a batch of one against a padded batch


class PairObjective:
    """Works only when components 3 and 7 are both knocked out; alone, 7 helps a little more than 3."""

    def value(self, record):
        chosen = set(record["chosen"])
        if {3, 7} <= chosen:
            return 0.9 - 0.01 * len(chosen)
        return 0.2 if 7 in chosen else 0.15 if 3 in chosen else 0.1

    def works(self, value):
        return value is not None and value > 0.5


class FakeEvaluator:
    def records(self, interventions):
        return [{"chosen": [int(name[1:].split(".")[0]) for name in intervention.lesions], "heads": [], "mlp_blocks": [],
                 "lesions": list(intervention.lesions), "probabilities": None, "cosine": None, "answer": None,
                 "story": ""} for intervention in interventions]


def check_cross_entropy_loop():
    components = [("lesion", f"L{i}.up") for i in range(20)]
    objective = PairObjective()
    # NOTE: lesion names here are fake ("L3.up" stands for
    # component 3); the fake evaluator reads the number back from the name.
    evaluator = FakeEvaluator()
    floor = 1 / len(components)
    start_p = np.full(len(components), floor)
    start_p[[3, 5, 9]] = 0.3   # component 7 is not among the starting favourites
    first = cross_entropy_search(evaluator, objective, components, start_p, floor, seed=0, log=lambda m: None)
    second = cross_entropy_search(evaluator, objective, components, start_p, floor, seed=0, log=lambda m: None)
    assert first["works"], first["history"]
    assert {"L3.up", "L7.up"} <= set(first["found"]["lesions"]), first["found"]
    assert first["history"] == second["history"], "same seed, different draws"
    print(f"1. cross-entropy loop: found the pair at round {first['found_at_iteration']} "
          f"({len(first['found']['lesions'])} components); same draws with the same seed", flush=True)


def check_speakers():
    # Made-up names with the shapes of AphasiaBank's file names.
    examples = {"site34a.cha_aphasia": "site34", "site34b.cha_aphasia": "site34", "PWA-900-1.gem.cex": "900",
                "900-2.gem0.cex": "900", "9034-ABCD.cha_aphasia": "9034", "XYZ19.cha_control": "XYZ19",
                "XYZ09d.cha_aphasia": "XYZ09", "place01a.gem.cex": "place01", "town15a 2.gem.cex": "town15",
                "town15b.gem.cex": "town15"}
    for file_name, speaker in examples.items():
        assert speaker_of(file_name) == speaker, (file_name, speaker_of(file_name))
    print("2. speaker_of: every example right", flush=True)


def check_lambda_path():
    gradients = torch.tensor([-1.0, -0.4, -0.3, -0.2, -0.1, -0.05, 0.2])
    expected = [1, 2, 2, 3, 4, 5, None]   # the first stage j with 1 / 2^j < -c_i (lambda_max = 1)

    def linear_loss(engine, data, components, amounts, target, contrast):
        k = amounts()
        loss = (gradients * k).sum()
        loss.backward()
        return float(loss), torch.zeros(len(target) + len(contrast))

    saved_functions = mask_search.backward_data_loss, mask_search.validation
    mask_search.backward_data_loss = linear_loss
    mask_search.validation = lambda *arguments: {"loss": 0.0}
    try:
        engine = type("FakeEngine", (), {"device": torch.device("cpu")})()
        data = {"split": {"A": {"train_target": torch.arange(4), "train_contrast": torch.arange(4),
                                "validation_target": [], "validation_contrast": []}}}
        with tempfile.TemporaryDirectory() as folder:
            trained = mask_search.train(engine, data, [("lesion", f"L{i}.up") for i in range(len(gradients))], "A",
                                        Path(folder) / "state.pt", seed=0, log=lambda message: None)
    finally:
        mask_search.backward_data_loss, mask_search.validation = saved_functions
    assert abs(trained["lambda_max"] - 1.0) < 1e-6, trained["lambda_max"]
    above = np.array([[a > 0.5 for a in checkpoint["amounts"]] for checkpoint in trained["checkpoints"]])
    for i, stage in enumerate(expected):
        joined = [j + 1 for j in range(len(above)) if above[j, i]]
        assert (joined == list(range(stage, len(above) + 1))) if stage else not joined, (i, stage, joined)
    print(f"2b. lambda path: every component joins at its expected stage {expected}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-1b")
    args = parser.parse_args()
    check_cross_entropy_loop()
    check_speakers()
    check_lambda_path()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    for parameter in engine.model.parameters():
        parameter.requires_grad_(False)
    names = mlp_matrix_names(engine)
    load_lesions(engine, names)
    _, story_ids = task_prompt_ids(engine, "cinderella")
    heads = [(0, 1), (1, 3), (engine.geometry["n_layers"] - 1, 0)]
    lesions = [names[0], names[3]]

    # 3. Amounts in generation.
    binary = Intervention.knockout(heads=heads, lesions=lesions)
    full = Intervention.soft({head: 1.0 for head in heads}, {name: 1.0 for name in lesions})
    logits = engine.next_token_logits(story_ids, [binary, full])
    assert torch.equal(logits[0], logits[1]), "amount 1 differs from the binary knockout"
    soft = Intervention.soft({heads[0]: 0.3, heads[1]: 0.0004}, {lesions[0]: 0.7})
    assert soft.head_amounts == ((heads[0][0], heads[0][1], 0.3),) and soft.lesion_amounts == ((lesions[0], 0.7),)
    assert Intervention.soft({heads[0]: 0.0001}, {}).key() == "clean"
    mixed = [soft, binary, Intervention(), Intervention.soft({heads[2]: 0.5}, {lesions[1]: 0.25})]
    together = engine.next_token_logits(story_ids, mixed)
    for row, intervention in enumerate(mixed):
        alone = engine.next_token_logits(story_ids, [intervention])[0]
        assert torch.equal(alone, together[row]), intervention.key()
    print("3. amount 1 = binary knockout bit for bit; soft, binary and clean in one batch = each alone", flush=True)

    # 4. Amount tensors in a forward pass against the same amounts as an Intervention.
    components = space_components(engine, "3_mlp_lesions_and_heads")
    k = torch.zeros(len(components), device=device)
    chosen = {("head", heads[0]): 0.3, ("head", heads[2]): 0.8, ("lesion", lesions[0]): 0.6, ("lesion", lesions[1]): 1.0}
    for index, component in enumerate(components):
        k[index] = chosen.get(component, 0.0)
    head_amounts, lesion_amounts = amounts_of(k, components, engine.geometry)
    with torch.no_grad(), engine.soft_knockout(head_amounts, lesion_amounts):
        forward = engine.model(torch.tensor([story_ids], device=device)).logits[0, -1].float().cpu()
    reference = engine.next_token_logits(story_ids, [mask_search.soft_intervention(components, k.tolist())])[0]
    difference = (forward - reference).abs().max().item()
    assert difference < MAX_BATCH_DIFFERENCE and forward.argmax() == reference.argmax(), difference
    print(f"4. soft_knockout forward = Intervention.soft: max difference {difference:.2e}", flush=True)

    # 5. Log-likelihoods: alone, in a padded batch, by hand.
    texts = ["Cinderella, um, she, the the ball. Prince. Shoe, glass shoe.", "Once upon a time there was a girl."]
    sequences = [story_ids + engine.tokenizer(" " + text, add_special_tokens=False)["input_ids"] for text in texts]
    with torch.no_grad():
        batch_sums, batch_counts = transcript_log_likelihoods(engine, len(story_ids), sequences)
        for row, sequence in enumerate(sequences):
            alone_sum, alone_count = transcript_log_likelihoods(engine, len(story_ids), [sequence])
            full_logits = engine.model(torch.tensor([sequence], device=device)).logits[0].float()
            by_hand = full_logits[:-1].log_softmax(-1)[torch.arange(len(sequence) - 1), torch.tensor(sequence[1:])]
            by_hand = by_hand[len(story_ids) - 1:].sum()
            assert alone_count == batch_counts[row] == len(sequence) - len(story_ids)
            for value in (alone_sum, by_hand):
                assert abs(float(value) - float(batch_sums[row])) < 1e-3, (row, float(value), float(batch_sums[row]))
    print(f"5. log-likelihoods alone = in a padded batch = by hand: {batch_sums.tolist()}", flush=True)

    # 6. Gradients in float64 against finite differences.
    engine64 = Engine(args.model, device, dtype=torch.float64)
    for parameter in engine64.model.parameters():
        parameter.requires_grad_(False)
    load_lesions(engine64, lesions)
    components64 = [("lesion", name) for name in lesions] + [("head", (layer, head))
                                                             for layer in range(engine64.geometry["n_layers"])
                                                             for head in range(engine64.geometry["n_heads"])]

    def log_likelihood(k):
        head_amounts, lesion_amounts = amounts_of(k, components64, engine64.geometry)
        with engine64.soft_knockout(head_amounts, lesion_amounts):
            return transcript_log_likelihoods(engine64, len(story_ids), sequences[:1])[0][0]

    k = torch.zeros(len(components64), device=device, dtype=torch.float64)
    probed = [0, 1] + [2 + layer * engine64.geometry["n_heads"] + head for layer, head in heads]
    k[probed] = torch.tensor([0.4, 0.2, 0.5, 0.1, 0.3], device=device, dtype=torch.float64)
    k.requires_grad_()
    gradient = torch.autograd.grad(log_likelihood(k), k)[0]
    epsilon = 1e-4
    for index in probed:
        with torch.no_grad():
            up, down = k.detach().clone(), k.detach().clone()
            up[index] += epsilon
            down[index] -= epsilon
            finite = (log_likelihood(up) - log_likelihood(down)) / (2 * epsilon)
        assert abs(float(gradient[index]) - float(finite)) <= 1e-4 * max(1.0, abs(float(finite))), \
            (components64[index], float(gradient[index]), float(finite))
        print(f"   d log p / d k[{components64[index]}] = {float(gradient[index]):+.6f} "
              f"(finite difference {float(finite):+.6f})", flush=True)
    assert all(parameter.grad is None for parameter in engine64.model.parameters())
    del engine64
    torch.cuda.empty_cache()
    print("6. gradients match finite differences in float64; no weight gets a gradient", flush=True)

    # 7. A short training run on the real transcripts, saved and resumed; the weights stay as they were.
    weights_before = {name: parameter.detach().clone() for name, parameter in engine.model.named_parameters()}
    mask_search.N_STAGES, mask_search.STEPS_PER_STAGE = 2, 3
    data = mask_search.prepare_data(engine, "cinderella", lambda message: print("   ", message, flush=True))
    components = space_components(engine, "2_heads")
    with tempfile.TemporaryDirectory() as folder:
        state_path = Path(folder) / "state.pt"
        first = mask_search.train(engine, data, components, "B", state_path, seed=1, log=print)
        assert state_path.exists() and len(first["checkpoints"]) == 2
        assert all(np.isfinite(c["train_loss"]) and np.isfinite(c["validation"]["loss"]) for c in first["checkpoints"])
        resumed = mask_search.train(engine, data, components, "B", state_path, seed=1, log=print)
        assert resumed["checkpoints"] == first["checkpoints"] and resumed["lambda_max"] == first["lambda_max"]
        # Stopped after stage 1, then resumed for stage 2, against the uninterrupted run above.
        mask_search.N_STAGES = 1
        stopped_path = Path(folder) / "stopped.pt"
        mask_search.train(engine, data, components, "B", stopped_path, seed=1, log=print)
        mask_search.N_STAGES = 2
        continued = mask_search.train(engine, data, components, "B", stopped_path, seed=1, log=print)
        difference = max(abs(a - b) for a, b in zip(continued["checkpoints"][1]["amounts"],
                                                       first["checkpoints"][1]["amounts"]))
        # NOTE: lambda_max is measured again in each run, and GPU
        # kernels add last-digit noise, so lambdas are compared to 1e-4 relative.
        lambdas = continued["checkpoints"][1]["lambda"], first["checkpoints"][1]["lambda"]
        assert difference < 1e-3 and abs(lambdas[0] - lambdas[1]) <= 1e-4 * lambdas[1], (difference, lambdas)
        print(f"   resumed mid-path: amounts within {difference:.1e}, lambdas {lambdas}", flush=True)
    for name, parameter in engine.model.named_parameters():
        assert parameter.grad is None and torch.equal(parameter, weights_before[name]), name
    print("7. training: finite losses, state saved, resumed without training again", flush=True)
    print("all checks passed", flush=True)


if __name__ == "__main__":
    main()
