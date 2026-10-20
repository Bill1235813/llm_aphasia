"""CPU checks of patching.py and benchmarks.py that need no model weights, only tokenizers.

1. The TAB port. benchmarks.py's 20 items are the prompts and targets that the
   earlier src/interp/exp07_patch_benchmarks.py stored with its results
   (data/interp/exp07_patch_benchmarks_llama2.json), and benchmarks.py's
   scoring gives exp07's answer, correct and contains on every stored
   Llama-2-7B-chat response.
2. The conditions. restore is the corrupt knockout plus one patched
   component; leave-one-out and alone knock out what their names say.
3. The MMLU prompts. For each model family, the answer letters are single
   tokens after "Answer:", and every prompt of both subjects fits the
   context window once the earliest shots are dropped.
4. When a Llama-2-7B-chat benchmarks file exists (written by a GPU run), its
   clean MMLU readout is compared with exp07's. exp07 renormalised the
   probabilities of the four letters, so ours are renormalised the same way
   before the comparison.

Usage (in the shell; nothing here runs a model):
    CUDA_VISIBLE_DEVICES= .venv/bin/python src/fiveway/test_patching_benchmarks.py
"""

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway import benchmarks, patching  # noqa: E402
from fiveway.engine import Intervention  # noqa: E402
from fiveway.models import MODELS  # noqa: E402
from fiveway.paths import DATA, RESULTS  # noqa: E402


def main():
    exp07 = json.loads((DATA / "interp" / "exp07_patch_benchmarks_llama2.json").read_text())

    # --- 1. the TAB port ----------------------------------------------------
    stored_items = {}
    for record in exp07["tab"]:
        stored_items.setdefault(record["item"], (record["subtest"], record["prompt"], record["target"]))
    ported_items = {item_id: (item["subtest"], item["prompt"], item.get("target"))
                    for item_id, item in enumerate(benchmarks.TAB_ITEMS)}
    assert ported_items == stored_items, "the TAB items differ from exp07's"
    n_compared = 0
    for record in exp07["tab"]:
        if record["subtest"] == "connected_text":
            continue
        scores = benchmarks.score_short_answer(benchmarks.TAB_ITEMS[record["item"]], record["response"])
        for key, value in scores.items():
            assert record[key] == value, f"item {record['item']} {key}: exp07 {record[key]!r}, ours {value!r}"
            n_compared += 1
    print(f"1. TAB: the 20 items are exp07's, and {n_compared} stored scores are reproduced", flush=True)

    # --- 2. the conditions --------------------------------------------------
    circuit = {"heads": [(3, 2), (3, 5)], "mlp_blocks": [(1, 4)]}
    corrupt = Intervention.knockout(heads=circuit["heads"], mlp_blocks=circuit["mlp_blocks"])
    components = patching.components_of(circuit)
    names = [patching.component_name(kind, pair) for kind, pair in components]
    assert names == ["L3H2", "L3H5", "L1MLP4"]
    restored = [patching.restore(corrupt, kind, pair) for kind, pair in components]
    assert [intervention.key() for intervention in restored] == [
        "H:L3H2,L3H5|M:L1B4|P:L3H2", "H:L3H2,L3H5|M:L1B4|P:L3H5", "H:L3H2,L3H5|M:L1B4|PB:L1B4"]
    assert [patching.leave_out(circuit, kind, pair).key() for kind, pair in components] == [
        "H:L3H5|M:L1B4", "H:L3H2|M:L1B4", "H:L3H2,L3H5"]
    assert [patching.alone(kind, pair).key() for kind, pair in components] == ["H:L3H2", "H:L3H5", "M:L1B4"]
    print("2. conditions: restore, leave-one-out and alone are the interventions they should be", flush=True)

    # --- 3. the MMLU prompts ------------------------------------------------
    subjects = {subject: benchmarks.load_mmlu(subject) for subject in benchmarks.MMLU_SUBJECTS}
    for model_key in ["gpt2-xl", "pythia-1b", "llama2-7b-chat"]:
        spec = MODELS[model_key]
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

        # NOTE: a stand-in for Engine with only what the
        # prompt functions use, so no weights are loaded; encode() is
        # Engine.encode's rule (the beginning-of-sequence token in front for
        # GPT-2 and Llama, nothing for Pythia).
        def encode(prompt, tokenizer=tokenizer, spec=spec):
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            return [tokenizer.bos_token_id] + ids if spec.prepend_bos else ids

        engine = SimpleNamespace(tokenizer=tokenizer, spec=spec, encode=encode)
        letter_ids = benchmarks.answer_letter_ids(tokenizer)
        assert [tokenizer.decode([letter_id]).strip() for letter_id in letter_ids] == list("ABCD")
        report = []
        for subject, (questions, shots) in subjects.items():
            shot_counts, longest = Counter(), 0
            for row in questions:
                prompt_ids, n_shots = benchmarks.mmlu_prompt_ids(engine, subject, shots, row)
                assert prompt_ids[-1] == tokenizer("Answer:", add_special_tokens=False)["input_ids"][-1]
                shot_counts[n_shots] += 1
                longest = max(longest, len(prompt_ids))
            assert longest <= benchmarks.CONTEXT_LENGTH[spec.family]
            report.append(f"{subject}: shots {dict(sorted(shot_counts.items()))}, longest {longest} tokens")
        print(f"3. {model_key}: letters {tokenizer.convert_ids_to_tokens(letter_ids)}; " + "; ".join(report),
              flush=True)

    # --- 4. the clean MMLU readout against exp07 (after a GPU run) -----------
    results_files = sorted((RESULTS / "benchmarks").glob("llama2-7b-chat_*.json"))
    if not results_files:
        print("4. no llama2-7b-chat benchmarks file yet: the comparison with exp07 is skipped", flush=True)
        return
    ours = json.loads(results_files[0].read_text())
    theirs = {(record["subset"], record["item"]): record for record in exp07["mmlu"] if record["condition"] == "clean"}
    differences, agreements = [], []
    for item in ours["mmlu"]:
        log_probabilities = np.array(item["conditions"]["clean"]["log_probabilities"])
        # NOTE: subtracting the log of the summed probabilities
        # (logsumexp) renormalises four log-probabilities to sum to one.
        renormalised = log_probabilities - np.logaddexp.reduce(log_probabilities)
        stored = theirs[(item["subject"], item["item"])]
        differences.append(np.abs(renormalised - np.array(stored["logps"])).max())
        agreements.append(int(np.argmax(log_probabilities)) == stored["pred"])
    print(f"4. {results_files[0].name}: clean MMLU vs exp07 on {len(differences)} questions: predictions agree on "
          f"{np.mean(agreements):.3f}, letter log-probabilities differ by {np.median(differences):.4f} (median) "
          f"and {np.max(differences):.4f} (max)", flush=True)


if __name__ == "__main__":
    main()
