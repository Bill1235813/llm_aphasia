"""Does a knockout circuit matter beyond its task? TAB and MMLU under circuit patching.

The circuit of one (model, task, objective), found by search.py and taken
apart by patching.py, is knocked out and patched on two external benchmarks,
under the conditions of patching.py:

    clean          no intervention
    corrupt        the whole circuit knocked out
    restore_<c>    corrupt, with component c's clean activations written back
                   at every prompt position
    restore_all    corrupt, with every component's clean activations written
                   back at every prompt position (patching.py's restore_all)

NOTE: the same restore means two different things on the two
benchmarks. MMLU is read out in one forward pass over the prompt, so every
position is a prompt position and c is restored everywhere. TAB generates,
and on the generated positions c stays knocked out (they have no clean
counterpart, engine.py), so c is restored only where the model reads the
question. A restore can therefore bring back a TAB answer but not an MMLU
score, or the other way round.

A circuit of more than MAX_RESTORED components restores only the
MAX_RESTORED whose leave-one-out value drops most in patching.py's results
(the components the circuit needs most), or its first MAX_RESTORED when that
file is missing.

TAB, the Text Aphasia Battery (Roll et al., 2025, arXiv:2511.20507): 20 items,
four subtests of five, ported with their scoring from the earlier
src/interp/exp07_patch_benchmarks.py:

    word comprehension      which of six named objects fits; the first option the answer names
    sentence comprehension  a yes/no question; the first "yes" or "no" in the answer
    repetition              repeat a word or a sentence; exact after normalising, and
                            leniently ("contains"), since chat models wrap the repetition
    connected text          3-5 sentences on a topic; scored by the five-way classifier

Chat models get exp07's prompts, a system and a user message in the Llama-2
chat template; base models get "Question: <system instruction> <prompt>\\nAnswer:",
and their answer is cut at the first question they write themselves, as in
search.py. Connected text gets 120 new tokens decoded with rep1.25, like the
narratives; the other subtests get 40, decoded plain greedy (DECODING).

MMLU global_facts (100 questions) and high_school_world_history (237): the
standard 5-shot prompt from each subject's dev split, read out without
generating. The predicted answer is the answer letter whose token has the
highest logit after "Answer:", and the log-probability of the correct letter
is taken over the full vocabulary. A prompt longer than the model's context
loses its earliest shots until it fits. The parquet files of cais/mmlu are
read from the local Hugging Face cache (HF_HOME) and never downloaded; they
were put there once with huggingface_hub.snapshot_download(repo_id="cais/mmlu",
repo_type="dataset", allow_patterns=["global_facts/*", "high_school_world_history/*"]).

--source em_search and --source grown_heads take those circuits instead, as in
patching.py, and write to em_search/benchmarks/ and grown_heads/benchmarks_patching/.

Output: RESULTS/benchmarks/<model>_<task>_<objective>.json, with every item's
texts (TAB) or answer-letter log-probabilities (MMLU) under every condition,
and a summary per condition, printed as tables at the end. The file is
rewritten after every item, and a rerun (a requeued job) skips the items it
already lists, as long as it lists the same conditions.

Run patching.py first: for a circuit of more than MAX_RESTORED components,
the restore conditions come from its results, and a rerun that finds them
after a run that did not starts over (its conditions changed).

Usage:
    bash slurm/fiveway_submit.sh bench-p1b-cind-A 40G fiveway/benchmarks.py --model pythia-1b --circuit-task cinderella --objective A
  TAB only, every grown-heads set of a model in one process, the model loaded once:
    bash slurm/fiveway_submit.sh bench-grown-p1b 40G fiveway/benchmarks.py --model pythia-1b \\
        --circuit-task cinderella sandwich --objective A B C --source grown_heads --benchmarks tab
  CPU smoke test with a circuit from the command line, writing outside results/ (about 15 minutes):
    GPUS=0 CPUS=8 bash slurm/fiveway_submit.sh bench-test 40G fiveway/benchmarks.py --model pythia-1b \\
        --circuit-task cinderella --objective A --heads 0:1 1:2 --limit 3 --output-dir runs/fiveway/benchmarks/test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.em_baseline import load_lesions  # noqa: E402
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.models import TASKS  # noqa: E402
from fiveway.patching import component_name, components_of, read_circuit, restore  # noqa: E402
from fiveway.paths import RESULTS, ROOT  # noqa: E402
from fiveway.prompts import chat_prompt, cut_at_new_question, long_enough  # noqa: E402
from fiveway.scorer import CLASSES, OBJECTIVES, FiveWayScorer, objective_values  # noqa: E402

# NOTE: connected text is decoded like the study's
# narratives, greedy with a repetition penalty of 1.25, so base models do not
# loop. The short-answer subtests are decoded plain greedy, as exp07 and the TAB
# paper did: the penalty acts on every token already in the context, prompt
# included, so it would push the model away from exactly the words the
# repetition subtest asks it to repeat and the options word comprehension asks
# it to name.
DECODING = {"connected_text": "rep1.25", "short_answer": "greedy"}
CONNECTED_TEXT_TOKENS = 120
SHORT_ANSWER_TOKENS = 40
MAX_RESTORED = 8
# The context window of each model family, in tokens.
CONTEXT_LENGTH = {"gpt2": 1024, "neox": 2048, "llama": 4096, "qwen3_5": 262144}
MMLU_SUBJECTS = ["global_facts", "high_school_world_history"]

# --- TAB items (arXiv:2511.20507v1, appendix), as in exp07 --------------------
# NOTE: exp07 transcribed the prompts from the arXiv HTML
# appendix (the paper links no repository). The paper gives the Connected
# Text system instruction verbatim; for the other three subtests it says only
# that a conversational system prompt is used, so exp07 chose a minimal one.
SYSTEM_CONNECTED = "Respond to the following prompt in 3-5 full sentences:"
SYSTEM_SHORT = "You are having a conversation. Answer the question directly and briefly."

TAB_ITEMS = [
    # connected text: open-ended
    *[{"subtest": "connected_text", "prompt": prompt, "system": SYSTEM_CONNECTED}
      for prompt in ["Tell me about the best trip you ever took.",
                     "Describe a happy childhood memory.",
                     "Tell me about your first job.",
                     "What do you like about where you live?",
                     "Describe the steps to make a simple meal."]],
    # word comprehension: the first option is the target, the other five are distractors
    *[{"subtest": "word_comprehension", "system": SYSTEM_SHORT,
       "prompt": f"{question} ({', '.join(options)})", "options": options, "target": options[0]}
      for question, options in [
          ("Which one is an animal with a mane?", ["lion", "drum", "violin", "giraffe", "boot", "boat"]),
          ("Which object is typically used to make music?", ["violin", "giraffe", "lion", "door", "boot", "boat"]),
          ("Which item is usually worn on the feet?", ["boot", "boat", "lion", "drum", "violin", "giraffe"]),
          ("Which object is used for cutting?", ["knife", "kite", "lion", "drum", "violin", "giraffe"]),
          ("Which one is a large mammal with a long neck?", ["giraffe", "horse", "lion", "drum", "violin", "boot"])]],
    # sentence comprehension: yes/no
    *[{"subtest": "sentence_comprehension", "system": SYSTEM_SHORT, "prompt": question, "target": answer}
      for question, answer in [("Are babies watched by babysitters?", "yes"),
                               ("Do you cut the grass with an axe?", "no"),
                               ("If you're about to leave, have you left yet?", "no"),
                               ("Are witnesses questioned by police?", "yes"),
                               ("If I was at the park when you arrived, did I get there first?", "yes")]],
    # repetition: exact reproduction
    *[{"subtest": "repetition", "system": SYSTEM_SHORT, "prompt": f"Please repeat exactly: {target}", "target": target}
      for target in ["house.", "breakfast.", "catastrophe.", "The sun rises in the East.",
                     "The ambitious journalist discovered where we'd be going."]],
]
SHORT_ANSWER_SUBTESTS = ["word_comprehension", "sentence_comprehension", "repetition"]


def normalize(text: str) -> str:
    """Lower case, outer quotes and final full stops or exclamation marks removed, whitespace collapsed."""
    text = text.lower().strip().strip("\"'“”‘’ ")
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".!").strip()


def score_short_answer(item: dict, response: str) -> dict:
    """exp07's scoring of a word comprehension, sentence comprehension or repetition answer."""
    lowered = response.lower()
    if item["subtest"] == "word_comprehension":
        # The first option the answer names is its choice.
        named = []
        for option in item["options"]:
            match = re.search(rf"\b{re.escape(option)}\b", lowered)
            if match:
                named.append((match.start(), option))
        choice = min(named)[1] if named else None
        return {"answer": choice, "correct": choice == item["target"]}
    if item["subtest"] == "sentence_comprehension":
        match = re.search(r"\b(yes|no)\b", lowered)
        answer = match.group(1) if match else None
        return {"answer": answer, "correct": answer == item["target"]}
    said, wanted = normalize(response), normalize(item["target"])
    return {"answer": response.strip(), "correct": said == wanted, "contains": wanted in said}


def tab_prompt(engine, item: dict) -> str:
    """The item as the model is asked it: its chat template (thinking off), or a Question/Answer pair for a base model."""
    if engine.spec.is_chat:
        return chat_prompt(engine.tokenizer, [{"role": "system", "content": item["system"]},
                                              {"role": "user", "content": item["prompt"]}])
    return f"Question: {item['system']} {item['prompt']}\nAnswer:"


# --- MMLU ---------------------------------------------------------------------

def load_mmlu(subject: str) -> tuple:
    """(test questions, dev questions) of one subject, read from the local Hugging Face cache."""
    splits = []
    for split in ("test", "dev"):
        path = hf_hub_download("cais/mmlu", f"{subject}/{split}-00000-of-00001.parquet", repo_type="dataset",
                               local_files_only=True)
        print(f"MMLU {subject} {split}: {path}", flush=True)
        splits.append(pd.read_parquet(path).to_dict("records"))
    return splits[0], splits[1]


def format_question(row: dict, with_answer: bool) -> str:
    text = row["question"].strip() + "\n"
    for letter, choice in zip("ABCD", row["choices"]):
        text += f"{letter}. {choice}\n"
    text += "Answer:"
    if with_answer:
        text += f" {'ABCD'[row['answer']]}\n\n"
    return text


def mmlu_prompt_ids(engine, subject: str, shots: list, row: dict) -> tuple:
    """Token ids of the 5-shot prompt of one question, and how many shots it kept.

    The earliest shots are dropped until the prompt fits the model's context:
    only GPT-2 (1024 tokens) needs it, for high_school_world_history, whose
    questions quote long passages.

    NOTE: a question too long to fit even without shots
    would return None here; the longest zero-shot prompt of these two
    subjects is 889 Llama tokens and 761 GPT-2 tokens, so it never happens.
    """
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    for n_shots in range(len(shots), -1, -1):
        examples = "".join(format_question(shot, True) for shot in shots[len(shots) - n_shots:])
        prompt_ids = engine.encode(header + examples + format_question(row, False))
        if len(prompt_ids) <= CONTEXT_LENGTH[engine.spec.family]:
            return prompt_ids, n_shots


def answer_letter_ids(tokenizer) -> list:
    """The token id of each answer letter as it follows "Answer:": " A" for GPT-2 and Pythia, "▁A" for Llama."""
    letter_ids = []
    for letter in "ABCD":
        with_letter = tokenizer("Answer: " + letter, add_special_tokens=False)["input_ids"]
        # The letter must be exactly one more token after those of "Answer:".
        assert with_letter[:-1] == tokenizer("Answer:", add_special_tokens=False)["input_ids"]
        letter_ids.append(with_letter[-1])
    return letter_ids


# --- summary --------------------------------------------------------------------

def summarize(results: dict) -> dict:
    """Per condition: TAB scores per subtest, and MMLU accuracy and mean log-probability per subject."""
    summary = {"tab": {}, "mmlu": {}}
    tab_names = list(results["conditions"]) if results["tab"] else []
    for name in tab_names:
        row = {}
        for subtest in SHORT_ANSWER_SUBTESTS:
            records = [item["responses"][name] for item in results["tab"] if item["subtest"] == subtest]
            row[subtest] = {"correct": sum(record["correct"] for record in records), "n": len(records)}
            if subtest == "repetition":
                row[subtest]["contains"] = sum(record["contains"] for record in records)
        records = [item["responses"][name] for item in results["tab"] if item["subtest"] == "connected_text"]
        scored = [record for record in records if record["probabilities"] is not None]
        row["connected_text"] = {
            "n": len(records), "n_scored": len(scored),
            "mean_probabilities": (np.mean([record["probabilities"] for record in scored], axis=0).round(6).tolist()
                                   if scored else None),
            "mean_objectives": ({objective: round(float(np.mean([record["objectives"][objective] for record in scored])), 6)
                                 for objective in OBJECTIVES} if scored else None),
        }
        summary["tab"][name] = row
    for subject in MMLU_SUBJECTS:
        items = [item for item in results["mmlu"] if item["subject"] == subject]
        if not items:
            continue
        summary["mmlu"][subject] = {}
        for name in results["conditions"]:
            summary["mmlu"][subject][name] = {
                "n": len(items),
                "accuracy": round(float(np.mean([item["conditions"][name]["correct"] for item in items])), 6),
                "mean_log_probability_correct": round(float(np.mean(
                    [item["conditions"][name]["log_probabilities"][item["answer"]] for item in items])), 6),
            }
    return summary


def print_tables(results: dict):
    summary = results["summary"]
    print(f"\n{results['config']['model']}: circuit of {results['config']['circuit_task']} objective "
          f"{results['config']['objective']}, {results['circuit']['heads']} heads, "
          f"{results['circuit']['mlp_blocks']} MLP blocks", flush=True)
    if summary["tab"]:
        print("\nTAB (correct / items; connected text: mean five-way probabilities of the scored answers)", flush=True)
        print(f"{'condition':<22}{'word':>7}{'sentence':>10}{'repeat':>8}{'contains':>10}{'scored':>8}"
              + "".join(f"{name[:10]:>11}" for name in CLASSES), flush=True)
    for name, row in summary["tab"].items():
        cells = [f"{row[subtest]['correct']}/{row[subtest]['n']}" for subtest in SHORT_ANSWER_SUBTESTS]
        cells.append(f"{row['repetition']['contains']}/{row['repetition']['n']}")
        connected = row["connected_text"]
        cells.append(f"{connected['n_scored']}/{connected['n']}")
        means = connected["mean_probabilities"] or [None] * len(CLASSES)
        print(f"{name:<22}{cells[0]:>7}{cells[1]:>10}{cells[2]:>8}{cells[3]:>10}{cells[4]:>8}"
              + "".join("          -" if probability is None else f"{probability:>11.3f}" for probability in means),
              flush=True)
    for subject, rows in summary["mmlu"].items():
        print(f"\nMMLU {subject} ({next(iter(rows.values()))['n']} questions)", flush=True)
        print(f"{'condition':<22}{'accuracy':>10}{'mean log P(correct)':>22}", flush=True)
        for name, row in rows.items():
            print(f"{name:<22}{row['accuracy']:>10.3f}{row['mean_log_probability_correct']:>22.3f}", flush=True)


def run_circuit(args, circuit_task: str, objective: str, loaded: dict, log) -> None:
    """TAB and MMLU under one circuit's conditions, resumed from and written to its results file.

    ``loaded`` holds the engine and the scorer once a circuit has needed them,
    so the circuits of several tasks and objectives share one model in memory.
    """
    circuit = read_circuit(args.model, circuit_task, objective, args.heads, args.mlp_blocks, args.source)
    if circuit is None:
        log(f"the search found no circuit for {args.model} {circuit_task} objective {objective}: nothing to patch")
        return

    # --- which components to restore ---------------------------------------
    components = components_of(circuit)
    restored, selection = components, "every component"
    if len(components) > MAX_RESTORED:
        # NOTE: ROOT / path is path itself when path is absolute.
        patching_path = ROOT / args.patching_dir / f"{args.model}_{circuit_task}_{objective}.json"
        if patching_path.exists():
            patching = json.loads(patching_path.read_text())
            assert patching["circuit"]["heads"] == [list(pair) for pair in circuit["heads"]] and \
                patching["circuit"]["mlp_blocks"] == [list(pair) for pair in circuit["mlp_blocks"]] and \
                patching["circuit"].get("lesions", []) == circuit["lesions"], f"{patching_path} patched another circuit"
            drops = {entry["name"]: entry["leave_one_out_drop"] for entry in patching["components"]}

            def drop(component):
                # NOTE: a component whose leave-one-out
                # answer does not count has no drop (None) and ranks last.
                value = drops[component_name(*component)]
                return -np.inf if value is None else value

            restored = sorted(components, key=drop, reverse=True)[:MAX_RESTORED]
            selection = f"the {MAX_RESTORED} largest leave-one-out drops in {patching_path}"
        else:
            restored = components[:MAX_RESTORED]
            selection = f"the first {MAX_RESTORED} ({patching_path} is missing)"

    corrupt = Intervention.knockout(heads=circuit["heads"], mlp_blocks=circuit["mlp_blocks"],
                                    lesions=circuit["lesions"])
    conditions = {"clean": Intervention(), "corrupt": corrupt}
    for kind, pair in restored:
        conditions[f"restore_{component_name(kind, pair)}"] = restore(corrupt, kind, pair)
    # Every component's clean activations written back at the prompt positions, as patching.py's restore_all:
    # on TAB the circuit is then knocked out only while the answer is written; on MMLU, read out in one pass
    # over the prompt, it is the clean model.
    conditions["restore_all"] = dataclasses.replace(corrupt, patch_heads=tuple(circuit["heads"]),
                                                    patch_blocks=tuple(circuit["mlp_blocks"]),
                                                    patch_lesions=tuple(circuit["lesions"]))
    names = list(conditions)
    interventions = list(conditions.values())
    log(f"circuit from {circuit['source']}: {[component_name(*component) for component in components]}; "
        f"restoring {selection}; conditions {names}")

    # --- resume ----------------------------------------------------------------
    out_path = ROOT / args.output_dir / f"{args.model}_{circuit_task}_{objective}.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # NOTE: a preempted job can be requeued on another kind
    # of GPU, whose arithmetic can turn a near-tie in the logits the other
    # way, so every item records the device it was made on; the config holds
    # the latest run's.
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    results = {
        "config": {"model": args.model, "circuit_task": circuit_task, "objective": objective,
                   "decoding": DECODING, "connected_text_tokens": CONNECTED_TEXT_TOKENS,
                   "short_answer_tokens": SHORT_ANSWER_TOKENS, "device": str(device), "device_name": device_name},
        "circuit": {**circuit, "heads": [list(pair) for pair in circuit["heads"]],
                    "mlp_blocks": [list(pair) for pair in circuit["mlp_blocks"]]},
        "restored": [component_name(*component) for component in restored], "restore_selection": selection,
        "conditions": {name: intervention.key() for name, intervention in conditions.items()},
        "tab": [], "mmlu": [], "summary": {},
    }
    # NOTE: the stored items are reused only when they were
    # made under exactly the same conditions (the same intervention keys);
    # otherwise the file belongs to another circuit and is started afresh.
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous["conditions"] == results["conditions"]:
            results["tab"], results["mmlu"] = previous["tab"], previous["mmlu"]
    done_tab = {item["item"] for item in results["tab"]}
    done_mmlu = {(item["subject"], item["item"]) for item in results["mmlu"]}
    log(f"{out_path}: {len(done_tab)} TAB items and {len(done_mmlu)} MMLU questions already done")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        results["summary"] = summarize(results)
        # NOTE: writing a temporary file and renaming it over
        # the old one means a job preempted mid-write never leaves a cut-off
        # file behind, which the rerun could not read.
        temporary = out_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=1))
        temporary.replace(out_path)

    if "engine" not in loaded:
        loaded["engine"] = Engine(args.model, device)
    engine = loaded["engine"]
    if circuit["lesions"]:
        load_lesions(engine, circuit["lesions"])
    log(f"{args.model} on {device}: batches of {engine.batch_size}")

    # --- TAB ---------------------------------------------------------------------
    if "tab" in args.benchmarks:
        if "scorer" not in loaded:
            loaded["scorer"] = FiveWayScorer(device)
        scorer = loaded["scorer"]
        selected = []
        for subtest in ["connected_text"] + SHORT_ANSWER_SUBTESTS:
            item_ids = [item_id for item_id, item in enumerate(TAB_ITEMS) if item["subtest"] == subtest]
            selected += item_ids[:args.limit]
        for item_id in selected:
            if item_id in done_tab:
                continue
            item = TAB_ITEMS[item_id]
            prompt = tab_prompt(engine, item)
            is_connected = item["subtest"] == "connected_text"
            full_texts = engine.generate(engine.encode(prompt), interventions,
                                         CONNECTED_TEXT_TOKENS if is_connected else SHORT_ANSWER_TOKENS,
                                         DECODING["connected_text" if is_connected else "short_answer"])
            responses = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
            entry = {"item": item_id, **item, "rendered_prompt": prompt, "device_name": device_name, "responses": {}}
            if is_connected:
                scored = [response for response in responses if long_enough(response)]
                # NOTE: probabilities: (n_scored, 5); objectives: {"A" | "B" | "C": (n_scored,)}.
                probabilities = scorer.probabilities(scored)
                objectives = objective_values(probabilities)
                row = 0
                for name, full_text, response in zip(names, full_texts, responses):
                    record = {"response": response, "full_text": full_text, "probabilities": None, "objectives": None}
                    if long_enough(response):
                        record["probabilities"] = probabilities[row].round(6).tolist()
                        record["objectives"] = {objective: round(float(values[row]), 6)
                                                for objective, values in objectives.items()}
                        row += 1
                    entry["responses"][name] = record
            else:
                for name, full_text, response in zip(names, full_texts, responses):
                    entry["responses"][name] = {"response": response, "full_text": full_text,
                                                **score_short_answer(item, response)}
            results["tab"].append(entry)
            save()
            shown = {name: (record["objectives"][objective] if is_connected and record["objectives"]
                            else record.get("correct")) for name, record in entry["responses"].items()}
            log(f"TAB {item_id:2d} {item['subtest']}: {shown}\n    clean   {entry['responses']['clean']['response'][:90]!r}"
                f"\n    corrupt {entry['responses']['corrupt']['response'][:90]!r}")

    # --- MMLU ----------------------------------------------------------------------
    if "mmlu" in args.benchmarks:
        # NOTE: a Hugging Face forward pass keeps every
        # layer's keys and values (the KV cache that generation reuses) unless
        # the model's config turns it off, and next_token_logits makes a plain
        # forward pass. For one readout pass that cache is pure waste:
        # 2 x n_layers x batch_size x prompt_len x d_model floats, 71 GB for
        # Llama-2-7B and 138 GB for Llama-2-13B at the longest world-history
        # prompt (2,104 tokens), more than the weights themselves. Turned off,
        # the logits come out bit for bit the same.
        # NOTE: generation must keep the cache (generate()
        # takes it from its own config, so TAB above is unaffected): without
        # it every step reruns the whole sequence, and the engine's patch,
        # which fires on the pass as long as the prompt, would fire on the
        # first step only.
        use_cache = engine.model.config.use_cache
        engine.model.config.use_cache = False
        letter_ids = answer_letter_ids(engine.tokenizer)
        log(f"answer letter tokens {letter_ids} = {engine.tokenizer.convert_ids_to_tokens(letter_ids)}")
        for subject in MMLU_SUBJECTS:
            questions, shots = load_mmlu(subject)
            for question_id, row in enumerate(questions[:args.limit]):
                if (subject, question_id) in done_mmlu:
                    continue
                prompt_ids, n_shots = mmlu_prompt_ids(engine, subject, shots, row)
                # NOTE: logits: (n_conditions, vocab), the next-token
                # logits after the prompt's last token ("Answer:"), one row per
                # condition. log_softmax over the vocabulary, then the four
                # letter columns: (n_conditions, 4).
                # NOTE: the readout computes the last
                # position's logits only, and reads the conditions without the
                # padding copies that generation needs for exact reproducibility
                # (engine.py): with about ten conditions and batches of 32,
                # padding would triple the cost of every MMLU question.
                logits = engine.next_token_logits(prompt_ids, interventions, pad_batches=False)
                letter_log_probabilities = torch.log_softmax(logits, dim=-1)[:, letter_ids]
                # NOTE: the prediction compares the four letters
                # only, as the standard MMLU readout does, while the reported
                # log-probability is over the full vocabulary: it also falls
                # when the knockout moves probability away from the letters
                # altogether (to " The", say), which the accuracy cannot see.
                answer = int(row["answer"])
                entry = {"subject": subject, "item": question_id, "answer": answer, "n_shots": n_shots,
                         "prompt_tokens": len(prompt_ids), "device_name": device_name, "conditions": {}}
                for name, log_probabilities in zip(names, letter_log_probabilities.tolist()):
                    prediction = int(np.argmax(log_probabilities))
                    entry["conditions"][name] = {"prediction": prediction, "correct": prediction == answer,
                                                 "log_probabilities": [round(value, 5) for value in log_probabilities]}
                results["mmlu"].append(entry)
                save()
                if (question_id + 1) % 25 == 0 or question_id + 1 == len(questions[:args.limit]):
                    accuracy = {name: values["accuracy"] for name, values in results["summary"]["mmlu"][subject].items()}
                    log(f"MMLU {subject} {question_id + 1}/{len(questions[:args.limit])} ({n_shots} shots, "
                        f"{len(prompt_ids)} tokens): accuracy so far {accuracy}")
        # The next circuit's TAB generates with the same model, so the cache setting goes back as it was.
        engine.model.config.use_cache = use_cache

    save()
    log(f"wrote {out_path}")
    print_tables(results)


# Where each source's circuits are patched (patching.py) and benchmarked (this script).
FOLDERS = {"search": (RESULTS / "patching", RESULTS / "benchmarks"),
           "em_search": (RESULTS / "em_search" / "patching", RESULTS / "em_search" / "benchmarks"),
           # NOTE: not grown_heads/benchmarks, which holds grown_heads.py's own TAB and MMLU
           # checks of the same sets in another format.
           "grown_heads": (RESULTS / "grown_heads" / "patching", RESULTS / "grown_heads" / "benchmarks_patching")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--circuit-task", required=True, nargs="+", choices=TASKS,
                        help="the task whose circuit is used, or both: every task and objective given is run in "
                             "turn, the model loaded once")
    parser.add_argument("--objective", required=True, nargs="+", choices=sorted(OBJECTIVES))
    parser.add_argument("--heads", nargs="*", default=[],
                        help="layer:head pairs that replace the search's circuit (for tests)")
    parser.add_argument("--mlp-blocks", nargs="*", default=[],
                        help="layer:block pairs of 256-neuron MLP blocks, likewise")
    parser.add_argument("--benchmarks", nargs="+", default=["tab", "mmlu"], choices=["tab", "mmlu"])
    parser.add_argument("--limit", type=int, default=None, help="only the first N items of each subtest and subject")
    parser.add_argument("--source", default="search", choices=list(FOLDERS),
                        help="whose circuit: search.py's, em_search.py's or the grown-heads sets")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="where the results file goes (default: the source's benchmarks folder in FOLDERS); "
                             "a test points it outside results/")
    parser.add_argument("--patching-dir", type=Path, default=None,
                        help="patching.py's results, which rank the components of a large circuit "
                             "(default: the source's patching folder in FOLDERS)")
    args = parser.parse_args()
    args.patching_dir = args.patching_dir or FOLDERS[args.source][0]
    args.output_dir = args.output_dir or FOLDERS[args.source][1]
    start = time.time()

    def log(message):
        print(f"[{(time.time() - start) / 60:6.1f} min] {message}", flush=True)

    loaded = {}
    for circuit_task in args.circuit_task:
        for objective in args.objective:
            run_circuit(args, circuit_task, objective, loaded, log)


if __name__ == "__main__":
    main()
