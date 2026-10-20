"""TAB judge prompt: the 19 aphasia features of connected text (Roll et al., 2025).

src/fiveway/llm_judge.py sends this prompt to DeepSeek-V4.1-Flash (key: DEEPSEEK_API_KEY):
the paper's system instruction, a definition and example for each feature, and two
annotated example transcripts. main() below is an earlier standalone run with Gemini
2.5 Flash through OpenRouter, on outputs of an earlier pipeline that are not included.

NOTE: the paper's repository (which holds its exact
definitions and annotated examples) is not linked from the arXiv text, so the
definitions and the two examples below are our own reconstruction from the
feature names and standard aphasiology usage. The two example transcripts are
synthetic — no AphasiaBank participant text is sent to the API.

The API key is read from the environment variable OPENROUTER_API_KEY, or
else from the file ~/.openrouter_api_key (one line). It is never printed.

    python src/interp/tab_judge.py --estimate     # count tokens, print cost, send nothing
    python src/interp/tab_judge.py --limit 3      # smoke test on 3 transcripts
    python src/interp/tab_judge.py                # judge everything not yet judged

Results are cached incrementally in data/interp/exp07_tab_judge_llama2.json,
so re-running only judges transcripts that have no verdict yet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "interp"
IN_JSON = DATA_DIR / "exp07_patch_benchmarks_llama2.json"
OUT_JSON = DATA_DIR / "exp07_tab_judge_llama2.json"

MODEL = "google/gemini-2.5-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
PRICE_IN, PRICE_OUT = 0.30, 2.50          # USD per 1M tokens (OpenRouter, Gemini 2.5 Flash)

FEATURES = [
    ("anomia", "Word-finding difficulty: pauses, circumlocutions, vague fillers ('thing', 'stuff'), or an explicit failure to retrieve a word.",
     "\"we went to the... the place with the, um, the water thing\""),
    ("abandoned_utterances", "An utterance is started and given up before it is completed.",
     "\"and then we drove to the—. It was a nice day.\""),
    ("empty_speech", "Fluent output that carries little information: generic words and vague statements in place of content.",
     "\"It was good. We did things there. It was really something, you know.\""),
    ("semantic_paraphasias", "A real word substituted for the intended one from a related meaning (fork for spoon).",
     "\"I poured the milk into the fork\""),
    ("phonemic_paraphasias", "Sound-level errors that produce a near-word: substituted, added, or transposed sounds.",
     "\"we watched the tevelision\""),
    ("neologisms", "Invented non-words used as if they were words.",
     "\"she brought the flimber to the table\""),
    ("jargon", "Fluent but largely incomprehensible speech made of neologisms and paraphasias.",
     "\"the plorbing came to the sandwitcher and it was gasping\""),
    ("perseverations", "Inappropriate repetition of a word, phrase or idea already produced, carried over where it no longer fits.",
     "\"I like the beach. The beach is nice. My job was at the beach. The beach.\""),
    ("stereotypies_and_automatisms", "Recurring stock phrases or automatic sequences used instead of content ('you know', 'and so on', counting).",
     "\"you know, you know, and so on, you know\""),
    ("short_simplified_utterances", "Telegraphic output: very short utterances with reduced syntax.",
     "\"Trip. Beach. Nice. Sun.\""),
    ("omission_of_bound_morphemes", "Missing inflections such as -ed, -s, -ing.",
     "\"yesterday I walk to the store and buy two apple\""),
    ("omission_of_function_words", "Missing articles, prepositions, auxiliaries, or pronouns.",
     "\"went store. bought bread. came home.\""),
    ("paragrammatism", "Grammatical errors of commission in otherwise fluent speech: wrong tense, agreement, or word order.",
     "\"the trip were going by us to have had yesterday\""),
    ("false_starts", "An utterance begins, stops, and restarts in a different form.",
     "\"My first— when I first— the job I had first was...\""),
    ("retracing", "Words or phrases repeated in order to self-correct or to continue.",
     "\"we went to the, we went to the beach\""),
    ("conduite_d_approche", "Successive approximations toward a target word.",
     "\"cat- catas- catastro- catastrophe\""),
    ("meaning_unclear", "The intended message cannot be recovered from the utterance.",
     "\"it was the one for the when they had it over\""),
    ("off_topic", "The response does not address the prompt.",
     "(asked about a trip) \"The stock market has been volatile recently.\""),
    ("overall_communication_impairment", "Global judgment: taken as a whole, communication is impaired.",
     "(any transcript where the features above materially reduce communicative success)"),
]

SYSTEM_PROMPT = (
    "You are an automated system for analyzing transcripts of speech for aphasic "
    "features. Your task is to read a transcript passage and determine whether each "
    "of the following features is present (1) or not present (0). Your output must "
    "be a JSON file with each feature as a key and its value as either 0 or 1.")

EXAMPLE_CONTROL = (
    "Prompt: Tell me about your first job.\n"
    "Transcript: My first job was at a small bakery near my school. I started at five in the "
    "morning, helped shape the dough, and served customers once we opened. It was tiring but "
    "I loved the smell of fresh bread. I stayed there for two summers before going to college.")
EXAMPLE_APHASIC = (
    "Prompt: Tell me about your first job.\n"
    "Transcript: Job... um, first job. Bakery. Bakery, no, the— the bread place. Five o'clock, "
    "morning. Make the... the thing, the dough. Dough. People come, I give bread. Two summer. "
    "Tired but good. Bread smell good, good.")
EXAMPLE_APHASIC_LABELS = {f: 0 for f, _, _ in FEATURES}
EXAMPLE_APHASIC_LABELS.update({"anomia": 1, "short_simplified_utterances": 1,
                               "omission_of_bound_morphemes": 1, "omission_of_function_words": 1,
                               "false_starts": 1, "retracing": 1, "perseverations": 1,
                               "overall_communication_impairment": 1})


def build_messages(prompt_text: str, transcript: str):
    definitions = "\n".join(f"- {name}: {definition} Example: {example}"
                            for name, definition, example in FEATURES)
    user = (
        "Features to score:\n" + definitions + "\n\n"
        "Annotated example 1:\n" + EXAMPLE_CONTROL + "\nLabels: "
        + json.dumps({f: 0 for f, _, _ in FEATURES}) + "\n\n"
        "Annotated example 2:\n" + EXAMPLE_APHASIC + "\nLabels: "
        + json.dumps(EXAMPLE_APHASIC_LABELS) + "\n\n"
        "Now score this transcript. Output only the JSON object with all 19 keys.\n"
        f"Prompt: {prompt_text}\nTranscript: {transcript}")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def approx_tokens(messages) -> int:
    # NOTE: a rough 4-characters-per-token estimate; the
    # exact count comes back in the API's usage field after each real call.
    return sum(len(m["content"]) for m in messages) // 4


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        path = Path.home() / ".openrouter_api_key"
        if path.exists():
            key = path.read_text().strip()
    assert key, ("no API key: export OPENROUTER_API_KEY=... or put the key on one line in "
                 "~/.openrouter_api_key (chmod 600)")
    return key


def judge(messages, key: str) -> (dict, dict):
    body = {"model": MODEL, "messages": messages, "temperature": 0,
            "response_format": {"type": "json_object"}}
    for attempt in range(6):
        # NOTE: transient network failures raise instead of
        # returning a status code, so they need their own retry path.
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                              json=body, timeout=120)
        except requests.exceptions.RequestException as e:
            print(f"    network error ({type(e).__name__}); retry {attempt + 1}/6", flush=True)
            time.sleep(5 * 2 ** attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", content, re.DOTALL)
            labels = json.loads(match.group(0)) if match else {}
            labels = {f: int(labels.get(f, 0)) for f, _, _ in FEATURES}
            return labels, data.get("usage", {})
        time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenRouter request failed: {r.status_code} {r.text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="count tokens and print the cost; send nothing")
    parser.add_argument("--limit", type=int, default=None, help="judge at most this many (smoke test)")
    args = parser.parse_args()

    records = [r for r in json.loads(IN_JSON.read_text())["tab"] if r["subtest"] == "connected_text"]
    done = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    key_of = lambda r: f"item{r['item']}|{r['condition']}|s{r['sample']}"
    todo = [r for r in records if key_of(r) not in done]
    print(f"{len(records)} connected-text transcripts, {len(done)} already judged, {len(todo)} to do")

    if args.estimate:
        tokens_in = sum(approx_tokens(build_messages(r["prompt"], r["response"])) for r in todo)
        tokens_out = 130 * len(todo)                # 19-key JSON is ~130 tokens
        cost = tokens_in * PRICE_IN / 1e6 + tokens_out * PRICE_OUT / 1e6
        print(f"estimated input tokens : {tokens_in:,}  ({tokens_in / max(len(todo), 1):.0f} per call)")
        print(f"estimated output tokens: {tokens_out:,}")
        print(f"estimated cost         : ${cost:.4f}  at ${PRICE_IN}/${PRICE_OUT} per 1M in/out")
        return

    key = load_key()
    if args.limit:
        todo = todo[:args.limit]
    used_in = used_out = 0
    for n, r in enumerate(todo, 1):
        labels, usage = judge(build_messages(r["prompt"], r["response"]), key)
        used_in += usage.get("prompt_tokens", 0)
        used_out += usage.get("completion_tokens", 0)
        done[key_of(r)] = {"item": r["item"], "condition": r["condition"], "sample": r["sample"],
                           "labels": labels, "n_features": sum(labels.values())}
        OUT_JSON.write_text(json.dumps(done, indent=2))
        print(f"[{n}/{len(todo)}] {key_of(r):<28} features present = {sum(labels.values()):2d}  "
              f"overall_impairment = {labels['overall_communication_impairment']}", flush=True)
        time.sleep(0.5)
    cost = used_in * PRICE_IN / 1e6 + used_out * PRICE_OUT / 1e6
    print(f"\nactual usage: {used_in:,} in / {used_out:,} out tokens  ->  ${cost:.4f}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
