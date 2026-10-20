"""An LLM judge (DeepSeek-V4.1-Flash) next to the five-way classifier.

Three questions are put to the judge, each one model answer at a time:

    binary   "Does the speaker of this answer have aphasia?"  yes or no
    fiveway  which of the classifier's five classes describes the speaker:
             Control, anomic or conduction aphasia, Broca's aphasia,
             Wernicke's aphasia, or not human speech (the classifier's
             Model_scrambles)
    tab      the Text Aphasia Battery's 19 aphasia features of a connected-text
             answer, with the prompt of src/interp/tab_judge.py (the TAB paper's
             protocol: feature definitions and two synthetic annotated examples)

NOTE: the binary and five-way prompts give the class names
and nothing else: no definitions and no examples. The classifier learned the
classes from data, and a judge handed hand-written descriptions or examples
would be judging partly by those instead; the plain version asks what the
model itself knows about aphasia.

Every request is cached by its full content in runs/fiveway/llm_judge/cache.jsonl,
so a rerun asks nothing twice. The key is read from DEEPSEEK_API_KEY or else
from ~/.deepseek_api_key (one line, chmod 600), and never printed. Only model
answers and task prompts are sent; no AphasiaBank text.

Usage (the TAB judge over every benchmarks file, results/fiveway/benchmarks,
results/fiveway/em_search/benchmarks and results/fiveway/em_attention/benchmarks; --circuits: the binary and five-way
judge on the answer of every circuit of search.py and em_search.py, with a
compression ratio that flags self-repeating strings; the annotation sets use
the judge through annotation_csv.py):
    GPUS=0 bash slurm/fiveway_submit.sh tab-judge 8G fiveway/llm_judge.py --tab --circuits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.paths import RESULTS, RUNS  # noqa: E402
from interp.generate import LLAMA2_STORY_USER, SANDWICH_USER  # noqa: E402
from interp.tab_judge import FEATURES, build_messages  # noqa: E402

API_URL = "https://api.deepseek.com/chat/completions"
JUDGE_MODEL = "deepseek-flash"      # DeepSeek-V4.1-Flash
CACHE_PATH = RUNS / "fiveway" / "llm_judge" / "cache.jsonl"
WORKERS = 8                         # requests in flight at once

BINARY_PROMPT = """Here is a question and a speaker's answer to it, transcribed.

Question: {question}
Answer: {answer}

Does the speaker have aphasia? Reply with one word: yes or no."""

FIVEWAY_PROMPT = """Here is a question and a speaker's answer to it, transcribed.

Question: {question}
Answer: {answer}

Which one of these describes the speaker?
Control (no aphasia)
Anomic or conduction aphasia
Broca's aphasia
Wernicke's aphasia
Not human speech (garbled machine text)

Reply with exactly one of these five options."""

# The judge's five options, mapped to the classifier's class names (scorer.CLASSES).
FIVEWAY_LABELS = [("control", "Control"), ("anomic", "Conduction/Anomic"), ("conduction", "Conduction/Anomic"),
                  ("broca", "Broca"), ("wernicke", "Wernicke"), ("not human", "Model_scrambles"),
                  ("garbled", "Model_scrambles")]


def binary_messages(question: str, answer: str) -> list:
    return [{"role": "user", "content": BINARY_PROMPT.format(question=question, answer=answer)}]


def fiveway_messages(question: str, answer: str) -> list:
    return [{"role": "user", "content": FIVEWAY_PROMPT.format(question=question, answer=answer)}]


def parse_binary(reply: str) -> str | None:
    """'aphasia' or 'not aphasia' from the judge's first yes or no; None when it gave neither."""
    match = re.search(r"\b(yes|no)\b", reply.lower())
    return None if match is None else {"yes": "aphasia", "no": "not aphasia"}[match.group(1)]


def parse_fiveway(reply: str) -> str | None:
    """The class whose option the reply names first; None when it names none."""
    lowered = reply.lower()
    named = [(lowered.find(word), label) for word, label in FIVEWAY_LABELS if word in lowered]
    return min(named)[1] if named else None


def parse_tab(reply: str) -> dict:
    """The 19 features (0 or 1) from the judge's JSON reply; a feature it left out counts as absent."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    labels = json.loads(match.group(0)) if match else {}
    return {name: int(labels.get(name, 0)) for name, _, _ in FEATURES}


class Judge:
    """DeepSeek chat completions, temperature 0, cached on disk by request content."""

    def __init__(self):
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        key_file = Path.home() / ".deepseek_api_key"
        if not key and key_file.exists():
            key = key_file.read_text().strip()
        assert key, "no API key: export DEEPSEEK_API_KEY=... or put it in ~/.deepseek_api_key"
        self.key = key
        self.cache = {}
        if CACHE_PATH.exists():
            for line in CACHE_PATH.read_text().splitlines():
                record = json.loads(line)
                # NOTE: a reasoning model is not deterministic
                # even at temperature 0: the same request can come back with a
                # different reply. The first reply a request ever got is the one
                # every run uses, so reruns reproduce earlier verdicts.
                self.cache.setdefault(record["request"], record)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def ask(self, messages: list, json_reply: bool = False) -> str:
        """The judge's reply to one conversation."""
        body = {"model": JUDGE_MODEL, "messages": messages, "temperature": 0}
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        request = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if request in self.cache:
            return self.cache[request]["reply"]
        for attempt in range(6):
            # NOTE: a network error raises instead of
            # returning a status, and a busy API answers 429 or 5xx; both are
            # retried with exponential backoff.
            try:
                response = requests.post(API_URL, json=body, timeout=300,
                                         headers={"Authorization": f"Bearer {self.key}"})
            except requests.exceptions.RequestException:
                time.sleep(5 * 2 ** attempt)
                continue
            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]
                record = {"request": request, "messages": messages, "reply": message["content"],
                          "reasoning": message.get("reasoning_content"), "usage": data.get("usage")}
                with self.lock:
                    if request not in self.cache:
                        self.cache[request] = record
                        with CACHE_PATH.open("a") as handle:
                            handle.write(json.dumps(record) + "\n")
                    return self.cache[request]["reply"]
            time.sleep(5 * 2 ** attempt)
        raise RuntimeError(f"DeepSeek request failed: {response.status_code} {response.text[:200]}")

    def ask_many(self, conversations: list, json_reply: bool = False) -> list:
        """Replies to many conversations, in order, WORKERS at a time; identical conversations are asked once.

        NOTE: the same answer often turns up under several
        conditions (a restore that changes nothing gives the circuit's answer
        again). Asked separately, at the same time, identical requests would both
        reach the API and could get different verdicts.
        """
        unique = list({json.dumps(messages, sort_keys=True): messages for messages in conversations}.values())
        with ThreadPoolExecutor(WORKERS) as pool:
            replies = dict(zip([json.dumps(messages, sort_keys=True) for messages in unique],
                               pool.map(lambda messages: self.ask(messages, json_reply), unique)))
        return [replies[json.dumps(messages, sort_keys=True)] for messages in conversations]


def judge_tab_benchmarks(judge: Judge, benchmarks_dir: Path, out_dir: Path):
    """The TAB judge on every connected-text answer of every benchmarks file in a folder, per condition."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(benchmarks_dir.glob("*.json")):
        benchmark = json.loads(path.read_text())
        items = [entry for entry in benchmark["tab"] if entry["subtest"] == "connected_text"]
        jobs = [(entry, condition, record["response"]) for entry in items
                for condition, record in entry["responses"].items()]
        replies = judge.ask_many([build_messages(entry["prompt"], response) for entry, _, response in jobs],
                                 json_reply=True)
        verdicts = {}
        for (entry, condition, response), reply in zip(jobs, replies):
            features = parse_tab(reply)
            verdicts.setdefault(condition, []).append({"item": entry["item"], "prompt": entry["prompt"],
                                                       "response": response, "features": features})
        summary = {condition: {"n": len(rows),
                               "mean_features_present": sum(sum(row["features"].values()) for row in rows) / len(rows),
                               "share_impaired": sum(row["features"]["overall_communication_impairment"]
                                                     for row in rows) / len(rows)}
                   for condition, rows in verdicts.items()}
        (out_dir / path.name).write_text(json.dumps({"judge": JUDGE_MODEL, "source": path.name, "summary": summary,
                                                     "verdicts": verdicts}, indent=1))
        cells = "  ".join(f"{condition} {row['mean_features_present']:.1f}/{row['share_impaired']:.1f}"
                          for condition, row in list(summary.items())[:4])
        print(f"{path.stem:<34} features present / share impaired: {cells}", flush=True)


MIN_COMPRESSION_CHARS = 100   # below this, zlib's fixed overhead dominates and the ratio says nothing


def compression_ratio(text: str) -> float | None:
    """zlib-compressed size over raw size: about 0.5-0.6 for prose, far lower for a string that repeats itself.

    NOTE: None for answers shorter than MIN_COMPRESSION_CHARS,
    whose ratio exceeds 1 from the compressed format's overhead alone.
    """
    raw = text.encode()
    if len(text) < MIN_COMPRESSION_CHARS:
        return None
    return round(len(zlib.compress(raw, 9)) / len(raw), 3)


def judge_circuits(judge: Judge):
    """The binary and five-way judge on the answer of every circuit search.py and em_search.py found."""
    questions = {"cinderella": LLAMA2_STORY_USER, "sandwich": SANDWICH_USER}
    rows = []
    for source, pattern in (("search", "search/*_rep1.25_classifier.json"), ("em_search", "em_search/*.json")):
        for path in sorted(RESULTS.glob(pattern)):
            results = json.loads(path.read_text())
            model, task = results["config"]["model"], results["config"]["task"]
            for objective, entry in results["objectives"].items():
                if not entry.get("pruned"):
                    continue
                circuit = entry["pruned"]["circuit"]
                rows.append({"source": source, "model": model, "task": task, "objective": objective,
                             "n_heads": len(circuit["heads"]), "n_mlp_blocks": len(circuit["mlp_blocks"]),
                             "n_lesions": len(circuit.get("lesions", [])), "value": circuit["value"],
                             "probabilities": circuit["probabilities"], "answer": circuit["story"],
                             "compression_ratio": compression_ratio(circuit["story"])})
    binary = judge.ask_many([binary_messages(questions[row["task"]], row["answer"]) for row in rows])
    fiveway = judge.ask_many([fiveway_messages(questions[row["task"]], row["answer"]) for row in rows])
    for row, binary_reply, fiveway_reply in zip(rows, binary, fiveway):
        row.update(judge_binary=parse_binary(binary_reply), judge_fiveway=parse_fiveway(fiveway_reply),
                   judge_replies={"binary": binary_reply, "fiveway": fiveway_reply})
    out = RESULTS / "report" / "circuit_judge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"judge": JUDGE_MODEL, "circuits": rows}, indent=1))
    for row in rows:
        print(f"{row['source']:<9} {row['model']:<16} {row['task']:<10} {row['objective']} value {row['value']:.2f} "
              f"judge {row['judge_binary']} / {row['judge_fiveway']}, compression {row['compression_ratio']}",
              flush=True)
    print(f"wrote {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tab", action="store_true", help="judge every benchmarks file's connected text")
    parser.add_argument("--circuits", action="store_true", help="judge the answer of every circuit found")
    args = parser.parse_args()
    if args.circuits:
        judge_circuits(Judge())
    if args.tab:
        judge = Judge()
        # benchmarks.py's results for search.py's circuits and em_search.py's,
        # em_attention.py's checks of the six spaces' sets, grown_heads.py's of
        # the grown-heads sets and task_overlap.py's of the Cinderella and
        # sandwich sets combined (the same format).
        for folder in (RESULTS, RESULTS / "em_search", RESULTS / "em_attention", RESULTS / "grown_heads",
                       RESULTS / "task_overlap"):
            if (folder / "benchmarks").exists():
                judge_tab_benchmarks(judge, folder / "benchmarks", folder / "tab_judge")


if __name__ == "__main__":
    main()
