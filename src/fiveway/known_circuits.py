"""Score named knockouts under the five-way protocol: earlier circuits, sub-masks, cross-objective checks.

The search of search.py finds its own knockouts; this script asks about
knockouts given by name, for example the earlier study's Llama-2-7B-chat
circuit, which was found with greedy decoding and a binary classifier:

    --knockout circuit=L0H25,L1H8,L1H29 --knockout L0H25+L1H8=L0H25,L1H8 --knockout mlp=L1B30

Each knockout is generated with the prompts, decoding and answer cutting of
search.py (prompts.py, rep1.25 unless --decoding says otherwise), scored with the
five-way classifier, and stored with its text. A clean run is always included.

Output: results/fiveway/known_circuits/<model>_<task>_<decoding>.json, and a table.

Usage:
    PARTITION=<partition> GRES=gpu:1 TIME=1:00:00 bash slurm/fiveway_submit.sh known-l7b 64G \\
        fiveway/known_circuits.py --model llama2-7b-chat --task cinderella \\
        --knockout circuit=L0H25,L1H8,L1H29 --knockout mlp=L1B30
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.engine import Engine, Intervention  # noqa: E402
from fiveway.paths import RESULTS  # noqa: E402
from fiveway.prompts import (STORY_TOKENS, YESNO_TOKENS, cut_at_new_question, long_enough,  # noqa: E402
                             parse_yes_no, task_prompt_ids)
from fiveway.scorer import CLASSES, FiveWayScorer, objective_values  # noqa: E402


def parse_knockout(text: str) -> tuple:
    """'name=L0H25,L1B30' -> (name, Intervention) with heads (layer, head) and MLP blocks (layer, block)."""
    name, components = text.split("=")
    heads = [(int(layer), int(head)) for layer, head in re.findall(r"L(\d+)H(\d+)", components)]
    blocks = [(int(layer), int(block)) for layer, block in re.findall(r"L(\d+)B(\d+)", components)]
    return name, Intervention.knockout(heads=heads, mlp_blocks=blocks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True, choices=["cinderella", "sandwich"])
    parser.add_argument("--decoding", default="rep1.25", choices=["greedy", "rep1.25"])
    parser.add_argument("--knockout", action="append", default=[], help="name=L<l>H<h>,L<l>B<b>,...")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = Engine(args.model, device)
    scorer = FiveWayScorer(device)
    named = [("clean", Intervention())] + [parse_knockout(text) for text in args.knockout]
    interventions = [intervention for _, intervention in named]

    yesno_ids, story_ids = task_prompt_ids(engine, args.task)
    answers = [None] * len(named)
    yesno_texts = [""] * len(named)
    if yesno_ids is not None:
        yesno_texts = engine.generate(yesno_ids, interventions, YESNO_TOKENS, args.decoding)
        answers = [parse_yes_no(text)[0] for text in yesno_texts]
    full_texts = engine.generate(story_ids, interventions, STORY_TOKENS, args.decoding)
    stories = full_texts if engine.spec.is_chat else [cut_at_new_question(text) for text in full_texts]
    probabilities = scorer.probabilities(stories)
    values = objective_values(probabilities)

    # NOTE: as in search.py, the gate applies only when the
    # unablated model passes it (named[0] is the clean run).
    gate = answers[0] is not False
    records = []
    for i, (name, intervention) in enumerate(named):
        valid = (not gate or answers[i] is not False) and long_enough(stories[i])
        records.append({"name": name, "key": intervention.key(), "answer": answers[i], "yesno": yesno_texts[i],
                        "valid": valid, "story": stories[i], "full_text": full_texts[i],
                        "probabilities": dict(zip(CLASSES, probabilities[i].round(6).tolist())),
                        "objectives": {objective: round(float(value[i]), 6) for objective, value in values.items()}})
        print(f"{name:<16} valid={valid!s:<5} A={values['A'][i]:.3f} B={values['B'][i]:.3f} C={values['C'][i]:.3f} "
              f"scr={probabilities[i][4]:.3f}  {stories[i][:150]!r}", flush=True)

    out = RESULTS / "known_circuits" / f"{args.model}_{args.task}_{args.decoding}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "records": records}, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
