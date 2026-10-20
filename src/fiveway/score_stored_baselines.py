"""Score the earlier study's unablated answers with the five-way classifier.

``data/interp/exp00_decoding_<model>.json`` (an earlier pipeline's outputs, not included) holds every
generator's unablated
answer on both tasks under four greedy decoding presets (plain greedy, a
repetition penalty of 1.25, DRY, no-repeat-4-grams), from the transformer_lens
pipeline. Scoring them again with the five-way classifier shows, before any
new generation, which decoding leaves each model's own answer non-aphasic,
the starting point every knockout search needs. Each answer is scored in full
and cut before the first question turn the base models write themselves.

Output: results/fiveway/stored_baselines.json, and a table on stdout.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.paths import DATA, RESULTS  # noqa: E402
from fiveway.prompts import cut_at_new_question, long_enough  # noqa: E402
from fiveway.scorer import CLASSES, FiveWayScorer, objective_values  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = FiveWayScorer(device)
    rows = []
    for path in sorted((DATA / "interp").glob("exp00_decoding_*.json")):
        for record in json.loads(path.read_text())["records"]:
            for cut in (False, True):
                story = cut_at_new_question(record["story"]) if cut else record["story"]
                rows.append({"model": record["model"], "task": record["task"], "decoding": record["decoding"],
                             "cut_at_question": cut, "answer": record["answer"], "story": story})
    scored = [row for row in rows if long_enough(row["story"])]
    probabilities = scorer.probabilities([row["story"] for row in scored])
    values = objective_values(probabilities)
    for i, row in enumerate(scored):
        row["probabilities"] = dict(zip(CLASSES, np.round(probabilities[i], 4).tolist()))
        row["argmax"] = CLASSES[int(probabilities[i].argmax())]
        row.update({f"objective_{name}": round(float(value[i]), 4) for name, value in values.items()})

    print(f"{'model':<16} {'task':<11} {'decoding':<8} {'cut':<5} {'P(A)':>6} {'P(B)':>6} {'P(C)':>6} "
          f"{'P(scr)':>6}  argmax")
    for row in rows:
        if "argmax" not in row:
            print(f"{row['model']:<16} {row['task']:<11} {row['decoding']:<8} {str(row['cut_at_question']):<5} "
                  f"(shorter than 20 characters)")
            continue
        print(f"{row['model']:<16} {row['task']:<11} {row['decoding']:<8} {str(row['cut_at_question']):<5} "
              f"{row['objective_A']:6.3f} {row['objective_B']:6.3f} {row['objective_C']:6.3f} "
              f"{row['probabilities']['Model_scrambles']:6.3f}  {row['argmax']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "stored_baselines.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
