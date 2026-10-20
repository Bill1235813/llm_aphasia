"""
predict.py — score new transcripts with the trained ensemble.

    python src/predict.py --text "Cinderella she uh the ball and the prince..."
    python src/predict.py --csv my_transcripts.csv --text-column transcript

The first form prints P(aphasia) for each --text (you may pass several). The
second scores every row of a CSV and writes <name>_scored.csv with two new
columns, P_aphasia and predicted_class.

For each transcript the script recomputes exactly the features listed in
config.FEATURES — with the same code that built the training table — then
runs every model in models/ and averages their probabilities.

NOTE: gold_* features (CLAN's own measures) only exist for
the AphasiaBank corpus and cannot be computed for new text. If config.FEATURES
lists one, this script stops with a clear message rather than guess.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data import CLASSES
from features import narrative, repetition, proxies, semsimilar
from model import AphasiaClassifier


class FeatureExtractor:
    """Computes config.FEATURES for one text, loading only the models needed."""

    def __init__(self):
        wanted = set(config.FEATURES)
        computable = set(narrative.COLUMNS + repetition.COLUMNS
                         + proxies.COLUMNS + semsimilar.COLUMNS)
        impossible = wanted - computable
        assert not impossible, (f"cannot compute {sorted(impossible)} for new text; "
                                f"remove them from config.FEATURES and retrain")

        self.reference = None
        if wanted & set(semsimilar.COLUMNS):
            transcripts = json.loads(Path(config.TRANSCRIPTS_FILE).read_text())
            labels_list = [r[config.LABEL_COLUMN] for r in transcripts]
            self.reference = semsimilar.SimilarityReference(
                [r[config.TEXT_COLUMN] for r in transcripts],
                labels_list,
                config.SIMILARITY_MODEL)


    def compute(self, text):
        values = {}
        values.update(narrative.compute(text))
        values.update(repetition.compute(text))
        values.update(proxies.compute(text))
        if self.reference is not None:
            values.update(self.reference.compute(text))
        #if self.scorer is not None:
        #    values.update(self.scorer.compute(text))
        return np.array([values[f] for f in config.FEATURES], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", action="append", help="a transcript to score (repeatable)")
    parser.add_argument("--csv", help="a CSV file with one transcript per row")
    parser.add_argument("--text-column", default=config.TEXT_COLUMN,
                        help="which CSV column holds the transcript")
    args = parser.parse_args()
    assert args.text or args.csv, "give --text ... or --csv ..."

    if args.csv:
        table = pd.read_csv(args.csv)
        texts = table[args.text_column].fillna("").tolist()
    else:
        texts = args.text

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_dirs = sorted(config.MODELS_DIR.glob("seed_*"))
    assert seed_dirs, f"no trained models in {config.MODELS_DIR} — run train.py first"
    print(f"{len(texts)} transcripts, {len(seed_dirs)} models, device {device}")

    extractor = FeatureExtractor()
    features = np.stack([extractor.compute(t) for t in texts])   # (n, n_features)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)
    tokens = tokenizer(texts, padding="max_length", truncation=True,
                       max_length=config.MAX_TOKENS, return_tensors="pt").to(device)

    prob_sum = np.zeros((len(texts), config.NUM_CLASSES))
    for seed_dir in seed_dirs:
        model, meta = AphasiaClassifier.load(seed_dir, device)
        assert meta["features"] == config.FEATURES, \
            f"{seed_dir} was trained with different features: {meta['features']}"
        standardised = (features - np.array(meta["feature_mean"])) / np.array(meta["feature_std"])
        with torch.no_grad():
            logits = model(tokens["input_ids"], tokens["attention_mask"],
                           torch.from_numpy(standardised.astype(np.float32)).to(device))
        prob_sum += F.softmax(logits, dim=-1).cpu().numpy()
        del model
        torch.cuda.empty_cache()
    probs = prob_sum / len(seed_dirs)
    predicted = [CLASSES[c] for c in probs.argmax(1)]
    confidences = probs.max(1)

    if args.csv:
        table["P_Control"] = probs[:, 0]
        table["P_Cond_Anomic"] = probs[:, 1]
        table["P_Broca"] = probs[:, 2]
        table["P_Wernicke"] = probs[:, 3]
        table["P_Model_scrambles"] = probs[:, 4]
        table["predicted_class"] = predicted
        table["confidence"] = confidences
        out = Path(args.csv).with_name(Path(args.csv).stem + "_scored.csv")
        table.to_csv(out, index=False)
        print(f"wrote {out}")
    for text, p, label in zip(texts, confidences, predicted):
        print(f"{label}\n  {text[:120]!r}")


if __name__ == "__main__":
    main()
