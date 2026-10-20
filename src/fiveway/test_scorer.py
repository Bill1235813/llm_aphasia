"""Check FiveWayScorer against the classifier package's own outputs.

1. The text-only features (narrative, repetition, proxies) of 300 corpus
   transcripts must equal the package's features.csv. The similarity features
   in features.csv are leave-one-out, so they are checked separately: with each
   transcript removed from its own reference groups they must match too.
2. The ensemble probabilities of the package's example transcripts must equal
   the ones predict.py wrote to data/cindsand/example_new_transcripts_scored.csv.

Usage (CPU is enough; about 3 minutes):
    CUDA_VISIBLE_DEVICES= python src/fiveway/test_scorer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiveway.scorer import CLASSES, PACKAGE_DIR, FiveWayScorer, classifier_config  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = FiveWayScorer(device)
    print(f"device {device}, {len(scorer.adapter_names)} seeds, features {scorer.feature_names}")

    # --- 1. features of corpus transcripts against features.csv -------------
    transcripts = json.loads(Path(classifier_config.TRANSCRIPTS_FILE).read_text())
    table = pd.read_csv(classifier_config.FEATURES_FILE).set_index(classifier_config.ID_COLUMN)
    rng = np.random.default_rng(0)
    picked = rng.choice(len(transcripts), size=300, replace=False)
    texts = [transcripts[i][classifier_config.TEXT_COLUMN] for i in picked]
    ids = [transcripts[i][classifier_config.ID_COLUMN] for i in picked]
    ours = scorer.features(texts)
    theirs = table.loc[ids, scorer.feature_names].to_numpy(dtype=np.float32)
    text_only = [i for i, name in enumerate(scorer.feature_names) if not name.startswith("cossim_delta")]
    worst = np.abs(ours[:, text_only] - theirs[:, text_only]).max(axis=0)
    print("text-only features, max |ours - features.csv| per feature:")
    for i, value in zip(text_only, worst):
        print(f"  {scorer.feature_names[i]:<26} {value:.2e}")
    assert worst.max() < 1e-4, "text-only features differ from features.csv"

    # Leave-one-out similarity, as build_features.py computed it for the corpus.
    loo = scorer.reference.leave_one_out_deltas()
    loo_values = np.array([[loo[i][name] for name in scorer.feature_names if name.startswith("cossim_delta")]
                           for i in picked])
    similarity_columns = [i for i, name in enumerate(scorer.feature_names) if name.startswith("cossim_delta")]
    worst_similarity = np.abs(loo_values - theirs[:, similarity_columns]).max()
    print(f"similarity features (leave-one-out), max |ours - features.csv|: {worst_similarity:.2e}")
    assert worst_similarity < 1e-3, "similarity features differ from features.csv"

    # --- 2. probabilities of the example transcripts against predict.py ------
    examples = pd.read_csv(PACKAGE_DIR / "data" / "cindsand" / "example_new_transcripts_scored.csv")
    probabilities = scorer.probabilities(examples["transcript"].tolist())
    expected = examples[["P_Control", "P_Cond_Anomic", "P_Broca", "P_Wernicke", "P_Model_scrambles"]].to_numpy()
    print("example transcripts, ours vs predict.py:")
    for name, row, want in zip(examples["name"], probabilities, expected):
        print(f"  {name:<20} {np.round(row, 4)}  {np.round(want, 4)}  argmax {CLASSES[row.argmax()]}")
    difference = np.abs(probabilities - expected).max()
    print(f"max |difference| = {difference:.2e}")
    assert difference < 2e-3, "ensemble probabilities differ from predict.py"
    print("OK")


if __name__ == "__main__":
    main()
