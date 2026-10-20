"""
data.py — load the transcripts and the feature table into arrays for training.

The single function ``load_dataset`` joins config.TRANSCRIPTS_FILE (text and
labels) with config.FEATURES_FILE (numbers) on the transcript id and returns
everything the trainer needs, already converted to the binary target.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# The three classes, matching the 0, 1, 2, 3, 4 mapping in config.py
CLASSES = ["Control", "Conduction/Anomic", "Broca", "Wernicke", "Model_scrambles"]

def load_dataset():
    """Returns a dictionary with:
        ids       list[str]     transcript ids
        texts     list[str]     the narratives
        features  ndarray       (n, len(config.FEATURES)) float32
        y         ndarray       (n,) int64 — 0 = control, 1 = cond/anomic, 2 = broca, 3 = wernicke,...
        groups    list[str]     the original labels (used to stratify splits)
    """
    transcripts = json.loads(Path(config.TRANSCRIPTS_FILE).read_text())
    
    # Filter out any transcripts with labels not in our mapping
    valid_data = [t for t in transcripts if t[config.LABEL_COLUMN] in config.CLASS_MAPPING]
    
    ids = [row[config.ID_COLUMN] for row in valid_data]
    texts = [row[config.TEXT_COLUMN] for row in valid_data]
    groups = [row[config.LABEL_COLUMN] for row in valid_data]
    
    table = pd.read_csv(config.FEATURES_FILE).set_index(config.ID_COLUMN)
    features = table.loc[ids, config.FEATURES].to_numpy(dtype=np.float32)
    
    # Map the AphasiaBank label to 0, 1, 2, 3, 4
    y = np.array([config.CLASS_MAPPING[g] for g in groups], dtype=np.int64)

    print(f"loaded {len(ids)} transcripts: {int((y == 0).sum())} control, "
          f"{int((y == 1).sum())} cond/anomic, {int((y == 2).sum())} broca, {int((y == 3).sum())} wernicke, {int((y ==4).sum())} model scrambles")
    return {"ids": ids, "texts": texts, "features": features, "y": y, "groups": groups}
