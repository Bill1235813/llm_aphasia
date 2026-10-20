"""Where the repository's data lives, also when the code runs from a snapshot.

``slurm/fiveway_submit.sh`` copies ``src/`` into a snapshot directory before a
job starts, so that later edits cannot change what a pending or requeued job
runs. The snapshot sets ``APHASIA_ROOT`` to the repository, and every path
to data, checkpoints and results is taken from it.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("APHASIA_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "fiveway"   # small final results, in git
RUNS = ROOT / "runs"                     # large outputs (caches, activations), not in git
CLASSIFIER_DIR = Path(os.environ.get("FIVEWAY_CLASSIFIER_DIR",
                                     ROOT / "classifier"))
