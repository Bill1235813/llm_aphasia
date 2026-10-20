"""
config.py — every setting you are likely to change lives in this one file.

If you only edit ONE file in this package, it is this one. The other scripts
(build_features.py, train.py, predict.py) read their settings from here, so
you never have to touch their code to change the data, the features, the
models, or the training budget.

The file is organised top-to-bottom in the order you will usually need it:

    1. TASK/DATA — which corpus (cinderella / sandwich / yours) and which columns
    2. FEATURES  — which numeric features go into the classifier
    3. MODELS    — which pretrained language models to use
    4. TRAINING  — how long / how many times to train

Every setting has a comment saying what it does and what happens if you
change it.
"""

import os
from pathlib import Path

# The folder this package lives in. All other paths are relative to it, so
# the package keeps working if you move or rename the whole folder.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"


# =============================================================================
# 1. TASK and DATA
# =============================================================================

# The package supports two AphasiaBank elicitation tasks (the corpora are not included; build them as in
# data/README.md):
#
#   "cinderella"  retelling the Cinderella story          1,147 narratives
#   "sandwich"    describing how to make a PB&J sandwich   1,437 narratives
#
# Pick one here. The data files, the feature table and the saved models all
# live in a folder named after the task (data/<TASK>/, models/<TASK>/), so
# the two never overwrite each other.
#
# You can also choose the task without editing this file, from the terminal:
#     APHASIA_TASK=sandwich python src/train.py
TASK = os.environ.get("APHASIA_TASK", "cindsand")

# The transcript file. Each entry is one narrative plus metadata. See
# data/README.md for the columns. To use YOUR OWN data, point this at a JSON
# list of dictionaries that has at least the three columns named just below
# (or add a folder data/<your_task>/transcripts.json and set TASK to its name).
TRANSCRIPTS_FILE = DATA_DIR / TASK / "transcripts.json"

# The table of numeric features, one row per transcript. It is produced by
# build_features.py and read by train.py. You can open it in Excel and add
# columns of your own — see "Adding your own feature" in the README.
FEATURES_FILE = DATA_DIR / TASK / "features.csv"

# Where train.py saves the trained models and predict.py loads them from.
MODELS_DIR = PACKAGE_DIR / "models" / TASK

# Column names inside TRANSCRIPTS_FILE.
ID_COLUMN = "File_DB"          # a unique name for each transcript
TEXT_COLUMN = "text_response"  # the transcript text itself
LABEL_COLUMN = "Group"         # the diagnostic group

# The classifier is five-way: Control, Conduction/Anomic (both labels map to 1), Broca, Wernicke and
# Model_scrambles. Transcripts whose label is not in CLASS_MAPPING are dropped (data.py).
CLASS_MAPPING = {
    "Control": 0,
    "Anomic": 1,
    "Conduction": 1,
    "Broca": 2,
    "Wernicke": 3,
    "Model_scrambles": 4
}
NUM_CLASSES = 5



# =============================================================================
# 2. FEATURES
# =============================================================================
#
# Exclusively using proxy features for CLANfeatures
FEATURES = [
    "Total_Utts", 
    "FREQ_tokens", 
    "MLU_Words", 
    "rep_shortest_len", 
    "rep_max_count",
    "Proxy_Open_Closed_Ratio",
    "Proxy_Density",
    "Proxy_Adverb_Count",
    "Proxy_Task_Nouns_Count",
    "Proxy_OOV_Error_Count",
    "cossim_delta_broca",
    "cossim_delta_wernicke",
    "cossim_delta_anomic",
    "cossim_delta_scrambles"
]

# Which feature groups build_features.py should (re)compute. Surprisal needs a
# GPU and a large language model; the other three run on a laptop in minutes.
FEATURE_GROUPS_TO_BUILD = ["narrative", "repetition", "proxies", "semsimilar"]

# =============================================================================
# 3. MODELS
# =============================================================================

# The neural text encoder. It reads the whole transcript and produces a
# summary vector (ENCODER_DIM numbers) that is concatenated with the FEATURES
# above. Any Hugging Face encoder works; if you change it, ENCODER_DIM must
# match the model's hidden size. Measured peak GPU memory (BATCH_SIZE 4 / 8):
#
#     "BAAI/bge-small-en-v1.5"   ENCODER_DIM 384    train 0.8 / 1.4 GB   evaluate 0.2 GB
#     "BAAI/bge-base-en-v1.5"    ENCODER_DIM 768    train 1.7 / 2.9 GB   evaluate 0.5 GB
#     "BAAI/bge-large-en-v1.5"   ENCODER_DIM 1024   train 4.6 / 7.7 GB   evaluate 1.5 GB   (default)
#
# Smaller encoders train several times faster at some cost in accuracy.
TEXT_ENCODER = "BAAI/bge-large-en-v1.5"
ENCODER_DIM = 1024

# We do not fine-tune all of the encoder's weights (too many for ~1,000
# transcripts). LoRA adds a small number of trainable weights instead;
# LORA_RANK controls how many. 16 is the paper setting; 8 is a cheaper choice.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# The language model that computes surprisal. Llama-2-7B-chat is the paper
# setting but is a gated model (you must accept Meta's licence on Hugging Face
# and log in with `huggingface-cli login`). Measured GPU memory, inference only:
#
#     "gpt2"                             0.5 GB
#     "gpt2-xl"                          3.3 GB     open, a good no-licence choice
#     "meta-llama/Llama-2-7b-chat-hf"   13.0 GB     (default; needs a 16 GB+ GPU)
#
# If you change this, rebuild the surprisal features AND retrain.
SURPRISAL_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SURPRISAL_HIGH_THRESHOLD = 8.0   # nats; a word above this counts as "high surprisal"
SURPRISAL_MAX_TOKENS = 2048      # longest transcript the surprisal model reads

# The small sentence-embedding model behind cossim_delta_semantic.
SIMILARITY_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# 4. TRAINING
# =============================================================================

# The classifier is trained N_SEEDS times with different random splits, and the
# final prediction averages all of them (an "ensemble"). More seeds = more
# stable and slightly more accurate, but training time scales linearly:
# roughly 8 minutes per seed on one GPU. The paper used 20. Use 3–5 to try
# things out.
N_SEEDS = 20

# How the transcripts are divided for each seed: train / validation / test.
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)

EPOCHS = 20          # maximum passes over the training data
PATIENCE = 5         # stop early if validation F1 has not improved for this many epochs
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_TOKENS = 384     # transcripts longer than this are truncated by the encoder
