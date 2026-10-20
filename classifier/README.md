# Five-way classifier

The classifier that scores every answer in the study. It reads a narrative and gives five probabilities:
Control, Conduction/Anomic, Broca, Wernicke and Model_scrambles (degenerate machine output).

- **Model.** `BAAI/bge-large-en-v1.5` with LoRA adapters (rank 16) reads the first 384 tokens of the text. Its
  vector is concatenated with 14 hand-made features (`src/config.py`, `FEATURES`): utterance and word counts,
  repetition statistics, word-class proxies (open- to closed-class ratio, lexical density, adverbs, task nouns,
  out-of-vocabulary words), and how much closer the text is to each group's transcripts than to the control
  transcripts (`all-MiniLM-L6-v2` cosine similarity). A two-layer head (256 hidden units) gives the five logits.
- **Ensemble.** `src/train.py` trains 20 seeds on different 70/10/20 splits; a prediction averages the 20 softmax
  outputs. `src/fiveway/scorer.py` (in the study code) loads the 20 seeds from `models/cindsand/` and reproduces
  `src/predict.py`.

This folder holds **code only**. The training transcripts are AphasiaBank data and cannot be redistributed, and the
trained weights are not distributed either.

## Training it

1. Build the data files described in `data/README.md` (AphasiaBank access is needed for the human transcripts; the
   model-scramble class needs degenerate outputs of your own).
2. From this folder:

   ```bash
   python src/build_features.py   # writes data/cindsand/features.csv
   python src/train.py            # writes models/cindsand/seed_*/
   python src/predict.py --csv <your.csv> --text-column <column>   # optional: score new texts
   ```

`src/config.py` holds every setting; the task defaults to `cindsand` (both narratives and the model scrambles,
`APHASIA_TASK` overrides it).
