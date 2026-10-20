"""Score texts with the five-way ``cindsand`` aphasia classifier.

The classifier is the stand-alone package in ``classifier/`` (code only in this
release: its training transcripts are AphasiaBank data, and its 20 trained seeds
are not distributed). It was trained on
the Cinderella and sandwich narratives together, plus 259 degenerate outputs of
ablated models, and tells five classes apart:

    0 Control   1 Conduction/Anomic   2 Broca   3 Wernicke   4 Model_scrambles

It has no surprisal features, so one classifier scores every generator.

``FiveWayScorer.probabilities(texts)`` reproduces the package's own
``src/predict.py`` exactly (``test_scorer.py`` checks it against the scores
that ``predict.py`` writes for a CSV of example texts), but scores a
whole batch of texts at once: per seed, BGE-large with that seed's LoRA adapter
reads the text, the 14 hand-made features are standardised with that seed's
statistics, the head gives five logits, and the softmax is averaged over the
20 seeds.

The three search objectives of the study are read off the five probabilities:

    A  aphasia vs. others   P(Conduction/Anomic) + P(Broca) + P(Wernicke)
    B  Broca vs. others     P(Broca)
    C  Wernicke vs. others  P(Wernicke)

and an objective "works" when its value exceeds 0.5. ``CosineObjective`` is the
classifier-free alternative of the cos-sim ablation: how much closer a text is
to a target group of human transcripts than to the Control group.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fiveway.paths import CLASSIFIER_DIR as PACKAGE_DIR
# NOTE: the package's modules import each other by bare name
# (``import config``, ``from features import ...``), so its src/ directory goes
# on the path itself rather than being imported as a subpackage. Importing them
# here, not copying them, guarantees the features are computed by the same code
# that built the training table.
sys.path.insert(0, str(PACKAGE_DIR / "src"))
import config as classifier_config  # noqa: E402
from features import narrative, proxies, repetition, semsimilar  # noqa: E402

CLASSES = ["Control", "Conduction/Anomic", "Broca", "Wernicke", "Model_scrambles"]
# Column of each class in the probability matrix.
CONTROL, COND_ANOMIC, BROCA, WERNICKE, SCRAMBLES = range(5)

# The three objectives: which class columns count as the target.
OBJECTIVES = {
    "A": [COND_ANOMIC, BROCA, WERNICKE],   # aphasia (any subtype) vs. Control and Model_scrambles
    "B": [BROCA],                          # Broca vs. everything else
    "C": [WERNICKE],                       # Wernicke vs. everything else
}
SUCCESS_THRESHOLD = 0.5


def works_finally(objective: str, value, probabilities) -> bool:
    """Whether a final answer reaches the target: it counts, and its objective is above SUCCESS_THRESHOLD or the
    target is the most probable of the five classes (for A, any of the three aphasia classes).

    NOTE: the searches still stop and prune at SUCCESS_THRESHOLD. Only the final judgment of
    an answer also accepts a target that is the classifier's top class without reaching 0.5, such as
    P(Wernicke) = 0.42 with every other class lower. ``value`` is None for an answer that does not count.
    """
    if value is None:
        return False
    return value > SUCCESS_THRESHOLD or int(np.argmax(probabilities)) in OBJECTIVES[objective]


def objective_values(probabilities: np.ndarray) -> dict:
    """The value of every objective for each row of a (n_texts, 5) probability matrix."""
    return {name: probabilities[:, columns].sum(axis=1) for name, columns in OBJECTIVES.items()}


class FiveWayScorer:
    """The 20-seed ensemble, loaded once and applied to batches of texts."""

    def __init__(self, device: torch.device, encoder_batch_size: int = 32):
        from peft import PeftModel
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.encoder_batch_size = encoder_batch_size
        self.feature_names = classifier_config.FEATURES

        # The reference corpus of the similarity features: every training
        # transcript, labelled by group. predict.py builds it the same way.
        transcripts = json.loads(Path(classifier_config.TRANSCRIPTS_FILE).read_text())
        self.reference = semsimilar.SimilarityReference(
            [row[classifier_config.TEXT_COLUMN] for row in transcripts],
            [row[classifier_config.LABEL_COLUMN] for row in transcripts],
            classifier_config.SIMILARITY_MODEL)

        seed_dirs = sorted(classifier_config.MODELS_DIR.glob("seed_*"))
        assert len(seed_dirs) == 20, f"expected 20 seeds in {classifier_config.MODELS_DIR}"
        metas = [json.loads((seed_dir / "meta.json").read_text()) for seed_dir in seed_dirs]
        for meta in metas:
            assert meta["features"] == self.feature_names, "a seed was trained with other features"
        self.max_tokens = metas[0]["max_tokens"]
        self.tokenizer = AutoTokenizer.from_pretrained(metas[0]["text_encoder"])

        # NOTE: the 20 seeds share one BGE-large and
        # differ only in their LoRA adapters and heads. PEFT keeps all 20
        # adapters on one base model and switches between them with
        # set_adapter, so the ensemble costs one encoder's memory instead of
        # twenty. Merging each adapter into its own copy would make every
        # forward pass slightly cheaper at twenty times the memory.
        base = AutoModel.from_pretrained(metas[0]["text_encoder"])
        self.adapter_names = [seed_dir.name for seed_dir in seed_dirs]
        self.encoder = PeftModel.from_pretrained(base, seed_dirs[0], adapter_name=self.adapter_names[0])
        for name, seed_dir in zip(self.adapter_names[1:], seed_dirs[1:]):
            self.encoder.load_adapter(seed_dir, adapter_name=name)
        self.encoder.to(device).eval()

        self.heads = []
        for seed_dir in seed_dirs:
            head = torch.nn.Sequential(
                torch.nn.Linear(classifier_config.ENCODER_DIM + len(self.feature_names), 256),
                torch.nn.ReLU(), torch.nn.Dropout(0.3),
                torch.nn.Linear(256, classifier_config.NUM_CLASSES))
            head.load_state_dict(torch.load(seed_dir / "head.pt", map_location="cpu"))
            self.heads.append(head.to(device).eval())
        self.feature_means = [np.array(meta["feature_mean"]) for meta in metas]
        self.feature_stds = [np.array(meta["feature_std"]) for meta in metas]
        self.memory = {}   # text -> its probability row, see probabilities()

    def features(self, texts: list) -> np.ndarray:
        """The 14 hand-made features of each text, unstandardised: (n_texts, 14).

        The text-only groups are computed exactly as predict.py does. The
        similarity group is batched: one MiniLM call embeds every text, then
        each text's mean cosine to every reference group is taken as in
        ``SimilarityReference.compute``.
        """
        rows = []
        for text in texts:
            values = {}
            values.update(narrative.compute(text))
            values.update(repetition.compute(text))
            values.update(proxies.compute(text))
            rows.append(values)
        similarity_rows = self.similarity_deltas(texts)
        for values, deltas in zip(rows, similarity_rows):
            values.update(deltas)
        return np.array([[values[name] for name in self.feature_names] for values in rows], dtype=np.float32)

    def group_similarities(self, texts: list) -> dict:
        """Mean cosine similarity of each text to each reference group: {group: (n_texts,)}.

        NOTE: the embeddings are normalised to length 1, so a dot
        product is the cosine similarity. ``SimilarityReference.compute`` uses
        sklearn's cosine_similarity, which renormalises; the difference is
        float rounding.
        """
        reference = self.reference
        embeddings = reference.embedder.encode(texts, batch_size=128, normalize_embeddings=True,
                                               show_progress_bar=False)
        # NOTE: embeddings: (n_texts, 384) @ reference.embeddings.T:
        # (384, n_reference) gives (n_texts, n_reference) similarities.
        similarity = embeddings @ reference.embeddings.T

        def group_mean(mask):
            # NOTE: an empty group gives 0.0, as safe_mean
            # does. The package's "Model scrambles" mask never matches the
            # corpus label "Model_scrambles", so that group is always empty
            # and cossim_delta_scrambles is the mean similarity to Control.
            # The classifier was trained with the same quirk, so it is kept.
            if not mask.any():
                return np.zeros(len(texts))
            return similarity[:, mask].mean(axis=1)

        return {
            "Control": group_mean(reference.is_control),
            "Broca": group_mean(reference.is_broca),
            "Wernicke": group_mean(reference.is_wernicke),
            "Conduction/Anomic": group_mean(reference.is_anomic),
            "Model_scrambles_as_used": group_mean(reference.is_scramble),
            "Model_scrambles": group_mean(reference.labels == "Model_scrambles"),
            "aphasia": group_mean(reference.is_broca | reference.is_wernicke | reference.is_anomic),
        }

    def similarity_deltas(self, texts: list) -> list:
        """The four cossim_delta features of each text, as SimilarityReference.compute gives them."""
        means = self.group_similarities(texts)
        control = means["Control"]
        return [{
            "cossim_delta_broca": float(control[i] - means["Broca"][i]),
            "cossim_delta_wernicke": float(control[i] - means["Wernicke"][i]),
            "cossim_delta_anomic": float(control[i] - means["Conduction/Anomic"][i]),
            "cossim_delta_scrambles": float(control[i] - means["Model_scrambles_as_used"][i]),
        } for i in range(len(texts))]

    def probabilities(self, texts: list) -> np.ndarray:
        """Ensemble class probabilities, (n_texts, 5), columns in CLASSES order.

        NOTE: most knockouts leave the answer word
        for word unchanged, so each distinct text is scored once and its row
        remembered (``self.memory``); scoring is as expensive as generating
        on a CPU, where twenty BGE-large passes per text add up.
        """
        new_texts = list(dict.fromkeys(text for text in texts if text not in self.memory))
        if new_texts:
            for text, row in zip(new_texts, self._probabilities(new_texts)):
                self.memory[text] = row
        return np.array([self.memory[text] for text in texts]).reshape(len(texts), len(CLASSES))

    @torch.no_grad()
    def _probabilities(self, texts: list) -> np.ndarray:
        features = self.features(texts)
        # NOTE: predict.py pads every text to 384
        # tokens. Padding only to the longest text of the batch gives the same
        # numbers up to float rounding (padding is masked out of attention and
        # of the mean), at about half the cost for our 180-token answers.
        tokens = self.tokenizer(texts, padding="longest", truncation=True,
                                max_length=self.max_tokens, return_tensors="pt")
        probability_sum = np.zeros((len(texts), len(CLASSES)))
        for name, head, mean, std in zip(self.adapter_names, self.heads, self.feature_means, self.feature_stds):
            self.encoder.set_adapter(name)
            # NOTE: predict.py standardises in float64 and casts
            # to float32 just before the head; doing the same keeps the inputs
            # to the head bit-identical.
            standardised = torch.from_numpy(((features - mean) / std).astype(np.float32)).to(self.device)
            for start in range(0, len(texts), self.encoder_batch_size):
                end = start + self.encoder_batch_size
                input_ids = tokens["input_ids"][start:end].to(self.device)
                attention_mask = tokens["attention_mask"][start:end].to(self.device)
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                # NOTE: hidden: (batch, 384 tokens, 1024). The mean over
                # real tokens (mask is 0 on padding) gives (batch, 1024), then
                # each row is scaled to length 1, as in AphasiaClassifier.forward.
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = F.normalize(pooled, dim=-1)
                logits = head(torch.cat([pooled, standardised[start:end]], dim=-1))
                probability_sum[start:end] += F.softmax(logits, dim=-1).cpu().numpy()
        return probability_sum / len(self.adapter_names)


class CosineObjective:
    """The classifier-free objective of the cos-sim ablation.

    For a target group, a text's score is its mean cosine similarity (MiniLM,
    the classifier's own similarity model and reference corpus) to that
    group's human transcripts minus its mean similarity to the Control
    transcripts: the negative of the classifier's cossim_delta features. A
    positive score means the text is closer to the target group than to
    Control, which is the cos-sim counterpart of P(target) > 0.5.

        A  every aphasic transcript (Anomic, Conduction, Broca, Wernicke)
        B  Broca transcripts
        C  Wernicke transcripts
    """

    TARGET_GROUPS = {"A": "aphasia", "B": "Broca", "C": "Wernicke"}
    SUCCESS_THRESHOLD = 0.0

    def __init__(self, scorer: FiveWayScorer):
        self.scorer = scorer

    def values(self, texts: list) -> dict:
        """{objective: (n_texts,)} cosine margins over Control."""
        means = self.scorer.group_similarities(texts)
        return {name: means[group] - means["Control"] for name, group in self.TARGET_GROUPS.items()}
