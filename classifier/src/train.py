"""
train.py — train the classifier and report how well it works.

    python src/train.py                 # config.N_SEEDS seeds (20 = the paper)
    python src/train.py --n-seeds 3     # quick try-out

What happens, for each seed:
  1. Split the transcripts 70 / 10 / 20 into train / validation / test,
     keeping the same mix of diagnostic groups in each part (stratified).
  2. Standardise the features using the TRAINING split's mean and std.
  3. Train for up to config.EPOCHS passes; keep the epoch with the best
     validation aphasia-F1; stop early after config.PATIENCE epochs of no gain.
  4. Score the test split and save the model to models/seed_XX/.

Then the seeds are combined: every transcript was in the test split of some
seeds, so its final ("ensemble") prediction averages those seeds' probabilities.
That ensemble score is the headline number, and models/ is what predict.py uses.

Outputs
    models/seed_XX/            one trained model per seed
    models/results.json        per-seed and ensemble metrics
    models/ensemble_predictions.csv   P(aphasia) for every transcript
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from data import CLASSES, load_dataset
from model import AphasiaClassifier

from sklearn.metrics import f1_score, precision_score, recall_score


def stratified_split(groups, seed):
    """70/10/20 indices, with each diagnostic group split in that ratio.

    NOTE: the corpus is unbalanced (399 controls vs 58
    Wernicke). A plain random split could leave the test set with almost no
    Wernicke transcripts and make the score swing from seed to seed.
    """
    rng = np.random.default_rng(seed)
    train_frac, val_frac, _ = config.SPLIT_FRACTIONS
    train, val, test = [], [], []
    groups = np.asarray(groups)
    for group in np.unique(groups):
        idx = np.where(groups == group)[0]
        rng.shuffle(idx)
        n_train = round(train_frac * len(idx))
        n_val = round(val_frac * len(idx))
        train.extend(idx[:n_train])
        val.extend(idx[n_train:n_train + n_val])
        test.extend(idx[n_train + n_val:])
    return np.array(train), np.array(val), np.array(test)


def aphasia_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return f1, precision, recall


def make_loader(tokens, features, y, idx, shuffle_balanced):
    dataset = TensorDataset(tokens["input_ids"][idx], tokens["attention_mask"][idx],
                            torch.from_numpy(features[idx]), torch.from_numpy(y[idx]))
    if not shuffle_balanced:
        return DataLoader(dataset, batch_size=config.BATCH_SIZE)
    # NOTE: there are almost twice as many aphasia transcripts
    # as controls. Sampling each class in proportion to 1/count makes every
    # training batch roughly half-and-half, so the model is not rewarded for
    # simply guessing "aphasia".
    counts = np.bincount(y[idx], minlength=config.NUM_CLASSES)
    weights = torch.tensor((1.0 / counts)[y[idx]], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(idx), replacement=True)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=sampler)


def predict_probs(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for input_ids, attention_mask, features, _ in loader:
            logits = model(input_ids.to(device), attention_mask.to(device), features.to(device))
            probs.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)      # (n, 2): [P(aphasia), P(control)]


def train_one_seed(seed, data, tokens, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_idx, val_idx, test_idx = stratified_split(data["groups"], seed)

    # Standardise every feature to mean 0 / std 1 using the TRAINING rows
    # only. The same mean/std is saved and reused at prediction time.
    feature_mean = data["features"][train_idx].mean(0)
    feature_std = data["features"][train_idx].std(0) + 1e-6
    features = (data["features"] - feature_mean) / feature_std

    train_loader = make_loader(tokens, features, data["y"], train_idx, shuffle_balanced=True)
    val_loader = make_loader(tokens, features, data["y"], val_idx, shuffle_balanced=False)
    test_loader = make_loader(tokens, features, data["y"], test_idx, shuffle_balanced=False)

    model = AphasiaClassifier(n_features=features.shape[1]).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.LEARNING_RATE, weight_decay=1e-2)

    best_f1, best_state, epochs_without_gain = -1.0, None, 0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for input_ids, attention_mask, feats, y in train_loader:
            logits = model(input_ids.to(device), attention_mask.to(device), feats.to(device))
            loss = F.cross_entropy(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_pred = predict_probs(model, val_loader, device).argmax(1)
        f1, _, _ = aphasia_f1(data["y"][val_idx], val_pred)
        print(f"    seed {seed} epoch {epoch + 1:>2}: loss={total_loss / len(train_loader):.3f}"
              f"  val aphasia-F1={f1:.3f}", flush=True)
        if f1 > best_f1 + 1e-4:
            best_f1, epochs_without_gain = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= config.PATIENCE:
                break

    model.load_state_dict(best_state)
    test_probs = predict_probs(model, test_loader, device)
    model.save(config.MODELS_DIR / f"seed_{seed:02d}", feature_mean, feature_std)

    del model
    torch.cuda.empty_cache()
    return test_idx, test_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=config.N_SEEDS)
    n_seeds = parser.parse_args().n_seeds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset()
    print(f"device: {device}   features: {config.FEATURES}")

    # Tokenise every transcript once, up front; the tensors are shared by all seeds.
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_ENCODER)
    tokens = tokenizer(data["texts"], padding="max_length", truncation=True,
                       max_length=config.MAX_TOKENS, return_tensors="pt")

    n = len(data["y"])
    prob_sum = np.zeros((n, config.NUM_CLASSES))
    prob_count = np.zeros(n)
    per_seed = []
    for seed in range(n_seeds):
        print(f"\n=== seed {seed + 1}/{n_seeds} ===")
        test_idx, test_probs = train_one_seed(seed, data, tokens, device)
        f1, precision, recall = aphasia_f1(data["y"][test_idx], test_probs.argmax(1))
        accuracy = float((test_probs.argmax(1) == data["y"][test_idx]).mean())
        per_seed.append({"seed": seed, "accuracy": accuracy, "aphasia_f1": f1,
                         "precision": precision, "recall": recall})
        print(f"  -> test accuracy={accuracy:.3f}  aphasia-F1={f1:.3f}")
        prob_sum[test_idx] += test_probs
        prob_count[test_idx] += 1

    # Ensemble: average the probabilities each transcript received while held out.
    scored = prob_count > 0
    ensemble_probs = prob_sum[scored] / prob_count[scored, None]
    y_true = data["y"][scored]
    y_pred = ensemble_probs.argmax(1)
    f1, precision, recall = aphasia_f1(y_true, y_pred)
    ensemble = {"accuracy": float((y_pred == y_true).mean()), "aphasia_f1": f1,
                "precision": precision, "recall": recall,
                "n_transcripts_scored": int(scored.sum())}

    print("\n=== per-seed test results (mean ± std) ===")
    for key in ["accuracy", "aphasia_f1", "precision", "recall"]:
        values = [s[key] for s in per_seed]
        print(f"  {key:<11} {np.mean(values):.3f} ± {np.std(values):.3f}")
    print("\n=== ensemble (average over seeds, every transcript scored while held out) ===")
    for key, value in ensemble.items():
        print(f"  {key:<22} {value:.3f}" if isinstance(value, float) else f"  {key:<22} {value}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "results.json").write_text(json.dumps({
        "features": config.FEATURES, "text_encoder": config.TEXT_ENCODER,
        "surprisal_model": config.SURPRISAL_MODEL, "n_seeds": n_seeds,
        "classes": CLASSES, "per_seed": per_seed, "ensemble": ensemble}, indent=2))
    pd.DataFrame({
        config.ID_COLUMN: np.array(data["ids"])[scored],
        config.LABEL_COLUMN: np.array(data["groups"])[scored],
        "true_class": [CLASSES[c] for c in y_true],
        "P_Control": ensemble_probs[:, 0],
        "P_Cond_Anomic": ensemble_probs[:, 1],
        "P_Broca": ensemble_probs[:, 2],
        "P_Wernicke": ensemble_probs[:, 3],
        "P_Model_scrambles": ensemble_probs[:, 4],
        "predicted_class": [CLASSES[c] for c in y_pred],
    }).to_csv(config.MODELS_DIR / "ensemble_predictions.csv", index=False)


if __name__ == "__main__":
    main()
