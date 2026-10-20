"""
model.py — the classifier: a text encoder (fine-tuned with LoRA) plus the
hand-crafted features, feeding a small classification head.

        transcript text ──► encoder (BGE-large + LoRA) ──► mean-pool ──► 1024-d
                                                                            │ concat
        16 features ──► standardise (mean 0, std 1) ─────────────────────► 16-d
                                                                            │
                                                               2-layer head ──► P(aphasia), P(control)

Everything needed to score a new transcript later — the LoRA weights, the
head, and the feature means / standard deviations — is saved by ``save`` and
restored by ``load``.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


class AphasiaClassifier(nn.Module):

    def __init__(self, n_features, encoder=None):
        super().__init__()
        if encoder is None:
            # NOTE: LoRA freezes the encoder and learns a small
            # low-rank correction to every linear layer instead. With ~1,000
            # training transcripts, updating all 335M encoder weights would
            # overfit badly; LoRA (rank 16) trains ~1% of that.
            base = AutoModel.from_pretrained(config.TEXT_ENCODER)
            encoder = get_peft_model(base, LoraConfig(
                r=config.LORA_RANK, lora_alpha=config.LORA_ALPHA,
                lora_dropout=config.LORA_DROPOUT, target_modules="all-linear",
                bias="none", task_type=TaskType.FEATURE_EXTRACTION))
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(config.ENCODER_DIM + n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, config.NUM_CLASSES))

    def forward(self, input_ids, attention_mask, features):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # NOTE: hidden is (batch, n_tokens, 1024). We average over the
        # real tokens only (attention_mask is 0 on padding) to get one
        # (batch, 1024) vector per transcript, then L2-normalise it.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, dim=-1)
        return self.head(torch.cat([pooled, features], dim=-1))   # (batch, 2) logits

    # --- saving and loading one trained seed -----------------------------

    def save(self, directory, feature_mean, feature_std):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory)          # the LoRA weights only
        torch.save(self.head.state_dict(), directory / "head.pt")
        (directory / "meta.json").write_text(json.dumps({
            "features": config.FEATURES,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "text_encoder": config.TEXT_ENCODER,
            "max_tokens": config.MAX_TOKENS,
        }, indent=2))

    @classmethod
    def load(cls, directory, device):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        base = AutoModel.from_pretrained(meta["text_encoder"])
        model = cls(n_features=len(meta["features"]),
                    encoder=PeftModel.from_pretrained(base, directory))
        model.head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu"))
        model.to(device).eval()
        return model, meta
