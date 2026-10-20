import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

COLUMNS = ["cossim_delta_broca", "cossim_delta_wernicke", "cossim_delta_anomic", "cossim_delta_scrambles"]

class SimilarityReference:
    def __init__(self, texts, labels, model_name):
        self.labels = np.asarray(labels)
        
        # 1. Create boolean masks for all 5 specific groups
        self.is_control = (self.labels == "Control")
        self.is_broca = (self.labels == "Broca")
        self.is_wernicke = (self.labels == "Wernicke")
        self.is_anomic = np.isin(self.labels, ["Conduction", "Anomic", "Conduction/Anomic"])
        self.is_scramble = (self.labels == "Model scrambles")

        self.embedder = SentenceTransformer(model_name)
        self.embeddings = self.embedder.encode(texts, batch_size=128,
                                               normalize_embeddings=True,
                                               show_progress_bar=False)

    def leave_one_out_deltas(self):
        similarity = cosine_similarity(self.embeddings)
        n = similarity.shape[0]
        not_self = ~np.eye(n, dtype=bool)
        
        # Helper function to compute LOO means safely
        def loo_mean(mask):
            m = mask[None, :] & not_self
            return (similarity * m).sum(1) / m.sum(1).clip(min=1)
            
        mean_control = loo_mean(self.is_control)
        mean_broca = loo_mean(self.is_broca)
        mean_wernicke = loo_mean(self.is_wernicke)
        mean_anomic = loo_mean(self.is_anomic)
        mean_scramble = loo_mean(self.is_scramble)

        deltas = []
        for i in range(n):
            deltas.append({
                "cossim_delta_broca": float(mean_control[i] - mean_broca[i]),
                "cossim_delta_wernicke": float(mean_control[i] - mean_wernicke[i]),
                "cossim_delta_anomic": float(mean_control[i] - mean_anomic[i]),
                "cossim_delta_scrambles": float(mean_control[i] - mean_scramble[i])
            })
        return deltas

    def compute(self, text):
        embedding = self.embedder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        sim = cosine_similarity(embedding, self.embeddings)[0]
        
        # Helper function to prevent division-by-zero (NaN) if a group is missing
        def safe_mean(mask):
            arr = sim[mask]
            return float(arr.mean()) if arr.size > 0 else 0.0
            
        mean_control = safe_mean(self.is_control)
        return {
            "cossim_delta_broca": float(mean_control - safe_mean(self.is_broca)),
            "cossim_delta_wernicke": float(mean_control - safe_mean(self.is_wernicke)),
            "cossim_delta_anomic": float(mean_control - safe_mean(self.is_anomic)),
            "cossim_delta_scrambles": float(mean_control - safe_mean(self.is_scramble))
        }
