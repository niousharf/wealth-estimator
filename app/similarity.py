import numpy as np
import json
from pathlib import Path

mock_embeds = np.load("mock_profiles/wealthy_embeddings.npy")
with open("mock_profiles/wealthy_metadata.json", "r") as f:
    mock_metadata = json.load(f)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_top_matches(user_embedding, top_k=3):
    similarities = [cosine_similarity(user_embedding, emb) for emb in mock_embeds]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        person = mock_metadata[idx]
        results.append({
            "name": person["name"],
            "net_worth": person["net_worth"],
            "similarity_score": round(similarities[idx], 3)
        })
    return results