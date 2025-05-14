import numpy as np
import json
from pathlib import Path

# Load embeddings and metadata using robust relative path
mock_dir = Path(__file__).parent.parent / "mock_profiles"
mock_embeds = np.load(mock_dir / "wealthy_embeddings.npy")

with open(mock_dir / "wealthy_metadata.json", "r") as f:
    mock_metadata = json.load(f)

def cosine_similarity(a, b):
    """Computes the cosine similarity between two embedding vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors.
    It is defined as the cosine of the angle between them and ranges from -1 (opposite)
    to 1 (identical), with 0 indicating orthogonality (no similarity).

    Args:
        a (array-like): First embedding vector.
        b (array-like): Second embedding vector.

    Returns:
        float: Cosine similarity score between `a` and `b`.

    Raises:
        ValueError: If the input vectors do not have the same shape.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Embedding shapes don't match: {a.shape} vs {b.shape}")
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def find_top_matches(user_embedding, top_k=3):
    """Compare user embedding to wealthy profile embeddings and return top matches.

    Args:
        user_embedding (np.ndarray): 1D embedding of the input image.
        top_k (int): Number of top similar profiles to return.

    Returns:
        List[Dict]: Top K matches with name, net worth, and similarity score.
    """
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