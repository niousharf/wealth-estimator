import numpy as np
from app.similarity import find_top_matches
from pathlib import Path

# Load the same mock data as used in the actual app
mock_dir = Path(__file__).parent.parent / "mock_profiles"
mock_embeds = np.load(mock_dir / "wealthy_embeddings.npy")


def test_find_top_matches_structure():
    # Use the first embedding as a test query
    test_emb = mock_embeds[0]
    results = find_top_matches(test_emb, top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3

    for match in results:
        assert "name" in match
        assert "net_worth" in match
        assert "similarity_score" in match
        assert isinstance(match["similarity_score"], float)
        assert 0.0 <= match["similarity_score"] <= 1.0


def test_find_top_matches_ordering():
    # Ensure similarity ordering is descending
    test_emb = mock_embeds[0]
    results = find_top_matches(test_emb, top_k=3)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
