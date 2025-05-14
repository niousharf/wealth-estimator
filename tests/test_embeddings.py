import pytest
from app.embeddings import extract_embedding


@pytest.fixture
def invalid_image():
    """Fixture for an invalid image (e.g., corrupted data)."""
    return b"notanimage"


def test_extract_embedding_invalid_image(invalid_image):
    """Test that invalid images raise ValueError."""
    with pytest.raises(ValueError, match="cannot identify image file"):
        extract_embedding(invalid_image)
