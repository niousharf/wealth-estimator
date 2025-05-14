from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "upload" in response.text.lower()  # Basic check for content


def test_invalid_file_upload():
    # Simulate a non-image file upload
    file = {"file": ("test.txt", b"not-an-image", "text/plain")}
    response = client.post("/predict-form", files=file)
    assert response.status_code == 200
    assert "Invalid file type" in response.text
