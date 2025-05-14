"""
This script generates mock wealthy profile embeddings and metadata
for testing the Selfie Net Worth Estimator API.

Run once before starting the API.
"""

import numpy as np
import json
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# Create directories
Path("mock_profiles").mkdir(exist_ok=True)

# Generate random embeddings: 5 profiles with 512-dim vectors
embeddings = np.random.rand(5, 512)
np.save("mock_profiles/wealthy_embeddings.npy", embeddings)

# Corresponding fake metadata
metadata = [
    {"name": "Elon Musk", "net_worth": "300B"},
    {"name": "Oprah Winfrey", "net_worth": "2.5B"},
    {"name": "Jeff Bezos", "net_worth": "200B"},
    {"name": "Rihanna", "net_worth": "1.4B"},
    {"name": "Mark Zuckerberg", "net_worth": "100B"},
]

with open("mock_profiles/wealthy_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Mock embeddings and metadata saved to 'mock_profiles/'")
