# (Use random embeddings for mock — shape: (5, 512), save with numpy)
# save_mock.py (run once to generate mock)
import numpy as np
np.save("mock_profiles/wealthy_embeddings.npy", np.random.rand(5, 512))