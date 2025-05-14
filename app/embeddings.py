from torchvision import models, transforms
from PIL import Image
import torch
import io
from functools import lru_cache


@lru_cache()
def get_model():
    """Loads and caches a pretrained ResNet-18 model with the classification head removed.
    This model is used to extract embeddings from images.

    Returns:
        torch.nn.Module: A pretrained ResNet-18 model with final FC layer replaced by Identity.
    """
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model


@lru_cache()
def get_transform():
    """Returns a cached torchvision transform function for preprocessing input images.

    Returns:
        torchvision.transforms.Compose: Transform pipeline for resizing, normalization, etc.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def extract_embedding(image_bytes):
    """Processes a raw image (as bytes) and returns a ResNet-based feature embedding.

    Args:
        image_bytes (bytes): Raw bytes of the input image.

    Returns:
        np.ndarray: 1D embedding vector representing the image features.

    Raises:
        ValueError: If the image cannot be processed or embedded.
    """
    try:
        model = get_model()
        transform = get_transform()

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img_tensor).squeeze().numpy()
        return embedding

    except Exception as e:
        raise ValueError(f"Failed to extract embedding: {str(e)}")
