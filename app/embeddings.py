from torchvision import models, transforms
from PIL import Image
import torch
import io

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # remove classification head
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()
    return embedding