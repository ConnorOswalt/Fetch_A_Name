import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

# Load model
checkpoint = torch.load("dog_namer.pth", weights_only=True)
classes = checkpoint["classes"]
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def name_dog(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
        probs = F.softmax(pred, dim=1)
        top3_prob, top3_idx = probs.topk(3)
        return [(classes[i], float(p)*100) for i, p in zip(top3_idx[0], top3_prob[0])]

# Test on any images
test_folder = Path("data/test_images")   # ← put a few new dog photos here
for img_path in list(test_folder.glob("*.jpg"))[:5]:
    print(f"\n{img_path.name}")
    for name, conf in name_dog(img_path):
        print(f"  → {name} ({conf:.1f}%)")