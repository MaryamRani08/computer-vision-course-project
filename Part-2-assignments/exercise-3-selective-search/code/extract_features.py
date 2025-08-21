import os
import json
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

Data_Dir = "data"
Labled_File = "results/labeled_proposals.json"
Ouput_File = "results/features.json"

# Loading pretrained CNN:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # removing final layer of model
model.eval()
model.to(device)

# Pre-processing
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load proposals labeld
with open(Labled_File, "r") as f:
    labeled_data = json.load(f)

# Extract features
img_features = []

grouped = defaultdict(list)
for entry in labeled_data:
    grouped[entry["file_name"]].append(entry)


for file_n, entries in tqdm(grouped.items()):
    image_path = None
    for split in ["train", "valid"]:
        path = os.path.join(Data_Dir, split, file_n)
        if os.path.exists(path):
            image_path = path
            break
    if image_path is None:
        continue

    img = Image.open(image_path).convert("RGB")

    for entry in entries:
        x, y, w, h = entry["box"]
        cropped_img = img.crop((x, y, x + w, y + h))
        input_tensor = tf(cropped_img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model(input_tensor).squeeze().cpu().numpy().tolist()

        img_features.append({
            "feature": feat,
            "label": entry["label"]
        })

with open(Ouput_File, "w") as f:
    json.dump(img_features, f)

print(f"features Saved to {Ouput_File}")
