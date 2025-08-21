import os
import json
import torch
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
import torchvision.models as models
from selective_search import selective_search

# Paths
Model_Path = "results/svm_model.joblib"
Test_Img = "data/test/485227412_e335662bb5_b_jpg.rf.c3b02e09097409dfeb31d85ce92175b2.jpg"
Conf_Thresh = 0.9  # only show boxes with high confidence

# SVM Load
clf = joblib.load(Model_Path)

#CNN Feature Extractor Loading 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final layer
model.eval().to(device)

# Preprocessing
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Test images loading and Processing
img = Image.open(Test_Img).convert("RGB")
img_np = np.array(img)
_, regions = selective_search(img_np, scale=500, min_size=20)

#Ballons Prediction
predict_boxes = []
for r in regions:
    x, y, w, h = map(int, r["rect"])
    if w * h < 2000 or w / h > 1.5 or h / w > 1.5:
        continue
    crop_img = img.crop((x, y, x + w, y + h))
    input_tensor = tf(crop_img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(input_tensor).squeeze().cpu().numpy().reshape(1, -1)
        prob = clf.predict_proba(feature)[0][1]  

    if prob > Conf_Thresh:
        predict_boxes.append((x, y, w, h, prob))

# images Visualization
fig, ax = plt.subplots(1)
ax.imshow(img)
for x, y, w, h, prob in predict_boxes:
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y - 5, f"{prob:.2f}", color='white',
            bbox=dict(facecolor='red', alpha=0.5))

plt.title("Predicted Balloons")
plt.axis("off")
plt.tight_layout()
plt.show()


