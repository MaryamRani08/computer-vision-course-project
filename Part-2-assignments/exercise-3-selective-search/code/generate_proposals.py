import os
import json
import skimage.io
from tqdm import tqdm
from selective_search import selective_search

Data_split = ["train", "valid"]
Data_Dir = "data"
Output_save = "results/balloon_proposals.json"
Img_extensions = [".jpg", ".jpeg", ".png"]

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in Img_extensions)

def image_path(split):
    split_path = os.path.join(Data_Dir, split)
    return [
        os.path.join(split_path, fname)
        for fname in os.listdir(split_path)
        if is_image_file(fname)
    ]

def generate_and_save_proposals():
    proposals = {}
    
    for split in Data_split:
        print(f"Processing split: {split}")
        img_paths = image_path(split)
        for img_path in tqdm(img_paths):
            image = skimage.io.imread(img_path)
            _, img_regions = selective_search(image, scale=500, min_size=20)
            boxes = [list(map(int, r['rect'])) for r in img_regions]
            proposals[os.path.basename(img_path)] = boxes

    
    os.makedirs(os.path.dirname(Output_save), exist_ok=True)
    with open(Output_save, "w") as f:
        json.dump(proposals, f)
    print(f"Proposals save in {Output_save}")

if __name__ == "__main__":
    generate_and_save_proposals()
