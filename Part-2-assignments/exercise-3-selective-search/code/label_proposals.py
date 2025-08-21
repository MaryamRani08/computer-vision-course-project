import os
import json
import numpy as np
from pycocotools.coco import COCO

Data_dir = "data"
Proposal_File = "results/balloon_proposals.json"
Data_Splits = ["train", "valid"]
Tp_Threshold = 0.75
Tn_Threshold = 0.25
Output_Save = "results/labeled_proposals.json"

def compute_intersectin_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    intersec_over_union = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return intersec_over_union

def label_proposals():
    with open(Proposal_File, 'r') as f:
        proposals = json.load(f)

    labeled_data = []

    for split in Data_Splits:
        annotat_file = os.path.join(Data_dir, split, "_annotations.coco.json")
        coco = COCO(annotat_file)

        for img_id in coco.imgs:
            img_info = coco.imgs[img_id]
            file_n = img_info['file_name']
            gt_ann_ids = coco.getAnnIds(imgIds=img_id)
            gt_anns = coco.loadAnns(gt_ann_ids)
            gt_boxes = [ann['bbox'] for ann in gt_anns]

            if file_n not in proposals:
                continue

            for box in proposals[file_n]:
                ious = [compute_intersectin_over_union(box, gt_box) for gt_box in gt_boxes]
                maximum_iou = max(ious) if ious else 0

                if maximum_iou >= Tp_Threshold:
                    label = 1
                elif maximum_iou <= Tn_Threshold:
                    label = 0
                else:
                    continue  # not accept ambiguous samples

                labeled_data.append({
                    "file_name": file_n,
                    "box": box,
                    "label": label
                })

    with open(Output_Save, "w") as f:
        json.dump(labeled_data, f)
    print(f"Saved Labeled Proposals to: {Output_Save}")

if __name__ == "__main__":
    label_proposals()
