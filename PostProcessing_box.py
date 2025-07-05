import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === CONFIGURATION ===
model_path = "second_data_set/runs/dataset2_yolov8_imgsz640_epoch50_original/weights/best.pt"
test_dir = "second_data_set/val_orginal/images"
label_dir = "second_data_set/val_orginal/labels"
output_dir = Path("After_transfore/eval_outputs_orginal")
prediction_vis_dir = output_dir / "visualized"
mismatch_dir = output_dir / "mismatched"
output_dir.mkdir(exist_ok=True)
prediction_vis_dir.mkdir(exist_ok=True)
mismatch_dir.mkdir(exist_ok=True)

class_names = ["motobike", "person", "traffic_light", "vehicle"]
GT_class_map = {i: name for i, name in enumerate(class_names)}

# === LOAD MODEL ===
model = YOLO(model_path)
results = model(test_dir, save=False, save_txt=False, verbose=False)

# === INIT CONFUSION MATRIX COLLECTION ===
y_true = []
y_pred = []

# === HELPER FUNCTION ===
def draw_boxes(img, boxes, color, label_prefix="", is_gt=False):
    for box in boxes:
        cls_id = int(box['cls'])
        label = f"{label_prefix}{class_names[cls_id]}"
        x1, y1, x2, y2 = map(int, box['xyxy'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# === LOOP OVER RESULTS ===
for i, result in enumerate(results):
    image_path = result.path
    label_path = os.path.join(label_dir, Path(image_path).stem + ".txt")
    if not os.path.exists(label_path):
        continue

    # Load image
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    # Load GT labels
    with open(label_path, "r") as f:
        gt_lines = f.readlines()

    gt_classes = []
    gt_boxes = []
    for line in gt_lines:
        cls, x, y, w, h = map(float, line.strip().split())
        cx, cy = x * img_w, y * img_h
        bw, bh = w * img_w, h * img_h
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        gt_classes.append(int(cls))
        gt_boxes.append({'cls': int(cls), 'xyxy': [x1, y1, x2, y2]})

    pred_boxes = []
    mismatch_flag = False

    for box in result.boxes:
        conf = float(box.conf)
        pred_cls_id = int(box.cls)

        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        pred_boxes.append({'cls': pred_cls_id, 'xyxy': [x1, y1, x2, y2]})

        # Record for confusion matrix
        for gt_cls in gt_classes:
            y_true.append(gt_cls)
            y_pred.append(pred_cls_id)

        # If prediction class not in GT, mark as mismatch
        if pred_cls_id not in gt_classes:
            mismatch_flag = True

    # Draw GT (green) and Predicted (red)
    draw_boxes(img, gt_boxes, (0, 255, 0), label_prefix="GT: ", is_gt=True)
    draw_boxes(img, pred_boxes, (0, 0, 255), label_prefix="PR: ")

    # Save visualized image
    save_path = prediction_vis_dir / Path(image_path).name
    cv2.imwrite(str(save_path), img)

    # Save mismatched image
    if mismatch_flag:
        shutil.copy(image_path, mismatch_dir / Path(image_path).name)
