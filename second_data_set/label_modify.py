import pandas as pd
import os

# Image dimensions
IMG_WIDTH = 480
IMG_HEIGHT = 300


# Desired YOLO class order (matches your class_names)
NEW_CLASSES = ["motobike", "person", "traffic_light", "vehicle"]
NEW_CLASS_IDX = {cls: i for i, cls in enumerate(NEW_CLASSES)}

# Original dataset class_id to label mapping
# 0 = car, 1 = truck, 2 = pedestrian, 3 = bicyclist, 4 = light
CLASS_MAP = {
    1: "vehicle",        # car
    2: "vehicle",        # truck
    3: "person",         # pedestrian
    4: "motobike",       # bicyclist
    5: "traffic_light"   # light
}

# Load CSV
csv_path = "labels_trainval.csv"  # ← change to labels_train.csv or labels_test.csv as needed
df = pd.read_csv(csv_path)

# Output folder
output_dir = "new_labels/trainval"  # ← change to match the dataset
os.makedirs(output_dir, exist_ok=True)

# Group by image name and write YOLO labels
for img_file, group in df.groupby("frame"):
    label_lines = []
    for _, row in group.iterrows():
        old_class = int(row["class_id"])  # ✅ ensure integer type
        new_class_name = CLASS_MAP.get(old_class)
        if new_class_name is None:
            continue  # skip unknown classes
        new_class_id = NEW_CLASS_IDX[new_class_name]

        # Convert to YOLO format
        x_center = ((row["xmin"] + row["xmax"]) / 2) / IMG_WIDTH
        y_center = ((row["ymin"] + row["ymax"]) / 2) / IMG_HEIGHT
        bbox_width = (row["xmax"] - row["xmin"]) / IMG_WIDTH
        bbox_height = (row["ymax"] - row["ymin"]) / IMG_HEIGHT

        label_lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write to .txt file (same name as image, but .txt)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    with open(os.path.join(output_dir, txt_filename), "w") as f:
        f.write("\n".join(label_lines))





# Load CSV
csv_path = "labels_train.csv"  # ← change to labels_train.csv or labels_test.csv as needed
df = pd.read_csv(csv_path)

# Output folder
output_dir = "new_labels/train"  # ← change to match the dataset
os.makedirs(output_dir, exist_ok=True)

# Group by image name and write YOLO labels
for img_file, group in df.groupby("frame"):
    label_lines = []
    for _, row in group.iterrows():
        old_class = int(row["class_id"])  # ✅ ensure integer type
        new_class_name = CLASS_MAP.get(old_class)
        if new_class_name is None:
            continue  # skip unknown classes
        new_class_id = NEW_CLASS_IDX[new_class_name]

        # Convert to YOLO format
        x_center = ((row["xmin"] + row["xmax"]) / 2) / IMG_WIDTH
        y_center = ((row["ymin"] + row["ymax"]) / 2) / IMG_HEIGHT
        bbox_width = (row["xmax"] - row["xmin"]) / IMG_WIDTH
        bbox_height = (row["ymax"] - row["ymin"]) / IMG_HEIGHT

        label_lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write to .txt file (same name as image, but .txt)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    with open(os.path.join(output_dir, txt_filename), "w") as f:
        f.write("\n".join(label_lines))




# Load CSV
csv_path = "labels_val.csv"  # ← change to labels_train.csv or labels_test.csv as needed
df = pd.read_csv(csv_path)

# Output folder
output_dir = "new_labels/trainval"  # ← change to match the dataset
os.makedirs(output_dir, exist_ok=True)

# Group by image name and write YOLO labels
for img_file, group in df.groupby("frame"):
    label_lines = []
    for _, row in group.iterrows():
        old_class = int(row["class_id"])  # ✅ ensure integer type
        new_class_name = CLASS_MAP.get(old_class)
        if new_class_name is None:
            continue  # skip unknown classes
        new_class_id = NEW_CLASS_IDX[new_class_name]

        # Convert to YOLO format
        x_center = ((row["xmin"] + row["xmax"]) / 2) / IMG_WIDTH
        y_center = ((row["ymin"] + row["ymax"]) / 2) / IMG_HEIGHT
        bbox_width = (row["xmax"] - row["xmin"]) / IMG_WIDTH
        bbox_height = (row["ymax"] - row["ymin"]) / IMG_HEIGHT

        label_lines.append(f"{new_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # Write to .txt file (same name as image, but .txt)
    txt_filename = os.path.splitext(img_file)[0] + ".txt"
    with open(os.path.join(output_dir, txt_filename), "w") as f:
        f.write("\n".join(label_lines))