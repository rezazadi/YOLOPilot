import os
import pandas as pd

# Define original and new class mappings
ORIG_CLASSES = [
    "bike", "motobike", "person",
    "traffic_light_green", "traffic_light_orange", "traffic_light_red",
    "traffic_sign_30", "traffic_sign_60", "traffic_sign_90",
    "vehicle"
]

CLASS_MAP = {
    "bike":        "motobike",
    "motobike":    "motobike",
    "person":      "person",
    "traffic_light_green":  "traffic_light",
    "traffic_light_orange": "traffic_light",
    "traffic_light_red":    "traffic_light",
    # 🚫 traffic_sign_* are now excluded completely
    "vehicle":      "vehicle",
}

NEW_CLASSES = ["motobike", "person", "traffic_light", "vehicle"]
NEW_CLASS_IDX = {cls: i for i, cls in enumerate(NEW_CLASSES)}
ORIG_CLASS_IDX = {cls: i for i, cls in enumerate(ORIG_CLASSES)}

def process_label_file(src_path, dst_path):
    """
    Convert YOLO label file from original classes to merged class mapping.
    Always writes a label file, even if empty.
    """
    new_lines = []
    if os.path.getsize(src_path) > 0:  # File is not empty
        with open(src_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            orig_class_id, x, y, w, h = parts
            orig_class_name = ORIG_CLASSES[int(orig_class_id)]
            new_class_name = CLASS_MAP.get(orig_class_name)
            if new_class_name is None:
                continue  # skip traffic_sign_* and others not in CLASS_MAP
            new_class_id = NEW_CLASS_IDX[new_class_name]
            new_line = f"{new_class_id} {x} {y} {w} {h}"
            new_lines.append(new_line)

    # Write even if empty
    with open(dst_path, 'w') as f_out:
        f_out.write("\n".join(new_lines))

# Process labels inside each dataset folder directly
base_dirs = ["train", "valid", "test"]

for base_dir in base_dirs:
    label_dir = os.path.join(base_dir, "labels_old")
    new_label_dir = os.path.join(base_dir, "labels")
    os.makedirs(new_label_dir, exist_ok=True)

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            src = os.path.join(label_dir, file)
            dst = os.path.join(new_label_dir, file)
            process_label_file(src, dst)

# Generate and display class mapping summary
summary = pd.DataFrame({
    "Original Classes": ORIG_CLASSES,
    "Mapped To": [CLASS_MAP.get(c, "(ignored)") for c in ORIG_CLASSES]
})

