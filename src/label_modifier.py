# src/label_modifier.py

import os
import pandas as pd

class FirstDatasetLabelModifier:
    ORIG_CLASSES = [
        "bike", "motobike", "person",
        "traffic_light_green", "traffic_light_orange", "traffic_light_red",
        "traffic_sign_30", "traffic_sign_60", "traffic_sign_90",
        "vehicle"
    ]

    CLASS_MAP = {
        "bike": "motobike",
        "motobike": "motobike",
        "person": "person",
        "traffic_light_green": "traffic_light",
        "traffic_light_orange": "traffic_light",
        "traffic_light_red": "traffic_light",
        "vehicle": "vehicle",
        # traffic_sign_* are ignored
    }

    NEW_CLASSES = ["motobike", "person", "traffic_light", "vehicle"]
    NEW_CLASS_IDX = {cls: i for i, cls in enumerate(NEW_CLASSES)}

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def process_label_file(self, src_path, dst_path):
        new_lines = []
        if os.path.getsize(src_path) > 0:
            with open(src_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                orig_class_id, x, y, w, h = parts
                orig_class_name = self.ORIG_CLASSES[int(orig_class_id)]
                new_class_name = self.CLASS_MAP.get(orig_class_name)
                if new_class_name is None:
                    continue
                new_class_id = self.NEW_CLASS_IDX[new_class_name]
                new_lines.append(f"{new_class_id} {x} {y} {w} {h}")

        with open(dst_path, 'w') as f_out:
            f_out.write("\n".join(new_lines))

    def modify_all_labels(self):
        for split in ["train", "valid", "test"]:
            label_old = os.path.join(self.dataset_dir, split, "labels_old")
            label_new = os.path.join(self.dataset_dir, split, "labels")
            os.makedirs(label_new, exist_ok=True)

            for file in os.listdir(label_old):
                if file.endswith(".txt"):
                    src = os.path.join(label_old, file)
                    dst = os.path.join(label_new, file)
                    self.process_label_file(src, dst)

    def show_mapping_summary(self):
        summary = pd.DataFrame({
            "Original Class": self.ORIG_CLASSES,
            "Mapped To": [self.CLASS_MAP.get(c, "(ignored)") for c in self.ORIG_CLASSES]
        })
        print(summary)



class SecondDatasetLabelModifier:
    IMG_WIDTH = 480
    IMG_HEIGHT = 300

    CLASS_MAP = {
        1: "vehicle",        # car
        2: "vehicle",        # truck
        3: "person",         # pedestrian
        4: "motobike",       # bicyclist
        5: "traffic_light"   # light
    }

    NEW_CLASSES = ["motobike", "person", "traffic_light", "vehicle"]
    NEW_CLASS_IDX = {cls: i for i, cls in enumerate(NEW_CLASSES)}

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def _convert_csv_to_labels(self, csv_name, target_folder):
        csv_path = os.path.join(self.dataset_dir, csv_name)
        df = pd.read_csv(csv_path)

        label_dir = os.path.join(self.dataset_dir, target_folder, "labels")
        os.makedirs(label_dir, exist_ok=True)

        for img_file, group in df.groupby("frame"):
            label_lines = []
            for _, row in group.iterrows():
                class_id = int(row["class_id"])
                new_class = self.CLASS_MAP.get(class_id)
                if new_class is None:
                    continue
                new_class_id = self.NEW_CLASS_IDX[new_class]

                x_center = ((row["xmin"] + row["xmax"]) / 2) / self.IMG_WIDTH
                y_center = ((row["ymin"] + row["ymax"]) / 2) / self.IMG_HEIGHT
                bbox_width = (row["xmax"] - row["xmin"]) / self.IMG_WIDTH
                bbox_height = (row["ymax"] - row["ymin"]) / self.IMG_HEIGHT

                label_lines.append(
                    f"{new_class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                )

            txt_file = os.path.splitext(img_file)[0] + ".txt"
            with open(os.path.join(label_dir, txt_file), "w") as f:
                f.write("\n".join(label_lines))

    def modify_all_labels(self):
        self._convert_csv_to_labels("labels_train.csv", "train_original")
        self._convert_csv_to_labels("labels_trainval.csv", "val_original")
        self._convert_csv_to_labels("labels_val.csv", "test_original")