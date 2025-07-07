import os
import cv2
from pathlib import Path

class SecondDatasetResizer:
    def __init__(self, dataset_dir, target_size=416, orig_w=480, orig_h=300):
        self.dataset_dir = Path(dataset_dir)
        self.target_size = target_size
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.scale_x = target_size / orig_w
        self.scale_y = target_size / orig_h

        self.splits = ["train", "val", "test"]

    def resize_and_save(self):
        for split in self.splits:
            src_img_dir = self.dataset_dir / f"{split}_original" / "images"
            src_lbl_dir = self.dataset_dir / f"{split}_original" / "labels"

            dst_img_dir = self.dataset_dir / split / "images"
            dst_lbl_dir = self.dataset_dir / split / "labels"

            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            for label_file in os.listdir(src_lbl_dir):
                if not label_file.endswith(".txt"):
                    continue

                # Image path
                image_name = Path(label_file).stem + ".jpg"
                input_image_path = src_img_dir / image_name
                output_image_path = dst_img_dir / image_name

                if not input_image_path.exists():
                    print(f"⚠️ Missing image: {input_image_path}")
                    continue

                # Resize image
                img = cv2.imread(str(input_image_path))
                if img is None:
                    print(f"⚠️ Failed to load: {input_image_path}")
                    continue
                resized_img = cv2.resize(img, (self.target_size, self.target_size))
                cv2.imwrite(str(output_image_path), resized_img)

                # Resize label
                input_label_path = src_lbl_dir / label_file
                output_label_path = dst_lbl_dir / label_file

                with open(input_label_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, x, y, w, h = map(float, parts)

                    # Unnormalize from 480x300
                    x_abs = x * self.orig_w
                    y_abs = y * self.orig_h
                    w_abs = w * self.orig_w
                    h_abs = h * self.orig_h

                    # Rescale and normalize to 416x416
                    x_new = (x_abs * self.scale_x) / self.target_size
                    y_new = (y_abs * self.scale_y) / self.target_size
                    w_new = (w_abs * self.scale_x) / self.target_size
                    h_new = (h_abs * self.scale_y) / self.target_size

                    new_lines.append(f"{int(cls)} {x_new:.6f} {y_new:.6f} {w_new:.6f} {h_new:.6f}")

                with open(output_label_path, "w") as f:
                    f.write("\n".join(new_lines))
