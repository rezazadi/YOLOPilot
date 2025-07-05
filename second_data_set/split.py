import os
from pathlib import Path
import shutil

# === Input paths ===
label_split_dirs = {
    "train": "new_labels/train",
    "val": "new_labels/val",
    "test": "new_labels/test"
}
all_images_dir = "images"  # all images are in this folder

# === Output folder name mapping ===
folder_map = {
    "train": "train_orginal",
    "val": "val_orginal",
    "test": "test_orginal"
}

# === Create output folders in YOLOv8 format ===
for folder in folder_map.values():
    os.makedirs(f"{folder}/images", exist_ok=True)
    os.makedirs(f"{folder}/labels", exist_ok=True)

# === Match images to labels and copy ===
for split, label_dir in label_split_dirs.items():
    output_folder = folder_map[split]
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        base_name = Path(label_file).stem
        label_src = Path(label_dir) / label_file
        label_dst = Path(output_folder) / "labels" / label_file

        # Copy label file
        shutil.copy(label_src, label_dst)

        # Try common image extensions
        found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            image_file = Path(all_images_dir) / f"{base_name}{ext}"
            if image_file.exists():
                image_dst = Path(output_folder) / "images" / f"{base_name}{ext}"
                shutil.copy(image_file, image_dst)
                print(f"✅ Copied image: {image_file} → {image_dst}")
                found = True
                break

        if not found:
            print(f"⚠️ Image for label '{label_file}' not found in '{all_images_dir}/'.")

print("✅ All label-matched images and labels copied to YOLOv8 folders.")
