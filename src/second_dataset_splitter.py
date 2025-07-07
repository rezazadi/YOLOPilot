import os
from pathlib import Path
import shutil

class SecondDatasetSplitter:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.image_dir = Path(dataset_dir) / "images"
        self.splits = {
            "train": Path(dataset_dir) / "train_original",
            "val": Path(dataset_dir) / "val_original",
            "test": Path(dataset_dir) / "test_original"
        }

    def split_images(self):
        for split, folder_path in self.splits.items():
            label_dir = folder_path / "labels"
            image_out_dir = folder_path / "images"
            image_out_dir.mkdir(parents=True, exist_ok=True)

            for label_file in os.listdir(label_dir):
                if not label_file.endswith(".txt"):
                    continue

                base_name = Path(label_file).stem
                found = False
                for ext in [".jpg", ".jpeg", ".png"]:
                    image_file = self.image_dir / f"{base_name}{ext}"
                    if image_file.exists():
                        shutil.copy(image_file, image_out_dir / image_file.name)
                        found = True
                        break

                if not found:
                    print(f"⚠️ Image for '{label_file}' not found.")
