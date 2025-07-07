# preprocessing.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from label_modifier import FirstDatasetLabelModifier
from label_modifier import SecondDatasetLabelModifier
from second_dataset_splitter import SecondDatasetSplitter
from second_dataset_resizer import SecondDatasetResizer

if __name__ == "__main__":

#   Label Modifcation on first dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'First_data_set')

    modifier = FirstDatasetLabelModifier(dataset_path)
    modifier.modify_all_labels()
    modifier.show_mapping_summary()
    print("All preprocessing for first dataset completed.")

#   Label Modifcation on secomd dataset
    print("Step 1: Convert CSV to YOLO labels...")

    dataset_path = os.path.join(os.path.dirname(__file__), "Second_data_set")

    modifier = SecondDatasetLabelModifier(dataset_path)
    modifier.modify_all_labels()

    print("Label conversion for second dataset completed.")

#   Splitting images of second dataset based on lables

    print("Step 2: Split images based on converted labels...")
    splitter = SecondDatasetSplitter(dataset_path)
    splitter.split_images()

#   Resizing second dataset
    print("Step 3: Resizing images and adjusting labels...")
    resizer = SecondDatasetResizer(dataset_path)
    resizer.resize_and_save()

    print("All preprocessing for second dataset completed.")
