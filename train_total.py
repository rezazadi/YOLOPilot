import os
from pathlib import Path
from ultralytics import YOLO
import torch.multiprocessing

def train_yolov8(
    model_arch="yolov8s.pt",
    yaml_path="second_data_set/second_dataset.yaml",
    train_epochs=100,
    imgsz=416,
    batch_size=16,
    run_name="dataset1_yolov8",
    project_dir="runs/detect",
    workers=4  # Set to 0 for Windows compatibility
):
    # Create output directories
    output_dir = Path("eval_outputs")
    output_dir.mkdir(exist_ok=True)
    mismatch_dir = output_dir / "mismatched"
    mismatch_dir.mkdir(exist_ok=True)

    # Load model
    print("🔧 Training YOLOv8...")
    model = YOLO(model_arch)

    # Start training
    model.train(
        data=yaml_path,
        epochs=train_epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=run_name,
        project=project_dir,
        workers=workers,
        # No Augmentaion
        augment=False,
        mosaic=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        flipud=0.0,
        fliplr=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        mixup=0.0,
        erasing=0.0
    )


if __name__ == "__main__":
    # For Windows: set multiprocessing mode to 'spawn'
    torch.multiprocessing.set_start_method('spawn')

    # Optional: dry-run (uncomment to test training quickly)
    # train_yolov8(train_epochs=1, batch_size=2, run_name="debug_run")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    # Actual training call
    train_yolov8(
        model_arch="second_data_set/yolov8s.pt",
        yaml_path = "second_data_set/second_dataset_original.yaml",
        train_epochs=50,
        imgsz=640,
        batch_size=32,
        run_name="dataset2_yolov8_imgsz640_epoch50_original_No_TL",
        project_dir="second_data_set/runs",
        workers=4
    )
