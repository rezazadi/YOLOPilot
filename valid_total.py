from ultralytics import YOLO
import os

model_path = "second_data_set/runs/dataset1_yolov8_imgsz416_epoch50_resize/weights/best.pt"
data_yaml = "second_data_set/second_dataset.yaml"
output_dir = "After_transfore/dataset1_yolov8_imgsz416_epoch50_resize"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# Run evaluation
results = model.val(data=data_yaml, split="val", save=True, plots=True)

# Print out key metrics
print("📊 Validation Results Summary:")
print("Metrics Dictionary:", results.results_dict)
print("Class Names:", results.names)

# Try to save confusion matrix if it exists
conf_matrix = results.confusion_matrix
if conf_matrix and conf_matrix.matrix is not None:
    conf_matrix.plot(save_dir=output_dir)
    print("✅ Official-style confusion matrix saved to", output_dir)
else:
    print("⚠️ No confusion matrix could be generated.")
