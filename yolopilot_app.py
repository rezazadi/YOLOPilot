import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64

# Load trained YOLOv8 model
model = YOLO("second_data_set/runs/dataset2_yolov8_imgsz640_epoch50_original/weights/best.pt")

# Inference function
def detect_objects(input_image):
    img_np = np.array(input_image)
    results = model(img_np)[0]
    result_img = results.plot()
    return Image.fromarray(result_img)


# Encode logo to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

# Load logo (make sure path is correct)
logo_base64 = encode_image_to_base64("docs/images/logo.png")

# Build Gradio app with centered logo and title
with gr.Blocks() as demo:
    gr.HTML(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="{logo_base64}" alt="YOLOPilot Logo" style="width: 120px; display: block; margin: 0 auto;">
            <h1 style="margin: 10px 0;">YOLOPilot - Object Detection</h1>
            <p>Upload a road scene image to detect motobike, vehicle, pedestrian, and traffic lights.</p>
        </div>
    """)

    with gr.Row():
        input_image = gr.Image(type="pil", label="Upload Image")
        output_image = gr.Image(type="pil", label="Detected Output")

    detect_button = gr.Button("Detect")
    detect_button.click(fn=detect_objects, inputs=input_image, outputs=output_image)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_port=7861,server_name="127.0.0.1")
