<p align="center">
  <img src="docs/images/logo.png" alt="YOLOPilot Logo" width="150"/>
</p>

# YOLOPilot 

YOLOPilot is a road-scene object detection system based on **YOLOv8**, designed to detect **motorbikes**, **pedestrians**, **vehicles**, and **traffic lights** in real-world traffic videos and images.

The project involves training on two datasets — one labeled manually and another extracted frame-by-frame from video. It explores **transfer learning** and **training from scratch**, compares multiple approaches (resizing vs. padding), and evaluates performance using confusion matrices and F1 curves.

## Features

- YOLOv8-based object detection
- Data preprocessing with label remapping and image handling
- Transfer learning with resizing and auto-padding
- Gradio-powered web application
- Docker-ready for deployment

---

## Getting Started

### Requirements

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Main libraries used:

- ultralytics

- gradio

- opencv-python

- Pillow

- pandas

- numpy

### Datasets

This project uses two publicly available datasets for object detection in self-driving car scenarios:

1. [Object Detection - CARLA Self-Driving Car](https://www.kaggle.com/datasets/ibrahimalobaid/object-detection-carla-self-driving-car)
2. [Self-Driving Cars Dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars)

### Important:

Before training, both datasets require label formatting and structural adjustments:

- The label files must follow YOLO format (.txt) and contain only the unified class mappings used in this project.

- For details on how labels are modified and how the datasets should be structured (images, labels, labels_old, etc.), please refer to the Project Structure and Preprocessing Scripts sections above in this README.

## Usage

### Preprocessing

Run the Preprocessing script before training:

``` bash
python PreProcessing.py # Label correction of first dataset and second dataset, spliting the seocnd dataset and resizing images and correct lables of second dataset from 480x300 to 416x416 pixels
```



## Project Structure
```
YOLOPilot/
├── first_dataset/
│ ├── train
│ │ ├── images
│ │ ├── lables_old
│ │ └── lables
│ ├── valid
│ │ ├── images
│ │ ├── lables_old
│ │ └── lables
│ ├── test
│ │ ├── images
│ │ ├── lables_old
│ │ └── lables
│ └── runs → Results after training
│ 
├── second_dataset/
│ ├── train(Resized), train_orginal
│ │ ├── images
│ │ └── lables
│ ├── val(Resized), val_orginal
│ │ ├── images
│ │ └── lables
│ ├── test(Resized), test_orginal
│ │ ├── images
│ │ └── lables
│ ├── labels_train.csv, labels_val.csv
│ ├── new_labels
│ ├── images_resized
│ ├── resized_labels
│ └── runs → Results after training
│
├── src/
│ ├── label_modifier.py
│ ├── second_dataset_resizer.py
│ ├── second_dataset_splitter.py
│
├── docs/
│ ├── images/
│ │ ├── logo.png
│ │ └── app_screenshot.png
│
├── PostProcessing_box.py
├── PreProcessing.py
├── train_total.py
├── Dockerfile
├── requirements.txt
├── yolopilot_app.py
├── Dockerfile
└── README.md
```

## Train the Model

For training from scratch on either the first or second dataset, run:

```bash
python train_total.py
```

The script will use the model architecture specified by the model_arch variable (e.g. "yolov8s.pt" for YOLOv8-small).

To apply transfer learning, change the model_arch in train_yolov8.py to point to the best weights of the first dataset:

```bash
model_arch = "first_data_set/runs/dataset1_yolov8_imgsz416_epoch50/weights/best.pt"
```
Make sure you also adjust the data YAML and project output names accordingly when switching between datasets.


## Postprocessing

After training or running inference, you can use the following scripts for evaluation and inspection:

- **`PostProcessing_box.py`**: Visualizes the predicted and actual bounding boxes on all images in a dataset folder, and Evaluates the trained model on a specific dataset folder (such as `test` or `valid`) and to generate confusion matrix.

## Run the App
Start the Gradio app locally:

```bash
python yolopilot_app.py
```

### Application Screenshot

<p align="center">
  <img src="docs/images/application.png" alt="YOLOPilot App Screenshot" width="600"/>
</p>


## Deployment

Install YOLOPilot with Docker:

```bash
docker build -t yolopilot_app .
docker run -p 7861:7861 yolopilot_app
```
