from ultralytics import YOLO

# Load the YOLO v8 model
model = YOLO("yolov8n.pt")

# 2. Train on the custom dataset
results = model.train(
    data="pothole_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="pothole_run",
    name="yolov8n_pothole",
    optimizer="AdamW", lr0=0.001
)