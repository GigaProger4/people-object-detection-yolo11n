from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
model.train(
    data="neuro_data/config.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    batch=32,
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)
