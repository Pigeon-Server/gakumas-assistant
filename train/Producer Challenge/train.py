from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

# Train the model with the two most idle GPUs
results = model.train(
    epochs=300,
    batch=148,
    data="data.yaml",
    device=[0,1,2,3],
    #device=0,
    cache="disk",
    save_period=10,
    workers=32,
    imgsz=640,
    multi_scale=0.5
)
