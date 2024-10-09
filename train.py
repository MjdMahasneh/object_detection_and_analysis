from ultralytics import YOLO
import torch

# Load a model
model = YOLO("./checkpoints/YOLO11n.pt")

# check gpu availability using torch
if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'
print("using device: ", device)


# Train the model
train_results = model.train(
    data="./configs/coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device=device,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()
print('Metrics: ', metrics)