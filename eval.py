from ultralytics import YOLO
import torch

# Load a model
model = YOLO("./checkpoints/YOLO11n.pt")


# Perform object detection on an image
results = model("yoga.jpg",
                data="./configs/coco8.yaml")
results[0].show()

print('Results: ', results)


# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model
print('Exported model to: ', path)