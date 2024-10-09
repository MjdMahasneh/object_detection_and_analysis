from ultralytics import YOLO
import torch

## check gpu availability
print("GPU available: ", torch.cuda.is_available())

## classification
model = YOLO("./checkpoints/YOLO11n-cls.pt")
results = model(source='yoga.jpg', conf=0.3, show=True, save=True)

## detection
model = YOLO("./checkpoints/YOLO11n.pt")
results = model(source='yoga.jpg', conf=0.3, show=True, save=True)

## oriented bounding box
model = YOLO("./checkpoints/YOLO11n-obb.pt")
results = model(source='yoga.jpg', conf=0.3, show=True, save=True)

## segmentation
model = YOLO("./checkpoints/YOLO11n-seg.pt")
results = model(source='yoga.jpg', conf=0.3, show=True, save=True)

# pose estimation
model = YOLO("./checkpoints/YOLO11n-pose.pt")
results = model(source='yoga.jpg', conf=0.3, show=True, save=True)


## results is an object of type 'Results' which is a list of dictionaries containing the following keys: boxes:
## ultralytics.engine.results.Boxes object, keypoints: ultralytics.engine.results.Keypoints object, masks, names, obb,
## orig_img
print('Results: ', results)

## access boxes and keypoints
print('boxes ', results[0].boxes)
print('keypoints ', results[0].keypoints)