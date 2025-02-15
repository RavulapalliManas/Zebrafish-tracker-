from ultralytics import YOLO
import torch

# Load the model
model = torch.load("/Users/manasvenkatasairavulapalli/Desktop/Research Work/ml/Models/yolov8m(1).pt")

# Print model summary
print(model.keys())