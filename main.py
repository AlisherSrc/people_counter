# import torch
# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # load a pretrained model

# # Train the model
# # model.train(data='coco128.yaml', epochs=100, imgsz=640)

# # Save the trained model
# torch.save(model.model.state_dict(), 'trained_model.pt')

# # Validate the model
# metrics = model.val()
# metrics.box.map
# metrics.box.map50
# metrics.box.map75
# metrics.box.maps

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import glob
import json

input_dir = 'test_images/'

# a = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
a = sorted(glob.glob(os.path.join(input_dir, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0][len('photo'):-2]))
print(a)
