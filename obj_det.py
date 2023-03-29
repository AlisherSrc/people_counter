# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# # Train the model
# model.train(data='coco128.yaml', epochs=100, imgsz=640)

# # # ----------------------------------------------------------------------------------------------s

from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model


# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category


# # ------------------------------------------------------------------------------

# from ultralytics import YOLO

# # Predict with the model
results = model('test_images\person_1.jpg')  # predict on an image
print(results)