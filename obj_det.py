# from ultralytics import YOLO

# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# # Train the model
# model.train(data='coco128.yaml', epochs=100, imgsz=640)

# # # ----------------------------------------------------------------------------------------------s

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

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
# print(results[0].boxes.xyxy)



# Writing boundary boxes to the image ------------------------------------------------------------



image = Image.open("test_images/person_1.jpg")

image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# results[0] = first YOLO Result element of the list 
# results[0].boxes = list of all boxes on the image
# results[0].boxes.xyxy = coordinates of top left(tl) corner and bottom right(br) corner

# # Loop over the results and draw bounding boxes on the image
for result in results[0].boxes.xyxy:
    # print(result[0])
    tl = (int(result[0]), int(result[1]))
    br = (int(result[2]), int(result[3]))
    cv2.rectangle(image_cv, tl, br, (0, 255, 0), 2)

# Display the output image
cv2.imshow("Output", image_cv)
cv2.waitKey(0)

# Save the output image
cv2.imwrite("out/output.jpg", image_cv)