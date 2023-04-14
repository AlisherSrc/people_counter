import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import glob
import json


labels = {0: 'person',
          1: 'bicycle',
          2: 'car',
          3: 'motorcycle',
          4: 'airplane',
          5: 'bus',
          6: 'train',
          7: 'truck',
          8: 'boat',
          9: 'traffic light',
          10: 'fire hydrant',
          11: 'stop sign',
          12: 'parking meter',
          13: 'bench',
          14: 'bird',
          15: 'cat',
          16: 'dog',
          17: 'horse',
          18: 'sheep',
          19: 'cow',
          20: 'elephant',
          21: 'bear',
          22: 'zebra',
          23: 'giraffe',
          24: 'backpack',
          25: 'umbrella',
          26: 'handbag',
          27: 'tie',
          28: 'suitcase',
          29: 'frisbee',
          30: 'skis',
          31: 'snowboard',
          32: 'sports ball',
          33: 'kite',
          34: 'baseball bat',
          35: 'baseball glove',
          36: 'skateboard',
          37: 'surfboard',
          38: 'tennis racket',
          39: 'bottle',
          40: 'wine glass',
          41: 'cup',
          42: 'fork',
          43: 'knife',
          44: 'spoon',
          45: 'bowl',
          46: 'banana',
          47: 'apple',
          48: 'sandwich',
          49: 'orange',
          50: 'broccoli',
          51: 'carrot',
          52: 'hot dog',
          53: 'pizza',
          54: 'donut',
          55: 'cake',
          56: 'chair',
          57: 'couch',
          58: 'potted plant',
          59: 'bed',
          60: 'dining table',
          61: 'toilet',
          62: 'tv',
          63: 'laptop',
          64: 'mouse',
          65: 'remote',
          66: 'keyboard',
          67: 'cell phone',
          68: 'microwave',
          69: 'oven',
          70: 'toaster',
          71: 'sink',
          72: 'refrigerator',
          73: 'book',
          74: 'clock',
          75: 'vase',
          76: 'scissors',
          77: 'teddy bear',
          78: 'hair drier',
          79: 'toothbrush'
          }


# Load the YOLO model
model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Define the input and output directories
input_dir = 'test_images/'
output_dir = 'output_images/'

# {
#     "photo1_y": {
#       "number_of_boxes" : "1",
#       "coordinates1": { "UpperLeft": ["0", "395"], "LowerRight": ["937", "1279"]  }
#     },
# }

dynamic_dict = {}
m = 0
# Loop over all the image files in the input directory
for filename in glob.glob(os.path.join(input_dir, '*.jpg')):

    # Load the input image
    image = Image.open(filename)

    # Perform object detection on the input image
    results = model(image)

    # Convert the image to a NumPy array and then to BGR format for OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # # Loop over the results and draw bounding boxes on the image
    cnt = 0
    sum_conf_score = 0

    upperLeft = []
    lowerRight = []
    for result in results[0].boxes:
        tl = (int(result.xyxy[0][0]), int(result.xyxy[0][1]))
        br = (int(result.xyxy[0][2]), int(result.xyxy[0][3]))
        class_label = int(result.cls[0])
        # print(int(class_label))
        if (class_label == 0):
            cnt += 1
            cv2.rectangle(image_cv, tl, br, (0, 255, 0), 2)
            cv2.putText(image_cv, str(int(result.conf[0] * 100)), (int(tl[0]), int(
                tl[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            sum_conf_score += int(result.conf[0] * 100)
            upperLeft.append([int(result.xyxy[0][0]), int(result.xyxy[0][1])])
            lowerRight.append([int(result.xyxy[0][2]), int(result.xyxy[0][3])])

    m += 1
    photoname = "photo" + str(m) + "_z"
    dynamic_dict[photoname] = {
        "number_of_boxes": cnt,
    }

    pattern = "coordinates"
    for i in range(1, cnt + 1):
        coord = pattern + str(i)
        dynamic_dict[photoname][coord] = {
            "UpperLeft": upperLeft[i - 1], "LowerRight": lowerRight[i - 1]}

    cv2.putText(image_cv, "Number of people: " + str(cnt),
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    # Save the output image
    output_filename = os.path.join(output_dir, os.path.basename(filename))
    cv2.imwrite(output_filename, image_cv)

with open("target.json", 'w') as f:
    json.dump(dynamic_dict, f)
print(sum_conf_score/cnt)

# "coordinates1": {"UpperLeft": [result.xyxy[0][0], result.xyxy[0][1]], "LowerRight": [result.xyxy[0][2], result.xyxy[0][3]]}
