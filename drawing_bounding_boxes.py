import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.yaml').load('yolov8n.pt')

# Load the input image
image = Image.open("test_images\person_1.jpg")

# Perform object detection on the input image
results = model(image)

# Convert the image to OpenCV format
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

print(results)

# # Loop over the results and draw bounding boxes on the image
# for pred in results.pred:
#     for result in pred:
#         tl = (int(result.xyxy[0]), int(result.xyxy[1]))
#         br = (int(result.xyxy[2]), int(result.xyxy[3]))
#         cv2.rectangle(image_cv, tl, br, (0, 255, 0), 2)

# Display the output image
# cv2.imshow("Output", image_cv)
# cv2.waitKey(0)

# # Save the output image
# cv2.imwrite("output.jpg", image_cv)