from ultralytics import YOLO
# YOLOv8 Detect models are the default YOLOv8 models, i.e. yolov8n.pt and are pretrained on COCO.
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='coco128.yaml', epochs=100, imgsz=640)

# Val:
# The validation (Val) mode is an important part of the training process in YOLO, and in fact, in any deep learning-based object detection system. 
# The main purpose of the validation mode is to evaluate the performance of the model on a separate set of images that were not used during the 
# training phase.

# During training, the model learns to identify and classify objects in the training data. However, the model may perform well on the training 
# data, but not on the new, unseen data. This is called overfitting.

# Validation mode helps to detect overfitting and fine-tune the model's hyperparameters to improve its generalization capabilities. In validation 
# mode, the model is run on a separate set of data (validation data), and its performance is evaluated using metrics such as precision, recall, 
# and F1-score. The validation metrics are used to tune the model's hyperparameters such as learning rate, batch size, and number of epochs to 
# achieve better performance on new, unseen data.

# In summary, the validation mode is a crucial step in the training process of YOLO and other deep learning-based object detection systems to 
# evaluate the model's performance on unseen data and fine-tune its hyperparameters to improve its generalization capabilities.



# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

# When you call model.val(), the method loads the validation dataset and runs inference on it using the current model parameters. 
# It then calculates various performance metrics to evaluate the model's accuracy.
# The code block you provided extracts the performance metrics from the validation process and stores them in the metrics object. 
# The metrics.box.map, metrics.box.map50, metrics.box.map75, and metrics.box.maps attributes of the metrics object contain the mean average 
# precision (mAP) values for the different evaluation settings.

# metrics.box.map: mean average precision over the intersection-over-union (IoU) threshold range 0.50 to 0.95.
# metrics.box.map50: mean average precision at IoU threshold 0.50.
# metrics.box.map75: mean average precision at IoU threshold 0.75.
# metrics.box.maps: a list of mAP values for each individual category in the dataset.


