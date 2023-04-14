import os
import json
import numpy as np
import cv2 as cv


import making_bounding_boxes



# get the path to the JSON file
json_file_path = "data.json"

dict = {}
# read the contents of the file
with open(json_file_path) as f:
    dict = json.load(f)
 
text = 'coordinates' 
photos = os.listdir('target/')
 
# print(photos)


def IoU( pred, target): 
  #print( pred_img[0, 0]) 
  pred = np.ravel(pred) 
  target = np.ravel(target) 
  for i in range(len(pred)): 
    if pred[i] > 0.5: 
      pred[i] = 1 
    else: 
      pred[i] = 0 
 
  overlap = float(0) 
  overall = float(0) 
  for i in range(len(pred)): 
    if ( pred[i] == 1 or target[i] == 1): 
      overall += 1 
    if ( pred[i] == 1 and target[i] == 1): 
      overlap += 1 
   
  IoU = overlap / overall 
  return IoU

for photo in photos: 
    photo_name = photo[:-4]
    print(int(dict[photo_name]["number_of_boxes"]))
    for m in range( 1 ,  int(dict[photo_name]["number_of_boxes"]) + 1): 
        coordin = text + str( m )
        


        # print(photo, dict[photo_name][ coordin ].get('UpperLeft')[0]) 
        # print(photo,dict[photo_name][ coordin ]['UpperLeft'][1]) 
        # print(photo,dict[photo_name][ coordin ]['LowerRight'][0]) 
        # print(photo,dict[photo_name][ coordin ]['LowerRight'][1])