import os
import json
import numpy as np
import cv2 as cv

from metrics import real_iou


# target_dict = {}
# data_dict = {}

# with open('target.json', 'r') as f:
#     target_dict = json.load(f)

# with open('data.json', 'r') as f:
#     data_dict = json.load(f)

# # we need two arrays of arrays of coordinates

# for photo_name_target, photo_name_data in zip(target_dict.keys(), data_dict.keys()):
#   print("Photo name:", photo_name_target)
#   print("Number of boxes in target dict:",
#         target_dict[photo_name_target]["number_of_boxes"])
#   print("Number of boxes in data dict:",
#         data_dict[photo_name_data]["number_of_boxes"])

#   target_coords = []
#   data_coords = []

#   if (int(target_dict[photo_name_target]["number_of_boxes"]) + 1 != 0):
#         # Iterate over boxes in target dict
#         for i in range(1, int(target_dict[photo_name_target]["number_of_boxes"]) + 1):
#             coord_name = "coordinates" + str(i)
#             try:
#                 target_coords.append([int(target_dict[photo_name_target][coord_name]["UpperLeft"][0]), int(target_dict[photo_name_target][coord_name]["UpperLeft"][1]),
#                                       int(target_dict[photo_name_target][coord_name]["LowerRight"][0]), int(target_dict[photo_name_target][coord_name]["LowerRight"][1])
#                                       ])
#             except KeyError:
#                 print(f"KeyError: Could not find {coord_name} in the dictionary")

#             try:
#                 data_coords.append([int(data_dict[photo_name_data][coord_name]["UpperLeft"][0]), int(data_dict[photo_name_data][coord_name]["UpperLeft"][1]),
#                                     int(data_dict[photo_name_data][coord_name]["LowerRight"][0]), int(data_dict[photo_name_data][coord_name]["LowerRight"][1])
#                                     ])
#             except KeyError:
#                 print(f"KeyError: Could not find {coord_name} in the dictionary")
#   elif (int(target_dict[photo_name_target]["number_of_boxes"]) + 1 == 0):
#         for i in range(1, int(data_dict[photo_name_data]["number_of_boxes"]) + 1):
#             coord_name = "coordinates" + str(i)
#             try:
#                 target_coords.append([int(target_dict[photo_name_target][coord_name]["UpperLeft"][0]), int(target_dict[photo_name_target][coord_name]["UpperLeft"][1]),
#                                       int(target_dict[photo_name_target][coord_name]["LowerRight"][0]), int(target_dict[photo_name_target][coord_name]["LowerRight"][1])
#                                       ])
#             except KeyError:
#                 print(f"KeyError: Could not find {coord_name} in the dictionary")

#             try:
#                 data_coords.append([int(data_dict[photo_name_data][coord_name]["UpperLeft"][0]), int(data_dict[photo_name_data][coord_name]["UpperLeft"][1]),
#                                     int(data_dict[photo_name_data][coord_name]["LowerRight"][0]), int(data_dict[photo_name_data][coord_name]["LowerRight"][1])
#                                     ])
#             except KeyError:
#                 print(f"KeyError: Could not find {coord_name} in the dictionary")

#   curr_image_iou = real_iou(data_coords, target_coords)
#   print(curr_image_iou)

import json

target_dict = {}
data_dict = {}

with open('target.json', 'r') as f:
    target_dict = json.load(f)

with open('data.json', 'r') as f:
    data_dict = json.load(f)

# we need two arrays of arrays of coordinates

for photo_name_target, photo_name_data in zip(target_dict.keys(), data_dict.keys()):
    print("Photo name:", photo_name_target)
    print("Number of boxes in target dict:", target_dict[photo_name_target]["number_of_boxes"])
    print("Number of boxes in data dict:", data_dict[photo_name_data]["number_of_boxes"])

    target_coords = []
    data_coords = []

    if int(target_dict[photo_name_target]["number_of_boxes"]) > 0:
        # Iterate over boxes in target dict
        for i in range(1, int(target_dict[photo_name_target]["number_of_boxes"]) + 1):
            coord_name = "coordinates" + str(i)
            try:
                target_coords.append([int(target_dict[photo_name_target][coord_name]["UpperLeft"][0]),
                                      int(target_dict[photo_name_target][coord_name]["UpperLeft"][1]),
                                      int(target_dict[photo_name_target][coord_name]["LowerRight"][0]),
                                      int(target_dict[photo_name_target][coord_name]["LowerRight"][1])])
            except:
                print("out of bounds in the target!")

            try:
                data_coords.append([int(data_dict[photo_name_data][coord_name]["UpperLeft"][0]),
                                    int(data_dict[photo_name_data][coord_name]["UpperLeft"][1]),
                                    int(data_dict[photo_name_data][coord_name]["LowerRight"][0]),
                                    int(data_dict[photo_name_data][coord_name]["LowerRight"][1])])
            except:
                print("out of bounds in the data!")
    else:
        for i in range(1, int(data_dict[photo_name_data]["number_of_boxes"]) + 1):
            coord_name = "coordinates" + str(i)
            try:
                target_coords.append([int(target_dict[photo_name_target][coord_name]["UpperLeft"][0]),
                                      int(target_dict[photo_name_target][coord_name]["UpperLeft"][1]),
                                      int(data_dict[photo_name_data][coord_name]["LowerRight"][0]),
                                      int(data_dict[photo_name_data][coord_name]["LowerRight"][1])
                                      ])
            except:
                print("out of bounds in the target!")

            try:
                data_coords.append([int(data_dict[photo_name_data][coord_name]["UpperLeft"][0]),
                                    int(data_dict[photo_name_data][coord_name]["UpperLeft"][1]),
                                    int(data_dict[photo_name_data][coord_name]["LowerRight"][0]),
                                    int(data_dict[photo_name_data][coord_name]["LowerRight"][1])
                                        ])
            except:
                print("out of bounds in the data!")

    print(data_coords,target_coords)
    curr_image_iou = real_iou(data_coords, target_coords)
    curr_image_iou = max(min(curr_image_iou, 1.0), 0.0) * 100
    print(curr_image_iou)
