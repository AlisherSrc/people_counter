import os
import json
import numpy as np
import cv2 as cv


# get the path to the JSON file
json_file_path = "data.json"

dict = {}
# read the contents of the file
with open(json_file_path) as f:
    dict = json.load(f)

text = 'coordinates'
photos = os.listdir('target/')

# print(photos)


def iou(rect1, rect2):
    """
    Calculate Intersection over Union (IoU) between two rectangles.
    Rectangles are defined by their top-left and bottom-right corners.

    Args:
        rect1: Tuple (x1, y1, x2, y2) representing the top-left and bottom-right corners of rectangle 1.
        rect2: Tuple (x1, y1, x2, y2) representing the top-left and bottom-right corners of rectangle 2.

    Returns:
        Float value representing the IoU between the two rectangles.
    """
    # Calculate intersection area
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union area
    area_rect1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    area_rect2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
    union_area = area_rect1 + area_rect2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    # This formula adds a small constant (0.1) to the denominator to avoid division by zero errors.
    # The effect of this formula is that the IoU will be lower when there are more people in the image,
    # and when the prediction rectangle is much larger or smaller than the target rectangle.

    # iou = intersection_area / (area_rect1 + area_rect2 - intersection_area + 0.1 * (area_rect1 + area_rect2))

    return iou


def area_of_rectangle(top_left, bottom_right):
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    area = width * height
    return area


def real_iou(coords_data, coords_target):
    # coords_data - data photo coordinates
    # coords_target - target photo coordinates

    # we need to use follwoing formula in order to solve nested rect issue
    # big IoU small rect is prefferable than big IoU big rect
    # IoU / Area of rectangle | we apply this formula for each rect and retrieve the max one

    # if there is no single person on the image and prediction found no people as well then we make IoU = 1
    # if there are people on the image

    if len(coords_target) == 0 and len(coords_data) == 0:
        print(coords_target, coords_data)
        return 1
    elif len(coords_target) == 0 or len(coords_data) == 0:
        if len(coords_target) != 0:
            return (1 / len(coords_target)) * 0.5
        if len(coords_data) != 0:
            return (1 / len(coords_data)) * 0.5

    overall_real_iou = 0

    for i in range(len(coords_data)):
        max_value_iou = -1
        real_iou = 0

        for j in range(len(coords_target)):
            data_coord = coords_data[i]
            target_coord = coords_target[j]

            # print(data_coord, target_coord)
            curr_iou = iou(data_coord, target_coord)
            area = area_of_rectangle([target_coord[0],target_coord[1]],[target_coord[2],target_coord[3]])
            # here we use this formula to choose right rectangle
            value_iou = curr_iou 

            if value_iou > max_value_iou:
                # we need to save value for IoU to figure out which IoU to sum to overall IoU
                max_value_iou = value_iou
                real_iou = curr_iou
                



        overall_real_iou += real_iou
    #  Give penalty for each extra person in the prediction
    if len(coords_data) != len(coords_target):
        print("penalty!")
        subtr_abs = abs(len(coords_data) - len(coords_target)) 
        print("Before:" + str(overall_real_iou))
        for _ in range(subtr_abs):
            overall_real_iou -= overall_real_iou * 0.1
        # overall_real_iou /= subtr_abs
    return overall_real_iou / len(coords_data)




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
#         # Iterate over boxes in targ et dict
#         for i in range(1, int(target_dict[photo_name_target]["number_of_boxes"]) + 1):
#             coord_name = "coordinates" + str(i)
#             try:
#                 target_coords.append([target_dict[photo_name_target][coord_name]["UpperLeft"][0], target_dict[photo_name_target][coord_name]["UpperLeft"][1],
#                                       target_dict[photo_name_target][coord_name]["LowerRight"][
#                                           0], target_dict[photo_name_target][coord_name]["LowerRight"][1]
#                                       ])
#             except:
#                 print("out of bounds in the target!")

#             try:
#                 data_coords.append([data_dict[photo_name_target][coord_name]["UpperLeft"][0], data_dict[photo_name_target][coord_name]["UpperLeft"][1],
#                                     data_dict[photo_name_target][coord_name]["LowerRight"][
#                                         0], data_dict[photo_name_target][coord_name]["LowerRight"][1]
#                                     ])
#             except:
#                 print("out of bounds in the data!")
#   elif (int(target_dict[photo_name_target]["number_of_boxes"]) + 1 == 0):
#         for i in range(1, int(data_coords[photo_name_target]["number_of_boxes"]) + 1):
#             coord_name = "coordinates" + str(i)
#             try:
#                 target_coords.append([target_dict[photo_name_target][coord_name]["UpperLeft"][0], target_dict[photo_name_target][coord_name]["UpperLeft"][1],
#                                       target_dict[photo_name_target][coord_name]["LowerRight"][
#                                           0], target_dict[photo_name_target][coord_name]["LowerRight"][1]
#                                       ])
#             except:
#                 print("out of bounds in the target!")

#             try:
#                 data_coords.append([data_dict[photo_name_target][coord_name]["UpperLeft"][0], data_dict[photo_name_target][coord_name]["UpperLeft"][1],
#                                     data_dict[photo_name_target][coord_name]["LowerRight"][
#                                         0], data_dict[photo_name_target][coord_name]["LowerRight"][1]
#                                     ])
#             except:
#                 print("out of bounds in the data!")

#   curr_image_iou = real_iou(data_coords, target_coords)
#   print(curr_image_iou)

    # # Iterate over boxes in data dict
    # for i in range(1, int(data_dict[photo_name_data]["number_of_boxes"]) + 1):
    #     coord_name = "coordinates" + str(i)

    #     print(coord_name, "UpperLeft in data dict:", data_dict[photo_name_data][coord_name]["UpperLeft"])
    #     print(coord_name, "LowerRight in data dict:", data_dict[photo_name_data][coord_name]["LowerRight"])


# print(photo, dict[photo_name][ coordin ].get('UpperLeft')[0])
# print(photo,dict[photo_name][ coordin ]['UpperLeft'][1])
# print(photo,dict[photo_name][ coordin ]['LowerRight'][0])
# print(photo,dict[photo_name][ coordin ]['LowerRight'][1])
