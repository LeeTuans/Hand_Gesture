import cv2
import copy
import itertools
import csv
import numpy as np

# ================== Read labels ==================
def ReadLb_H(path):
    with open(path,
              encoding='utf-8-sig') as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [
            row[0] for row in keypoint_labels
        ]
    return keypoint_labels

# ================== Save point to csv file ==================
def SavePoint_H(key, list, path):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        for i in range(9):
            for val in list:
                writer.writerow([key, *val])
    return

# ================== Save List Landmark ==================
def FindPoint_H(img, list):
    list1 = []
    for lm in range(44):
        h, w, c = img.shape
        cx = min(int(list[lm*2-1] *w), w - 1)
        cy = min(int(list[lm*2] * h), h - 1)

        list1.append([cx, cy])

    return list1

# ================== Create list hand point ==================
def pre_process_H(list1):
    temp_list = copy.deepcopy(list1)

    # Convert to relative coordinates
    x, y = 0, 0
    for id, point in enumerate(temp_list):
        if id == 0:
            x, y = point[0], point[1]

        temp_list[id][0] = temp_list[id][0] - x
        temp_list[id][1] = temp_list[id][1] - y

    # Convert to a one-dimensional list
    temp_list = list(
        itertools.chain.from_iterable(temp_list))

    # Normalization
    max_value = max(list(map(abs, temp_list)))

    def normalize_(n):
        return n / max_value

    temp_list = list(map(normalize_, temp_list))

    return temp_list

# ====================== Convert =========================
def pre_convert (lm_list):
    list = []

    for lm in lm_list:
        list = np.append(list, lm)
    
    return list