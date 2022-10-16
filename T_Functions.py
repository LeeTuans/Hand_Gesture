import cv2
import copy
import itertools
import csv
import numpy as np

# ================== Read labels ==================
def ReadLb (path):
    with open(path,
              encoding='utf-8-sig') as f:
        keypoint_labels = csv.reader(f)
        keypoint_labels = [
            row[0] for row in keypoint_labels
        ]
    return keypoint_labels

# ================== Save point to csv file ==================
def SavePoint(key, list2):
    if int(key) >= 97 & int(key) <= 122:
        with open('./model/Hand_Point/HandPoint.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([int(key)-97, *list2])
    return

# ================== Save List Landmark ==================
def FindPoint(img, result):
    list1 = []
    for lm in range(21):
        h, w, c = img.shape
        cx = min(int(result.landmark[lm].x * w), w - 1)
        cy = min(int(result.landmark[lm].y * h), h - 1)

        list1.append([cx, cy])

    return list1

# ================== Calc bounding rect ==================
def CalcBR(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x-17, y-17, x + w+17, y + h+17]

# ================== Create list hand point ==================
def pre_process(list1, index):
    temp_list = copy.deepcopy(list1)

    # Convert to relative coordinates
    x, y = 0, 0
    for id, point in enumerate(temp_list):
        if id == 0:
            x, y = point[0], point[1]

        if index == 1:
            temp_list[id][0] = temp_list[id][0] - x
        else:
            temp_list[id][0] = x - temp_list[id][0]
            
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

# ================== Draw Bounding Rect ==================
def DrawBR(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (121, 22, 76), 2)

    return image

# ================== Draw info text ==================
def DrawIT(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0]-1, brect[1]), (brect[2]+1, brect[1] - 22),
                 (121, 22, 76), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ' : ' + hand_sign_text
    else:
        info_text = info_text + ' : No'
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image