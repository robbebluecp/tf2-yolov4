import config
import cv2
import numpy as np

f = open(config.label_path)
label_lines = f.readlines()


for label_line in label_lines:
    info = label_line.split()
    image_file_path, cors = info[0], info[1:]
    cors = np.array([np.array(list(map(int, box.split(',')))) for box in cors])
    image_array = cv2.imread(image_file_path)
    h, w, _ = image_array.shape
    for cor in cors:
        x1, y1, x2, y2, class_id = cor
        if x2 > w or y2 > h or (x2 - x1) > w or (x2 - x1) < 0 or (y2 - y1) > h or (y2 - y1) < 0 or class_id > 20:
            print(image_file_path)
