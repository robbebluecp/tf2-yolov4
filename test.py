import cv2 as cv
from tools.utils_image import Augment


while 1:
    img_path1 = 'data/000030.jpg'
    boxes1 = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]

    img1 = cv.imread(img_path1)

    f = open('model_data/labels.txt')
    label_lines = f.readlines()[:]

    new_image, new_boxes = Augment(img=img1, boxes=boxes1, img_info_list=label_lines)()

    for box in new_boxes:
        x1, y1, x2, y2, _ = box
        new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
    cv.imshow('', new_image)
    cv.waitKey()
    cv.destroyAllWindows()
