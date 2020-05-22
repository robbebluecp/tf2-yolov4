import cv2 as cv
from tools.utils_image import Augment


while 1:
    img_path1 = 'data/000030.jpg'
    boxes1 = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]
    img_path2 = 'data/000003.jpg'
    boxes2 = [[123, 155, 215, 195], [239, 156, 307, 205]]
    img1 = cv.imread(img_path1)
    img2 = cv.imread(img_path2)
    # new_image, new_boxes = Augment(img1, boxes1, img2=img2, boxes2=boxes2)()
    new_image, new_boxes = Augment(img_path=img_path1, boxes=boxes1)()

    for box in new_boxes:
        x1, y1, x2, y2, _ = box
        new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
    cv.imshow('', new_image)
    cv.waitKey()
    cv.destroyAllWindows()
