import cv2 as cv
from tools.utils_image import Augment


img_path = '/Users/robbe/others/tf_data/voc2007/images/000030.jpg'
img = cv.imread(img_path)
boxes = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]
a = Augment()

new_image = cv.resize(img, (200, 400), interpolation=cv.INTER_CUBIC)

#
a.resize(img)
# cv.imshow('', new_image)
# cv.waitKey()
# cv.destroyAllWindows()


# for box in new_boxes:
#     x1, y1, x2, y2 = box
#     new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
# cv.imshow('', new_image)
# cv.waitKey()
# cv.destroyAllWindows()

