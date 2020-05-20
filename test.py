import cv2 as cv
from tools.utils_image import Augment
from PIL import Image
import numpy as np

img_path = '/Users/robbe/others/tf_data/voc2007/images/000030.jpg'
boxes = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]

# img_path = '/Users/robbe/others/tf_data/voc2007/images/000001.jpg'
# boxes = [[48, 240, 195, 371], [8, 12, 352, 498]]

img = cv.imread(img_path)

a = Augment()

# new_image, new_boxes = a.rotate(img, boxes, 180)
new_image, new_boxes = a.resize(img, boxes)
for box in new_boxes:
    x1, y1, x2, y2 = box
    print(box)
    new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
# img = Image.fromarray(np.asarray(new_image, dtype=np.int))
# img.show()
cv.imshow('', new_image)
cv.waitKey()
cv.destroyAllWindows()
