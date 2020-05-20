from tools import utils_image
import config
import cv2 as cv
import numpy as np
import eval
import models
import argparse
import tensorflow as tf

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, help='input h5 model path', default='model_train/ep027-loss18.731-valloss19.110.h5')
parser.add_argument('-i', '--image', type=str, help='input image file path', default='data/dog.jpg')


args = parser.parse_args()
model_file_path = args.model
image_file_path = args.image
class_file_path = config.classes_path


anchors = config.anchors
class_names = config.classes_names

num_anchors = len(anchors)
num_classes = len(class_names)
class_mapping = dict(enumerate(class_names))
colors = utils_image.get_random_colors(len(class_names))
class_mapping = {class_mapping[key]: key for key in class_mapping}
model = models.YOLO()()
model.load_weights('model_train/yolov4.h5')



image = cv.imread(image_file_path)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


new_image = utils_image.resize_image_by_cv(image, config.image_input_shape)
new_image = np.array(new_image, dtype='float32')
new_image /= 255.
new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
# return ([N, 19, 19, 75], [N, 38, 38, 75], [N, 76, 76, 75])
feats = model.predict(new_image)


boxes, scores, classes = eval.yolo_eval(feats, anchors, len(class_names), (image.shape[0], image.shape[1]))
out_boxes, out_scores, out_classes = boxes[:5], scores[:5], classes[:5]


image = utils_image.draw_rectangle(image, boxes, scores, classes, class_names, colors, mode='pillow')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cv.namedWindow("img", cv.WINDOW_NORMAL)
cv.imshow('img', image)
cv.waitKey()
