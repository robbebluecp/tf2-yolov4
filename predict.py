
from tools import utils_image
import config
import tensorflow.keras as keras
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import eval
from models import Mish
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', type=str, help='input h5 model path', default='model_train/yolov4.h5')
parser.add_argument('-i', '--image', type=str, help='input image file path', default='data/test2.png')


args = parser.parse_args()
model_file_path = args.model
image_file_path = args.image
class_file_path = config.classes_path

anchors = config.anchors
class_names = config.classes_names
num_anchors = len(anchors)
num_classes = len(class_names)
class_mapping = dict(enumerate(class_names))
class_mapping = {class_mapping[key]: key for key in class_mapping}
model = keras.models.load_model(model_file_path, custom_objects={"Mish": Mish})
model.summary()


image = Image.open(image_file_path)
font = ImageFont.truetype(font=config.font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness = (image.size[0] + image.size[1]) // 300
colors = utils_image.get_random_colors(len(class_names))


new_image = utils_image.resize_image(image, config.image_input_shape)
new_image = np.array(new_image, dtype='float32')
new_image /= 255.
new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
# return ([N, 19, 19, 75], [N, 38, 38, 75], [N, 76, 76, 75])
feats = model.predict(new_image)


boxes, scores, classes = eval.yolo_eval(feats, anchors, len(class_names), (image.size[1], image.size[0]))

out_boxes, out_scores, out_classes = boxes[:5], scores[:5], classes[:5]

for i, c in reversed(list(enumerate(out_classes))):
    class_name = class_names[c]
    box = out_boxes[i]
    score = out_scores[i]

    label = '{} {:.2f}'.format(class_name, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)

    top, left, bottom, right = box
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])

    # My kingdom for a good redistributable image drawing library.
    for i in range(thickness):
        draw.rectangle(
            [left + i, top + i, right - i, bottom - i],
            outline=colors[c])
    draw.rectangle(
        [tuple(text_origin), tuple(text_origin + label_size)],
        fill=colors[c])
    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
image.show()
