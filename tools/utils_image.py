"""
图像处理模块
"""

from .utils import rand
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv
import config
import colorsys


def augument(image_file_path, cors):
    '''

    :param image_file_path:
    :param cors:
    :return:
    '''

    image = Image.open(image_file_path)
    # PIL顺序为(w, h)
    # 原始大小
    rw, rh = image.size
    h, w = config.image_input_shape

    # resize image
    scale1 = w / h * rand(1 - config.jitter, 1 + config.jitter) / rand(1 - config.jitter, 1 + config.jitter)
    scale2 = rand(0.25, 2)
    # 新的大小
    if scale1 < 1:
        nh = int(scale2 * h)
        nw = int(nh * scale1)
    else:
        nw = int(scale2 * w)
        nh = int(nw / scale1)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # # 颜色微调
    # hue = rand(-config.hue, config.hue)
    # sat = rand(1, config.sat) if rand() < .5 else 1 / rand(1, config.sat)
    # val = rand(1, config.val) if rand() < .5 else 1 / rand(1, config.val)
    # x = rgb_to_hsv(np.array(image) / 255.)
    # x[..., 0] += hue
    # x[..., 0][x[..., 0] > 1] -= 1
    # x[..., 0][x[..., 0] < 0] += 1
    # x[..., 1] *= sat
    # x[..., 2] *= val
    # x[x > 1] = 1
    # x[x < 0] = 0
    # image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # 修正坐标 boxes
    cors_data = np.zeros((config.max_boxes, 5))
    if len(cors) > 0:
        np.random.shuffle(cors)
        cors[:, [0, 2]] = cors[:, [0, 2]] * nw / rw + dx
        cors[:, [1, 3]] = cors[:, [1, 3]] * nh / rh + dy
        if flip:
            cors[:, [0, 2]] = w - cors[:, [2, 0]]
        cors[:, 0:2][cors[:, 0:2] < 0] = 0
        cors[:, 2][cors[:, 2] > w] = w
        cors[:, 3][cors[:, 3] > h] = h
        box_w = cors[:, 2] - cors[:, 0]
        box_h = cors[:, 3] - cors[:, 1]
        box = cors[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > config.max_boxes:
            box = box[:config.max_boxes]
        cors_data[:len(box)] = box

    # 归一的image， 未归一的修正过的cors
    # (None, None, 3), (20, 5)
    return np.asarray(image) / 255.0, cors_data


def resize_image(image, new_size):
    iw, ih = image.size
    w, h = new_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', new_size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def resize_image_by_cv(image, new_size):
    iw, ih = image.shape[1], image.shape[0]
    w, h = new_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = cv.resize(image, (nw, nh), cv.INTER_CUBIC)

    new_image = np.full((h, w, 3), 128)
    h1 = (h - nh) // 2
    h2 = (h + nh) // 2
    w1 = (w - nw) // 2
    w2 = (w + nw) // 2

    new_image[h1:h2, w1:w2, :] = image
    return new_image


def get_random_colors(nums):
    hsv_tuples = [(x / nums, 1., 1.)
                  for x in range(nums)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    return colors


def draw_rectangle(image, boxes, scores, classes, class_names, colors, model='cv'):
    if model == 'pillow':
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        image_shape = image.size[::-1]
        thickness = (image.size[0] + image.size[1]) // 300
        font = ImageFont.truetype(font=config.font_path, size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    else:
        image_shape = image.shape
        thickness = 2
        font_scale = 1
        font = cv.FONT_HERSHEY_SIMPLEX

    for i, c in reversed(list(enumerate(classes))):
        class_name = class_names[c]
        score = scores[i]
        label = '{} {:.2f}'.format(class_name, score)

        top, left, bottom, right = boxes[i]
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image_shape[1], np.floor(right + 0.5).astype('int32'))

        if model == 'cv':
            label_size = cv.getTextSize(label, font, font_scale, thickness)
            text_width, text_height = label_size[0]

            if top - text_height >= 0:
                text_origin = np.array([left, top])
            else:
                text_origin = np.array([left, top + text_height])

            cv.rectangle(image, (left, top), (right, bottom), colors[c], thickness=2)
            cv.rectangle(image, tuple(text_origin), (left + text_width, top - text_height), colors[c], thickness=-1)
            cv.putText(image, label, tuple(text_origin), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)

        elif model == 'pillow':

            label_size = draw.textsize(label, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    if model == 'pillow':
        image = np.array(image)

    return image


class Augment:

    def __init__(self):
        pass
