"""
图像处理模块
"""

from .utils import rand
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv
import config
import colorsys
import numpy as np
import random


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


def draw_rectangle(image, boxes, scores, classes, class_names, colors, mode='cv'):
    if mode == 'pillow':
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
        print(c, label, 'x:{} y:{} x:{} y:{}'.format(left, top, right, bottom ))

        if mode == 'cv':
            label_size = cv.getTextSize(label, font, font_scale, thickness)
            text_width, text_height = label_size[0]

            if top - text_height >= 0:
                text_origin = np.array([left, top])
            else:
                text_origin = np.array([left, top + text_height])

            cv.rectangle(image, (left, top), (right, bottom), colors[c], thickness=2)
            cv.rectangle(image, tuple(text_origin), (left + text_width, top - text_height), colors[c], thickness=-1)
            cv.putText(image, label, tuple(text_origin), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)

        elif mode == 'pillow':

            label_size = draw.textsize(label, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

    if mode == 'pillow':
        image = np.array(image)

    return image


class Augment:

    def __init__(self):
        pass

    @staticmethod
    def correct_boxes(height, width, boxes, aug_type='rotate', **kwargs):
        """

        :param height:      image height
        :param width:       image width
        :param boxes:       boxes
        :param aug_type:    augment type
        :param kwargs:      params with different type of augment
        :return:
        """
        result = []
        boxes = np.asarray(boxes)

        w0 = (width - 0.5) / 2.0
        h0 = (height - 0.5) / 2.0
        for box in boxes:
            x1, y1, x2, y2 = box

            if aug_type == 'rotate':
                '''
                as normal, formula for Coordinate point rotation is :
                        x_new = (x - w0) * np.cos(angel) - (y - h0) * np.sin(angel) + w0
                        y_new = (x - w0) * np.sin(angel) + (y - h0) * np.cos(angel) + h0
                but in our case, the first quadrant should be changed into the forth quadrant in morphology fields.
                '''

                angel = kwargs.get('angel', 0)
                angel = angel * 2 * np.pi / 360


                fxy = lambda x, y: [(x - w0) * np.cos(angel) - (-y - -h0) * np.sin(angel) + w0,
                                    -((x - w0) * np.sin(angel) + (-y - -h0) * np.cos(angel) + -h0)]

                x11, y11 = fxy(x1, y1)
                x22, y22 = fxy(x2, y2)
                x33, y33 = fxy(x2, y1)
                x44, y44 = fxy(x1, y2)

                new_x1 = np.round(np.min([x11, x22, x33, x44])).astype(int)
                new_x2 = np.round(np.max([x11, x22, x33, x44])).astype(int)
                new_y1 = np.round(np.min([y11, y22, y33, y44])).astype(int)
                new_y2 = np.round(np.max([y11, y22, y33, y44])).astype(int)

                new_x1 = np.max([0, new_x1])
                new_x2 = np.min([width, new_x2])
                new_y1 = np.max([0, new_y1])
                new_y2 = np.min([height, new_y2])

                result.append([new_x1, new_y1, new_x2, new_y2])

            elif aug_type == 'flip':
                if kwargs.get('flip_code', 1) == 1:
                    new_x1 = width - x2
                    new_x2 = width - x1
                    new_y1 = y1
                    new_y2 = y2
                elif kwargs.get('flip_code', 0) == 0:
                    new_y1 = height - y2
                    new_y2 = height - y1
                    new_x1 = x1
                    new_x2 = x2
                elif kwargs.get('flip_code', -1) == -1:
                    new_x1 = width - x2
                    new_x2 = width - x1
                    new_y1 = height - y2
                    new_y2 = height - y1
                result.append([new_x1, new_y1, new_x2, new_y2])

        return result

    @staticmethod
    def rotate(img: np.ndarray,
               boxes: list or np.ndarray = None,
               angel: int or float = 0):
        """
        :param img:         np.ndarray type of an image
        :param angel:       counter clockwise rotation value from 0 - 360, but we suggest you use value multiple of 90 like 90, 180 and 270
        :param boxes:       boxes info
        :return:

        example:
                ima = cv.imread('xxx.jpg')
                new_image, new_boxes = Augment.rotate(img, 90, boxes=[[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]])
                for new_box in new_boxes:
                    x1, y1, x2, y2 = new_box
                    new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """
        h, w = img.shape[:2]
        state = cv.getRotationMatrix2D(((w - 0.5) / 2.0, (h - 0.5) / 2.0), angel, 1)  # 旋转中心x,旋转中心y，旋转角度，缩放因子
        new_img = cv.warpAffine(img, state, (w, h))
        if not boxes:
            return new_img
        new_boxes = Augment.correct_boxes(h, w, boxes, 'rotate', angel=angel)
        return new_img, new_boxes

    @staticmethod
    def flip(img: np.ndarray,
             boxes: list or np.ndarray = None,
             flip_code: int = 1):
        """

        :param img:
        :param boxes:
        :param flip_code:       1 left-right flip, 0 up-down flip
        :return:

        example:
            img = cv.imread('xxx.jpg')
            boxes = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]
            a = Augment()

            new_image, new_boxes = a.flip(img, boxes, 1)
            for box in new_boxes:
                x1, y1, x2, y2 = box
                new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
            cv.imshow('', new_image)
            cv.waitKey()
            cv.destroyAllWindows()
        """
        h, w = img.shape[:2]
        new_image = cv.flip(img, flip_code)
        if not boxes:
            return new_image
        new_boxes = Augment.correct_boxes(h, w, boxes, aug_type='flip', flip_code=flip_code)
        return new_image, new_boxes

    @staticmethod
    def mosaic(img: np.ndarray,
               boxes: list or np.ndarray = None,
               mosaic_num=10,
               mask_ratio=0.3,
               kernel_ratio=0.1):
        """

        :param img:
        :param boxes:
        :param mosaic_num:              count of mosaic fields
        :param mask_ratio:              size of a mosaic field(shape of square)
        :param kernel_ratio:            size of each mosaic cell, a mosaic field contains many mosaic cell
        :return:

        this part uses mosaic way for augment.
        Normally, an image includes several boxes, and each box here is divided into 【mosaic_num】small fields, each
        field's size is about 【mask_ratio】 the area(a ares is a square) by one box size. And every field also needs
        to be separated into many cell, with size【kernel_ratio】of a field.

        example:
                img = cv.imread(img_path)
                boxes = [[36, 205, 180, 289], [51, 160, 150, 292], [295, 138, 450, 290]]
                a = Augment()
                new_image, new_boxes = a.mosaic(img, boxes)
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """

        if not boxes:
            for block in range(np.random.randint(1, mosaic_num)):
                h, w = img.shape[:2]

                start_ratio_x = np.random.randint(1, 1000 * (1 - mask_ratio)) / 1000.0
                start_ratio_y = np.random.randint(1, 1000 * (1 - mask_ratio)) / 1000.0
                end_ratio_x = start_ratio_x + mask_ratio
                end_ratio_y = start_ratio_y + mask_ratio

                startx = np.floor(w * start_ratio_x).astype(int)
                starty = np.floor(h * start_ratio_y).astype(int)
                endx = np.ceil(w * end_ratio_x).astype(int)
                endy = np.ceil(h * end_ratio_y).astype(int)

                sub_img = img[starty: endy, startx: endx, :]

                base_len = min(h, w)
                kernel_size = max(np.round(base_len * kernel_ratio).astype(int), 2)
                for i in range(0, h - kernel_size, kernel_size):
                    for j in range(0, w - kernel_size, kernel_size):
                        color = img[i + kernel_size][j].tolist()
                        sub_img = cv.rectangle(sub_img, (j, i), (j + kernel_size - 1, i + kernel_size - 1),
                                                     color, -1)
                sub_img = cv.rectangle(sub_img, (0, 0), (w - 1, h - 1), (0, 250, 0))
                img[starty: endy, startx: endx, :] = sub_img
            return img

        for box in boxes:
            x1, y1, x2, y2 = box
            box_image = img[y1:y2, x1:x2, :]
            h_, w_ = box_image.shape[:2]

            for block in range(np.random.randint(1, mosaic_num)):
                start_ratio_x = np.random.randint(1, 1000 * (1 - mask_ratio)) / 1000.0
                start_ratio_y = np.random.randint(1, 1000 * (1 - mask_ratio)) / 1000.0
                end_ratio_x = start_ratio_x + mask_ratio
                end_ratio_y = start_ratio_y + mask_ratio

                startx = np.floor(w_ * start_ratio_x).astype(int)
                starty = np.floor(h_ * start_ratio_y).astype(int)
                endx = np.ceil(w_ * end_ratio_x).astype(int)
                endy = np.ceil(h_ * end_ratio_y).astype(int)

                sub_box_image = box_image[starty: endy, startx: endx, :]

                h__, w__ = sub_box_image.shape[:2]
                base_len = min(h__, w__)
                kernel_size = max(np.round(base_len * kernel_ratio).astype(int), 2)

                for i in range(0, h__ - kernel_size, kernel_size):
                    for j in range(0, w__ - kernel_size, kernel_size):
                        color = sub_box_image[i + kernel_size][j].tolist()
                        sub_box_image = cv.rectangle(sub_box_image, (j, i), (j + kernel_size - 1, i + kernel_size - 1), color, -1)
                sub_box_image = cv.rectangle(sub_box_image, (0, 0), (w__-1, h__-1), (0, 250, 0))
                box_image[starty: endy, startx: endx, :] = sub_box_image
            img[y1:y2, x1:x2, :] = box_image
        return img, boxes

    @staticmethod
    def masaic():
        pass



