"""
图像处理模块
"""

from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2 as cv
import config
import colorsys
import numpy as np
import random
from PIL import Image


def resize_image(image, new_size):
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
        print(c, label, 'x:{} y:{} x:{} y:{}'.format(left, top, right, bottom))

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
    my_all = ['rotate', 'flip', 'pixel', 'resize', 'mixup', 'makeup', 'noise']

    def __init__(self,
                 img: np.ndarray = None,
                 boxes: list or np.ndarray = None,
                 img_path: str = None,
                 **kwargs):
        self.img = img
        self.boxes = boxes
        if isinstance(self.boxes, list):
            self.boxes = np.asarray(self.boxes, dtype=int)
        elif boxes is None:
            self.boxes = np.array([])
        if self.boxes.shape[-1] == 4:
            self.boxes = np.concatenate([self.boxes, np.zeros(shape=(self.boxes.shape[0], 1))], axis=-1)
        self.img_path = img_path
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.augment()

    def augment(self):

        if self.img_path:
            self.img = cv.imread(self.img_path)
        if self.check_random(3):
            self.img, self.boxes = self.rotate(self.img, self.boxes, angel=self.set_random(3) * 90)
        if self.check_random(3):
            self.img, self.boxes = self.flip(self.img, self.boxes, flip_code=self.set_random(2) - 1)
        if self.check_random(2):
            self.img, self.boxes = self.pixelate(self.img, self.boxes)
        if self.check_random(4) and self.kwargs.get('img2') and self.kwargs.get('boxes2'):
            self.img, self.boxes = self.mixup(self.img, self.kwargs.get('img2'), self.boxes, self.kwargs.get('boxes2'))

        if self.check_random(1):
            self.img, self.boxes = self.resize(self.img, self.boxes,
                                               new_shape=self.kwargs.get('new_shape') or (608, 608))
        if self.check_random(3, 2):
            self.img, self.boxes = self.colors(self.img, self.boxes)
        return self.img / 255.0, self.boxes

    @staticmethod
    def check_random(num: int, target_num=1):
        """
        return True with a ratio of num * 10%
        :param num:
        :return:
        """
        if random.randint(1, num) <= target_num:
            return True
        return

    @staticmethod
    def set_random(end: int, start=0):
        return random.randint(start, end)

    @staticmethod
    def scope_random(a: int or float = 0.0,
                     b: int or float = 1.0):
        return np.random.rand() * (b - a) + a

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
            x1, y1, x2, y2, class_id = box
            rela_x0 = (x1 + x2) / float(width) / 2
            rela_y0 = (y1 + y2) / float(height) / 2
            rela_w0 = np.abs(x1 - x2) / float(width)
            rela_h0 = np.abs(y1 - y2) / float(height)

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

                result.append([new_x1, new_y1, new_x2, new_y2, class_id])

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
                result.append([new_x1, new_y1, new_x2, new_y2, class_id])

            elif aug_type == 'resize':
                new_h, new_w = kwargs.get('new_h'), kwargs.get('new_w')
                bg_h, bg_w = kwargs.get('bg_h'), kwargs.get('bg_w')

                dh = (bg_h - new_h) / 2.0
                dw = (bg_w - new_w) / 2.0

                abs_new_x0 = new_w * rela_x0
                abs_new_y0 = new_h * rela_y0
                abs_new_w0 = new_w * rela_w0
                abs_new_h0 = new_h * rela_h0

                if dh >= 0 and dw >= 0:
                    new_x1 = abs_new_x0 - abs_new_w0 / 2.0 + dw
                    new_x2 = abs_new_x0 + abs_new_w0 / 2.0 + dw
                    new_y1 = abs_new_y0 - abs_new_h0 / 2.0 + dh
                    new_y2 = abs_new_y0 + abs_new_h0 / 2.0 + dh

                elif dh < 0 and dw >= 0:
                    new_x1 = abs_new_x0 - abs_new_w0 / 2.0 + dw
                    new_x2 = abs_new_x0 + abs_new_w0 / 2.0 + dw
                    new_y1 = abs_new_y0 + dh - abs_new_h0 / 2.0
                    new_y2 = new_y1 + abs_new_h0

                elif dh >= 0 and dw < 0:
                    new_x1 = abs_new_x0 + dw - abs_new_w0 / 2.0
                    new_x2 = new_x1 + abs_new_w0
                    new_y1 = abs_new_y0 - abs_new_h0 / 2.0 + dh
                    new_y2 = abs_new_y0 + abs_new_h0 / 2.0 + dh

                else:
                    new_x1 = abs_new_x0 + dw - abs_new_w0 / 2.0
                    new_x2 = new_x1 + abs_new_w0
                    new_y1 = abs_new_y0 + dh - abs_new_h0 / 2.0
                    new_y2 = new_y1 + abs_new_h0

                new_x1 = np.max([0, new_x1])
                new_x2 = np.min([new_x2, bg_w - 1])
                new_y1 = np.max([0, new_y1])
                new_y2 = np.min([new_y2, bg_h - 1])

                result.append([new_x1, new_y1, new_x2, new_y2, class_id])

        return np.asarray(result, dtype=int)

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
                img_path = '../data/000030.jpg'
                img = cv.imread(img_path)
                new_image, new_boxes = Augment.rotate(img, angel = 90, boxes=[[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]])
                for new_box in new_boxes:
                    x1, y1, x2, y2, cls_id = new_box
                    new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """
        h, w = img.shape[:2]
        state = cv.getRotationMatrix2D(((w - 0.5) / 2.0, (h - 0.5) / 2.0), angel, 1)  # 旋转中心x,旋转中心y，旋转角度，缩放因子
        new_img = cv.warpAffine(img, state, (w, h))
        if not len(boxes):
            return new_img
        new_boxes = Augment.correct_boxes(h, w, boxes, 'rotate', angel=angel)
        return new_img, np.asarray(new_boxes, dtype=int)

    @staticmethod
    def flip(img: np.ndarray,
             boxes: list or np.ndarray = None,
             flip_code: int = 1):
        """

        :param img:
        :param boxes:
        :param flip_code:       1 left-right flip, 0 up-down flip, -1 equal to 180 degree ratation
        :return:

        example:
            img_path = 'data/000030.jpg'
            boxes = [[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]]
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
        if not len(boxes):
            return new_image, []
        new_boxes = Augment.correct_boxes(h, w, boxes, aug_type='flip', flip_code=flip_code)
        return new_image, np.asarray(new_boxes, dtype=int)

    @staticmethod
    def pixel(img: np.ndarray,
              boxes: list or np.ndarray = None,
              pixel_num=10,
              mask_ratio=0.3,
              kernel_ratio=0.1):
        """

        :param img:
        :param boxes:
        :param pixel_num:              count of pixel fields
        :param mask_ratio:              size of a pixel field(shape of square)
        :param kernel_ratio:            size of each pixel cell, a pixel field contains many pixel cell
        :return:

        this part uses pixel way for augment.
        Normally, an image includes several boxes, and each box here is divided into 【pixel_num】small fields, each
        field's size is about 【mask_ratio】 the area(a ares is a square) by one box size. And every field also needs
        to be separated into many cell, with size【kernel_ratio】of a field.

        example:
                img_path = '../data/000030.jpg'
                img = cv.imread(img_path)
                boxes = [[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]]
                a = Augment()
                new_image, new_boxes = a.pixel(img, boxes)
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """

        if not len(boxes):
            for block in range(np.random.randint(1, pixel_num)):
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
            return img, []

        for box in boxes:
            x1, y1, x2, y2, class_id = box
            box_image = img[y1:y2, x1:x2, :]
            h_, w_ = box_image.shape[:2]

            for block in range(np.random.randint(1, pixel_num)):
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
                        sub_box_image = cv.rectangle(sub_box_image, (j, i), (j + kernel_size - 1, i + kernel_size - 1),
                                                     color, -1)
                box_image[starty: endy, startx: endx, :] = sub_box_image
            img[y1:y2, x1:x2, :] = box_image
        return img, np.asarray(boxes, dtype=int)

    @staticmethod
    def noise(img, boxes, noisy_type=0):

        pass

    @staticmethod
    def mixup(img1: np.ndarray,
              img2: np.ndarray,
              boxes1: list or np.ndarray = None,
              boxes2: list or np.ndarray = None):
        """

        combine two pictures together with a certain value(128) mask
        the final shape will be the same with img1's shape

        :param img1:
        :param img2:
        :param boxes1:
        :param boxes2:
        :return:

        examples:
            img_path1 = '../data/000030.jpg'
            boxes1 = [[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]]
            img_path2 = '../data/000003.jpg'
            boxes2 = [[123,155,215,195], [239,156,307,205]]
            img1 = cv.imread(img_path1)
            img2 = cv.imread(img_path2)
            new_image, new_boxes = Augment.mixup(img1, img2, boxes1, boxes2)
            for box in new_boxes:
                x1, y1, x2, y2, class_id = box
                new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
            cv.imshow('', new_image)
            cv.waitKey()
            cv.destroyAllWindows()
        """

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img2 = cv.resize(img2, (w1, h1), interpolation=cv.INTER_CUBIC)
        ratio_x = w1 / w2
        ratio_y = h1 / h2
        img1 = Image.fromarray(img1)
        img2 = Image.fromarray(img2)
        mask = np.ones(shape=(h1, w1), dtype=np.uint8) * 128
        mask = Image.fromarray(mask)
        new_image = Image.composite(img1, img2, mask)
        new_image = np.asarray(new_image)
        if not len(boxes2) and len(boxes1):
            return new_image, []
        else:
            if isinstance(boxes1, list) or isinstance(boxes2, list):
                boxes1 = np.asarray(boxes1, dtype=int)
                boxes2 = np.asarray(boxes2, dtype=float)
            boxes2[:, 0] *= ratio_x
            boxes2[:, 2] *= ratio_x
            boxes2[:, 1] *= ratio_y
            boxes2[:, 3] *= ratio_y
            return new_image, np.concatenate([boxes1, boxes2]).astype(int)

    @staticmethod
    def makeup():
        pass

    @staticmethod
    def resize(img: np.ndarray,
               boxes: list or np.ndarray = None,
               new_shape=(608, 608)
               ):
        """

        :param img:
        :param boxes:
        :param new_shape:       yolo input shape
        :return:

        example:
                img_path = '../data/000030.jpg'
                boxes = [[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]]

                img = cv.imread(img_path)
                new_image, new_boxes = Augment.resize(img, boxes)
                for box in new_boxes:
                    x1, y1, x2, y2, class_id = box
                    new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """

        h, w = img.shape[:2]
        ratio_x, ratio_y = np.random.randint(50, 150) / 100.0, np.random.randint(50, 150) / 100.0
        new_h, new_w = np.round(h * ratio_y).astype(int), np.round(w * ratio_x).astype(int)
        new_image = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        bg_h, bg_w = new_shape
        bg_image = np.ones(shape=(bg_h, bg_w, 3), dtype=new_image.dtype) * np.random.random_integers(0, 255, size=(
            1, 1, 3)).astype(new_image.dtype)
        dh = int(round((new_h - bg_h + 0.5) // 2.0 - 1))
        dw = int(round((new_w - bg_w + 0.5) // 2.0 - 1))

        new_image_shape = new_image.shape
        bg_image_shape = bg_image.shape
        if new_h <= bg_h and new_w <= bg_w:
            bg_image[-dh:new_image_shape[0] - dh, -dw:new_image_shape[1] - dw, :] = new_image
        elif new_h > bg_h and new_w <= bg_w:
            bg_image[:, -dw:new_image_shape[1] - dw, :] = new_image[dh:bg_image_shape[0] + dh, ...]
        elif new_h <= bg_h and new_w > bg_w:
            bg_image[-dh:new_image_shape[0] - dh, ...] = new_image[:, dw:bg_image_shape[1] + dw, :]
        else:
            bg_image = new_image[dh:-dh, dw:-dw, :]
        if not len(boxes):
            return bg_image, []
        new_boxes = Augment.correct_boxes(h, w, boxes, 'resize', new_h=new_h, new_w=new_w, bg_w=bg_w, bg_h=bg_h)
        return bg_image, np.asarray(new_boxes, dtype=int)

    @staticmethod
    def colors(img, boxes=None, **kwargs):
        """

        :param img:
        :param boxes:
        :param kwargs:          mainly includes hue, sat and val
        :return:

        example:
                img_path1 = 'data/000030.jpg'
                boxes1 = [[36, 205, 180, 289, 1], [51, 160, 150, 292, 14], [295, 138, 450, 290, 14]]
                img1 = cv.imread(img_path1)
                new_image, new_boxes = Augment.colors(img1, boxes1)

                for box in new_boxes:
                    x1, y1, x2, y2, class_id = box
                    new_image = cv.rectangle(new_image, (x1, y1), (x2, y2), (0, 250, 0))
                cv.imshow('', new_image)
                cv.waitKey()
                cv.destroyAllWindows()
        """
        hue = kwargs.get('hue', 0.1)
        sat = kwargs.get('hue', 1.5)
        val = kwargs.get('hue', 1.5)
        hue = Augment.scope_random(-hue, hue)
        sat = Augment.scope_random(1, sat) if Augment.scope_random() < .5 else 1 / Augment.scope_random(1, sat)
        val = Augment.scope_random(1, val) if Augment.scope_random() < .5 else 1 / Augment.scope_random(1, val)
        x = rgb_to_hsv(np.array(img) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        new_image = hsv_to_rgb(x)
        if kwargs.get('return_back'):
            img *= 255
            new_image = np.asarray(new_image, np.uint8)
        if not len(boxes):
            boxes = []
        return new_image, boxes


    def mosaic(self, annotation_path):

        boxes = []
        images = []
        new_boxes = []
        ih, iw = (608, 608)
        new_image = np.full((ih, iw, 3), 128.)

        scale = round(self.scope_random(0.5, 0.8), 2)
        offset_x = 1 - scale
        offset_y = 1 - scale
        nw = int(iw * scale)
        nh = int(ih * scale)
        place_x = np.array([0, 0, int(iw * offset_x), int(iw * offset_x)])
        place_y = np.array([0, int(ih * offset_y), 0, int(ih * offset_y)])

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = f.readlines()
            for line in annotations:
                annotation = line.split(' ')
                img = cv.imread(annotation[0])
                box = np.array([np.array(list(map(int, box.split(',')))) for box in annotation[1:]])
                images.append(img)
                boxes.append(box)

        for i in range(4):
            img = images[i]
            box = boxes[i]
            if self.scope_random() < 0.5:
                flip_code = [-1, 0, 1]
                flip_code = flip_code[self.set_random(2)]
                img, box = self.flip(img, box, flip_code)
            if self.scope_random() < 0.5:
                angel = self.scope_random(0, 180)
                img, box = self.rotate(img, box, angel)
            if self.scope_random() < 0.5:
                pixel_num = self.set_random(10, 2)
                img, box = self.pixel(img, box, pixel_num=pixel_num, mask_ratio=0.3, kernel_ratio=0.1)
            if self.scope_random() < 0.5:
                img, box = self.colors(img, box)

            # resize
            h, w = img.shape[:2]
            img = cv.resize(img, (nw, nh), cv.INTER_CUBIC)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / w
            box[:, [1, 3]] = box[:, [1, 3]] * nh / h

            dx = place_x[i]
            dy = place_y[i]
            new_image[dy:dy + nh, dx:dx + nw, :] = img
            box[:, [0, 2]] = box[:, [0, 2]] + dx
            box[:, [1, 3]] = box[:, [1, 3]] + dy
            if i == 0:
                fx = lambda x: x if x < int(iw * offset_x) else int(iw * offset_x)
                fy = lambda x: x if x < int(ih * offset_y) else int(ih * offset_y)
                box[:, 2] = np.array(list(map(fx, box[:, 2])))
                box[:, 3] = np.array(list(map(fy, box[:, 3])))
            elif i == 1:
                fx = lambda x: x if x < int(iw * offset_x) else int(iw * offset_x)
                box[:, 2] = np.array(list(map(fx, box[:, 2])))
            elif i == 2:
                fy = lambda x: x if x < int(ih * offset_y) else int(ih * offset_y)
                box[:, 3] = np.array(list(map(fy, box[:, 3])))
            box_w = (box[:, 2] - box[:, 0]) > iw * 0.08
            box_h = (box[:, 3] - box[:, 1]) > ih * 0.08
            box = box[box_w & box_h]
            new_boxes.extend(box)
        new_image = new_image.astype(np.uint8)
        return new_image, new_boxes
