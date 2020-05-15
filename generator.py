"""
generator
"""

import numpy as np
from tools import utils_image, utils
import config


def preprocess_true_boxes(true_boxes: np.ndarray,
                          image_hw: np.ndarray,
                          input_shape,
                          anchors,
                          num_classes):
    """

    :param true_boxes:      (N, max_box, 5)
    :param input_shape:     (416, 416)
    :param anchors:         (9, 2)
    :param num_classes:     ...
    :return:
    """
    assert (true_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors) // 3  # default setting

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    image_hw = np.expand_dims(image_hw, 1)
    # 中心点, (N, 20, 2)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # 宽高,   (N, 20, 2)
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    # 剔除0项
    valid_mask = boxes_wh[..., 0] > 0
    # 放缩
    # 从这里开始，对角坐标值被个替换成中心坐标值 + wh, 归一
    true_boxes[..., 0:2] = boxes_xy / image_hw[::-1]
    true_boxes[..., 2:4] = boxes_wh / image_hw[::-1]

    batch_size = true_boxes.shape[0]
    # [13, 26, 52]
    grid_shapes = [input_shape // config.scale_size[l] for l in range(num_layers)]
    # [(N, 13, 13, 3, 15), (N, 26, 26, 3, 15), (N, 52, 52, 3, 15)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(config.anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    # 开始挑选最优anchor
    for N in range(batch_size):
        # Discard zero rows， 因为默认20个zeors_like, 一般都不够20的，有很多0填充项
        # (x, 2)
        wh = boxes_wh[N, valid_mask[N]]
        if len(boxes_wh) == 0:
            continue

        # 对于每个box，从9个anchor中，选出最佳anchor
        # (x, )
        best_anchor_indexes = utils.iou_area_index(wh, anchors)

        for t, n in enumerate(best_anchor_indexes):
            for l in range(num_layers):
                if n in config.anchor_mask[l]:
                    # 中点y
                    i = np.floor(true_boxes[N, t, 0] * grid_shapes[l][1]).astype('int32')
                    # 中点x
                    j = np.floor(true_boxes[N, t, 1] * grid_shapes[l][0]).astype('int32')
                    #
                    k = config.anchor_mask[l].index(n)
                    c = true_boxes[N, t, 4].astype('int32')
                    y_true[l][N, j, i, k, 0:4] = true_boxes[N, t, 0:4]
                    y_true[l][N, j, i, k, 4] = 1
                    y_true[l][N, j, i, k, 5 + c] = 1
    # [(N, 13, 13, 3, 15), (N, 26, 26, 3, 15), (N, 52, 52, 3, 15)]
    return y_true


def data_generator(label_lines, batch_size, input_shape, anchors, num_classes):
    """

    :param annotation_lines:    /xxx/VOCdevkit/VOC2007/JPEGImages/000017.jpg 185,62,279,199,14 90,78,403,336,12
    :param batch_size:          假设：32
    :param input_shape:         (416, 416)
    :param anchors:             9 x 2
    :param num_classes:         假设：10
    :return:
    """

    n = len(label_lines)
    i = 0
    while True:
        image_data = []
        image_hw = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(label_lines)

            # 处理数据
            label_line = label_lines[i]
            info = label_line.split()
            image_file_path, cors = info[0], info[1:]
            cors = np.array([np.array(list(map(int, box.split(',')))) for box in cors])


            # 数据增强
            # (None, None, 3), (20, 5)
            image, box = utils_image.augument(image_file_path, cors)
            image_h, image_w = image.shape[:2]

            image_hw.append((image_h, image_w))
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        image_hw = np.array(image_hw)
        # (N, 20, 5), (None, None), (9, 2), 10
        # return [N, 13, 13, 3, 15]
        y_true = preprocess_true_boxes(box_data, image_hw, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

if __name__ == '__main__':
    data_generator(['/Users/robbe/codes/keras-yolov4-core/data/000017.jpg 185,62,279,199,14 90,78,403,336,12'], 1, input_shape=(608, 608), anchors=config.anchors, num_classes=config.num_classes)