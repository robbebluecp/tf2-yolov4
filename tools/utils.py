"""
其他通用模块
"""

import numpy as np
import tensorflow.keras.backend as K


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def nms(boxes, scores, iou_threshold, max_boxes):
    """Pure Python NMS baseline."""

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]

    index = []
    while order.size > 0:
        i = order[0]
        index.append(i)

        x_min = np.maximum(x1[i], x1[order[1:]])
        y_min = np.maximum(y1[i], y1[order[1:]])
        x_max = np.minimum(x2[i], x2[order[1:]])
        y_max = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, x_max - x_min + 1)
        h = np.maximum(0.0, y_max - y_min + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        if len(index) > max_boxes:
            break
    return index


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def iou_area_index(boxes, anchors):
    """

    :param boxes:       (N, 2) --- N x (w, h)
    :param anchors:     (N, 2) --- N x (w, h)
    :return:
    """
    boxes = np.expand_dims(boxes, -2)
    box_maxes = boxes / 2.
    box_mins = -box_maxes

    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes

    intersect_mins = np.maximum(box_mins, anchor_mins)
    intersect_maxes = np.minimum(box_maxes, anchor_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    # anchor和box的交集
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # (x, 9)
    box_area = boxes[..., 0] * boxes[..., 1]
    # (x, 9)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    # 6!
    iou = intersect_area / (box_area + anchor_area - intersect_area)

    # Find best anchor for each true box

    # (5, )
    best_anchor_index = np.argmax(iou, axis=-1)
    return best_anchor_index


def iou_cors_index(boxes, anchors):
    """

    :param boxes:       (N, 2) --- N x (x, y, w, h)
    :param anchors:     (N, 2) --- N x (x, y, w, h)
    :return:
    """
    # Expand dim to apply broadcasting.
    boxes = K.expand_dims(boxes, -2)
    boxes_xy = boxes[..., :2]
    boxes_wh = boxes[..., 2:4]
    boxes_wh_half = boxes_wh / 2.
    boxes_mins = boxes_xy - boxes_wh_half
    boxes_maxes = boxes_xy + boxes_wh_half

    # Expand dim to apply broadcasting.
    anchors = K.expand_dims(anchors, 0)
    anchors_xy = anchors[..., :2]
    anchors_wh = anchors[..., 2:4]
    anchors_wh_half = anchors_wh / 2.
    anchors_mins = anchors_xy - anchors_wh_half
    # (1, x, 2)
    anchors_maxes = anchors_xy + anchors_wh_half

    intersect_mins = K.maximum(boxes_mins, anchors_mins)
    intersect_maxes = K.minimum(boxes_maxes, anchors_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = boxes_wh[..., 0] * boxes_wh[..., 1]
    b2_area = anchors_wh[..., 0] * anchors_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


# def tf_layer_name_compat(layer_v1_name):
#     """
#     layers' name are changed a lot from tf1 to tf2,
#     and name_mapping will be filled when needed.
#
#     """
#     tmp = layer_v1_name.split('_')
#     num = tmp[-1]
#     if layer_v1_name.startswith('input'):
#         layer_v2_name = layer_v1_name
#     else:
#         try:
#             num = int(num)
#             layer_v2_name = '_'.join(tmp[:-1]) + '_' + str(num + 1)
#         except:
#             layer_v2_name = layer_v1_name + '_1'
#
#     return layer_v2_name

def tf_layer_name_compat(layer_v1_name):
    """
    layers' name are changed a lot from tf1 to tf2,
    and name_mapping will be filled when needed.
    """
    # name_mapping = {
    #     'batchnormalization': 'batch_normalization',
    #     'leakyrelu': 'leaky_re_lu',
    #     'upsampling': 'up_sampling',
    #     'zeropadding': 'zero_padding',
    #     'maxpooling': 'max_pooling',
    #
    # }
    tmp = layer_v1_name.split('_')
    num = tmp[-1]
    if layer_v1_name.startswith('input'):
        tmp_layer_name = layer_v1_name
    else:
        try:
            num = int(num)
            tmp_layer_name = '_'.join(tmp[:-1]) + '_' + str(num + 1)
        except:
            tmp_layer_name = layer_v1_name + '_1'

    # for key in name_mapping:
    #     tmp_layer_name = tmp_layer_name.replace(key, name_mapping[key])
    layer_v2_name = tmp_layer_name
    return layer_v2_name


if __name__ == '__main__':

    print(tf_layer_name_compat('batch_normalization'))
