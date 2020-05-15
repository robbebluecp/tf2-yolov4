import tensorflow.keras.backend as K
import tensorflow as tf
from loss import yolo_head
import numpy as np


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """

    :param box_xy:          (N, 13, 13, 3, 2)
    :param box_wh:          (N, 13, 13, 3, 2)
    :param input_shape:     (416, 416)
    :param image_shape:     (None, None)
    :return:
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    # (None, None)
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    # (N, 13, 13, 3, 4)
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # (N, 13, 13, 3, 2), (N, 13, 13, 3, 2)， (N, 13, 13, 3, 1)， (N, 13, 13, 3, 10)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # (N, 13, 13, 3, 4)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # (x, 4)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    # (x, 10)
    box_scores = K.reshape(box_scores, [-1, num_classes])
    # (x, 4), (x, 10)
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """

    :param yolo_outputs:    (N, 13, 13, 255), ...
    :param anchors:         (9, 2)
    :param num_classes:     (15, )
    :param image_shape:     (None, None)
    :param max_boxes:
    :param score_threshold:
    :param iou_threshold:
    :return:
    """
    anchors = np.asarray(anchors)
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    # (416, 416)
    anchor_mask = np.asarray(anchor_mask, dtype=int)
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # (x, 4), (x, 10)
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)


    # (x, 4)
    boxes = K.concatenate(boxes, axis=0)
    # (x, 10)
    box_scores = K.concatenate(box_scores, axis=0)


    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # (xx ,4)
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        # (xx, )
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # NMS
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        # lookup
        # (xxx, 4)
        class_boxes = K.gather(class_boxes, nms_index)
        # (xxx, )
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    # (xxxx, 4), (xxxx, ), (xxxx, )
    return boxes_, scores_, classes_
