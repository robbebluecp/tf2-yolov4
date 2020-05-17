import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def nms(boxes, scores, iou_threshold, max_boxes):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]

    index = []
    while order.size > 0:
        i = order[0]
        index.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        x_min = np.maximum(x1[i], x1[order[1:]])
        y_min = np.maximum(y1[i], y1[order[1:]])
        x_max = np.minimum(x2[i], x2[order[1:]])
        y_max = np.minimum(y2[i], y2[order[1:]])

        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, x_max - x_min + 1)
        h = np.maximum(0.0, y_max - y_min + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= iou_threshold)[0]
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
        if len(index) > max_boxes:
            break
    return index


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """

        :param feats:           (N, 13, 13, 3 * (5+n_class)), ...
        :param anchors:         (3, 2)
        :param num_classes:     15
        :param input_shape:     (416, 416)
        :param calc_loss:
        :return:
        """

    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    # (1, 1, 1, 3, 2)
    anchors_tensor = np.reshape(np.array(anchors), [1, 1, 1, num_anchors, 2])
    # (13, 13)
    grid_shape = np.array(feats.shape[1:3])  # height, width
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                     [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                     [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y], axis=-1)
    # (13, 13, 1, 2)
    grid = grid.astype(feats.dtype)

    # (N, 13, 13, 3 * 15)
    feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = (sigmoid(feats[..., :2]) + grid) / grid_shape[..., ::-1].astype(feats.dtype)
    box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[..., ::-1].astype(feats.dtype)
    box_confidence = sigmoid(feats[..., 4:5])
    box_class_probs = sigmoid(feats[..., 5:])

    if calc_loss == True:
        # (13, 13, 1, 2), (N, 13, 13, 3, 15), (N, 13, 13, 3, 2), (N, 13, 13, 3, 2)
        return grid, feats, box_xy, box_wh
    # (N, 13, 13, 3, 2), (N, 13, 13, 3, 2), (N, 13, 13, 3, 1), (N, 13, 13, 3, 10)
    return box_xy, box_wh, box_confidence, box_class_probs


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
    input_shape = input_shape.astype(box_xy.dtype)
    image_shape = image_shape.astype(box_xy.dtype)
    new_shape = np.round(image_shape * np.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)

    # Scale boxes back to original image shape.
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    # (N, 13, 13, 3, 4)
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # (N, 13, 13, 3, 2), (N, 13, 13, 3, 2)， (N, 13, 13, 3, 1)， (N, 13, 13, 3, 10)
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # (N, 13, 13, 3, 4)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # (x, 4)
    boxes = np.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    # (x, 10)
    box_scores = np.reshape(box_scores, [-1, num_classes])
    # (x, 4), (x, 10)
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=100,
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
    image_shape = np.array(image_shape)
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # (416, 416)
    input_shape = np.array(yolo_outputs[0].shape[1:3]) * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        # (x, 4), (x, 10)
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # (x, 4)
    boxes = np.concatenate(boxes, axis=0)
    # (x, le of classes)
    box_scores = np.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = boxes[mask[:, c]]
        class_box_scores = box_scores[:, c][mask[:, c]]

        nms_index = nms(class_boxes, class_box_scores, iou_threshold, max_boxes)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = np.concatenate(boxes_, axis=0)
    scores_ = np.concatenate(scores_, axis=0)
    classes_ = np.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
