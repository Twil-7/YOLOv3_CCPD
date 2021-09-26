import tensorflow as tf
from keras import backend as K


# Convert final layer features to bounding box parameters.
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):

    num_anchors = len(anchors)
    feats = tf.convert_to_tensor(feats)

    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]    # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust offsets to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_prob = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh

    return box_xy, box_wh, box_confidence, box_class_prob


def box_iou(b1, b2):

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_min = b1_xy - b1_wh_half
    b1_max = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_min = b2_xy - b2_wh_half
    b2_max = b2_xy + b2_wh_half

    intersect_min = K.maximum(b1_min, b2_min)
    intersect_max = K.minimum(b1_max, b2_max)
    intersect_wh = K.maximum(intersect_max - intersect_min, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=0.5):

    # args = [yolo_outputs, y_true]
    # anchors: shape=(N, 2), w h
    # num_classes: integer
    # ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    # loss: tensor, shape=(1,)

    num_layers = len(anchors)//3
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]

    loss = 0
    m = K.shape(yolo_outputs[0])[0]    # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):

        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]],
                                                     num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))    # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][..., 2:3]*y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):

            true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))

            return b+1, ignore_mask

        # keras版本较低时使用第1行语法，python==3.8
        # keras版本较高时使用第2行语法，python==3.6

        # _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])
        _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_cross_entropy is helpful to avoid exp overflow.

        # YOLOv3边框损失xy_wh的计算，类似YOLOv1的做法，对四个值计算均方误差。
        # 而YOLOv4边框损失xy_wh的计算，直接利用c_iou进行衡量。
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[..., 2:4])

        confidence_loss1 = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True)
        confidence_loss2 = (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                   from_logits=True) * ignore_mask
        confidence_loss = confidence_loss1 + confidence_loss2

        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf

        loss += xy_loss + wh_loss + confidence_loss + class_loss

    return loss
