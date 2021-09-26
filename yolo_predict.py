import colorsys
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from keras.layers import Input, Lambda
from keras.models import Model
from keras import backend as K

from create_model import yolo_body
from decode import yolo_eval


def get_classes(path):

    with open(path) as f:
        cls_names = f.readlines()
    cls_names = [c.strip() for c in cls_names]
    return cls_names


def get_anchors(path):

    with open(path) as f:
        anchor = f.readline()
    anchor = [float(x) for x in anchor.split(',')]

    return np.array(anchor).reshape(-1, 2)


anchors_path = 'model_data/ccpd_anchors.txt'
classes_path = 'model_data/ccpd_classes.txt'
best_weights = 'ep122-loss4.643-val_loss4.524.h5'
conf_score = 0.3
iou_score = 0.2

anchors = get_anchors(anchors_path)
class_names = get_classes(classes_path)

num_anchors = len(anchors)
num_classes = len(class_names)
image_input = Input(shape=(416, 416, 3))

model = yolo_body(image_input, num_anchors // 3, num_classes)
model.summary()
model.load_weights(best_weights)


test_path = 'model_data/2021_test.txt'
with open(test_path) as f:
    test_lines = f.readlines()

t1 = time.time()
for i in range(100):

    annotation_line = test_lines[i]
    line = annotation_line.split()

    img1 = Image.open(line[0])
    image_shape = (img1.height, img1.width)    # (520, 660)

    img2 = img1.resize((416, 416))
    img3 = np.array(img2, dtype='float32')
    img4 = img3 / 255.
    img5 = np.expand_dims(img4, 0)

    # img_cv = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img_cv)
    # cv2.waitKey(0)

    result1 = model.predict(img5)
    # print(len(result1))        # 3
    # print(result1[0].shape)    # (1, 13, 13, 75)
    # print(result1[1].shape)    # (1, 26, 26, 75)
    # print(result1[2].shape)    # (1, 52, 52, 75)

    boxes_, scores_, classes_ = yolo_eval(result1, anchors, num_classes, image_shape,
                                          score_threshold=conf_score, iou_threshold=iou_score)
    # 借助keras.eval函数，将tensor张量转化为array数组
    boxes_1 = K.eval(boxes_)
    scores_1 = K.eval(scores_)
    classes_1 = K.eval(classes_)

    detect_img = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)

    if len(boxes_1) > 0:
        for k in range(len(boxes_1)):

            b1 = int(boxes_1[k, 0])
            a1 = int(boxes_1[k, 1])
            b2 = int(boxes_1[k, 2])
            a2 = int(boxes_1[k, 3])

            index = int(classes_1[k])
            pre_class = str(class_names[index])
            pre_score = round(scores_1[k], 2)    # 保留两位小数
            pre_score = str(pre_score)

            text = pre_class + ': ' + pre_score

            cv2.rectangle(detect_img, (a1, b1), (a2, b2), (0, 0, 255), 2)
            cv2.putText(detect_img, text, (int(a1), int((b1 + b2) / 2)), 1, 1, (0, 0, 255))

        # cv2.namedWindow("detect_img")
        # cv2.imshow("detect_img", detect_img)
        # cv2.waitKey(0)

    cv2.imwrite("demo/" + str(i) + '.jpg', detect_img/1.0)

t2 = time.time()
print('YOLOv3检测耗时： ', (t2 - t1) / 100)

