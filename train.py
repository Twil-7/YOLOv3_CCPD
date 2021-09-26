import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import cv2
import os

from create_model import yolo_body
from loss import yolo_loss
from data_generate import data_generator


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_classes(cls_path):

    with open(cls_path) as f:
        cls_names = f.readlines()

    cls_names = [c.strip() for c in cls_names]

    return cls_names


def get_anchors(anchor_path):

    with open(anchor_path) as f:
        anchor = f.readline()

    anchor = [float(x) for x in anchor.split(',')]

    return np.array(anchor).reshape(-1, 2)


if __name__ == '__main__':

    train_path = 'model_data/2021_train.txt'
    val_path = 'model_data/2021_val.txt'

    with open(train_path) as f:
        train_lines = f.readlines()
    with open(val_path) as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    log_dir = 'Logs/'
    classes_path = 'model_data/ccpd_classes.txt'
    anchors_path = 'model_data/ccpd_anchors.txt'
    weights_path = 'model_data/yolo3_py3.8_weights.h5'

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    K.clear_session()  # get a new session
    input_shape = (416, 416)
    image_input = Input(shape=(416, 416, 3))
    h, w = input_shape
    batch_size = 32

    # 搭建YOLOv3模型
    y_true = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],
                           num_anchors // 3, num_classes + 5)) for l in range(3)]
    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    model_body.summary()
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 载入download初始权重，冰冻前249层
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

    # 构建损失层，便于模型训练
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes,
                                   'ignore_thresh': 0.5})([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    freeze_num = 249
    for i in range(freeze_num):
        model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_num, len(model_body.layers)))

    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    model.fit_generator(data_generator(train_lines, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=27,
                        callbacks=[checkpoint])

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    model.fit_generator(data_generator(train_lines, batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(val_lines, batch_size, input_shape, anchors, num_classes),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=150,
                        initial_epoch=95,
                        callbacks=[checkpoint, reduce_lr, early_stopping])




