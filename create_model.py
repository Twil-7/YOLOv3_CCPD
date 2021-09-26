from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # Wrapper to set Darknet parameters for Convolution2D.
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    # Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)

    return compose(DarknetConv2D(*args, **no_bias_kwargs), BatchNormalization(), LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):

    # A series of res_blocks starting with a down_sampling Convolution2D
    # Darknet uses left and top padding instead of 'same' mode

    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)), DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])

    return x


def darknet_body(x):

    # Darknet body having 52 Convolution2D layers
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)

    return x


def make_last_layers(x, num_filters, out_filters):

    # 6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer
    x = compose(DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)

    y = compose(DarknetConv2D_BN_Leaky(num_filters*2, (3, 3)),
                DarknetConv2D(out_filters, (1, 1)))(x)

    return x, y


def yolo_body(inputs, num_anchors, num_classes):

    # Create YOLO_V3 model CNN body in Keras.
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1, y2, y3])


