from tensorflow import keras
import tensorflow.keras.backend as K
from functools import reduce
import config
import numpy as np
from tools import utils


class Mish(keras.layers.Layer):
    """
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        custom_config = super(Mish, self).get_config()
        return custom_config

    def compute_output_shape(self, input_shape):
        return input_shape


class YOLO:
    def __init__(self, input_shape=(None, None), pre_train=''):
        self.num_classes = config.num_classes
        self.num_anchors = config.num_anchors
        self.inputs = keras.layers.Input((*input_shape, 3))
        self.darknet = self.get_darknet()
        self.darknet_model = keras.models.Model(self.inputs, self.darknet)
        self.y1 = self.darknet_model.layers[-1].output
        self.y2 = self.darknet_model.layers[204].output
        self.y3 = self.darknet_model.layers[131].output
        self.spp = self.get_spp()
        self.pan = self.get_pan()
        self.yolo = keras.models.Model(self.inputs, [*self.pan])
        if pre_train:
            print('loading pre-weights file ...')
            self.yolo.load_weights(pre_train, by_name=True, skip_mismatch=True)
            num = (250, len(self.yolo.layers) - 3)[2 - 1]
            for i in range(num):
                self.yolo.layers[i].trainable = False
            print('loading finished')

    def conv_base_block(self, inputs, filters, kernel_size, strides=(1, 1), use_bias=True, name=None):
        """
        darknet base conv vlock
        """
        if strides == (2, 2):
            padding = 'valid'
        else:
            padding = 'same'
        o = keras.layers.Conv2D(filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias,
                                kernel_regularizer=keras.regularizers.l2(5e-4),
                                name=name)(inputs)
        return o

    def conv_mish_block(self, inputs, filters, kernel_size, strides=(1, 1), use_bias=False, name=None):
        """
        darknet conv + bn + mish block
        """
        x = self.conv_base_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                 use_bias=use_bias)
        x = keras.layers.BatchNormalization()(x)
        o = Mish(name=name)(x)
        return o

    def conv_leakyrelu_block(self, inputs, filters, kernel_size, strides=(1, 1), use_bias=False, name=None):
        """
        darknet conv + bn + leakyrelu block
        """
        x = self.conv_base_block(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                 use_bias=use_bias)
        x = keras.layers.BatchNormalization()(x)
        o = keras.layers.LeakyReLU(alpha=0.1, name=name)(x)
        return o

    def res_block(self, inputs, filters, block_num, shotcut=True, name=None):
        """
        darknet res block
        """
        x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        x = self.conv_mish_block(inputs=x, filters=filters, kernel_size=3, strides=(2, 2))
        x_short = self.conv_mish_block(inputs=x, filters=filters // 2 if shotcut else filters, kernel_size=1)
        x_next = self.conv_mish_block(inputs=x, filters=filters // 2 if shotcut else filters, kernel_size=1)
        for i in range(block_num):
            y = self.conv_mish_block(inputs=x_next, filters=filters // 2, kernel_size=1)
            y = self.conv_mish_block(inputs=y, filters=filters // 2 if shotcut else filters, kernel_size=3)
            x_next = keras.layers.Add()([x_next, y])
        x_next = self.conv_mish_block(inputs=x_next, filters=filters // 2 if shotcut else filters, kernel_size=1)
        x_next = keras.layers.Concatenate()([x_next, x_short])
        o = self.conv_mish_block(inputs=x_next, filters=filters, kernel_size=1, name=name)
        return o

    def get_darknet(self):
        """
        darknet part
        """
        x = self.conv_mish_block(inputs=self.inputs, filters=32, kernel_size=3)
        x = self.res_block(inputs=x, filters=64, block_num=1, shotcut=False)
        x = self.res_block(inputs=x, filters=128, block_num=2)
        x = self.res_block(inputs=x, filters=256, block_num=8)
        x = self.res_block(inputs=x, filters=512, block_num=8)
        o = self.res_block(inputs=x, filters=1024, block_num=4)
        return o

    def get_spp(self):
        """
        spp part
        """
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=1024, kernel_size=3)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)

        ratios = [(13, 13), (9, 9), (5, 5)]
        self.y1 = o = keras.layers.Concatenate()(
            [keras.layers.MaxPooling2D(pool_size=ratio, strides=1, padding='same')(self.y1) for ratio in ratios] + [
                self.y1])
        return o

    def get_pan(self):
        """
        pan part
        """
        # up 1
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=1024, kernel_size=3)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.y1_up = self.conv_leakyrelu_block(inputs=self.y1, filters=256, kernel_size=1)
        self.y1_up = keras.layers.UpSampling2D(size=2)(self.y1_up)

        # up 2
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2 = keras.layers.Concatenate()([self.y2, self.y1_up])
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=512, kernel_size=3)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=512, kernel_size=3)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2_up = self.conv_leakyrelu_block(inputs=self.y2, filters=128, kernel_size=1)
        self.y2_up = keras.layers.UpSampling2D(size=2)(self.y2_up)

        # up 3
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=128, kernel_size=1)
        self.y3 = keras.layers.Concatenate()([self.y3, self.y2_up])
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=128, kernel_size=1)
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=256, kernel_size=3)
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=128, kernel_size=1)
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=256, kernel_size=3)
        self.y3 = self.conv_leakyrelu_block(inputs=self.y3, filters=128, kernel_size=1)

        # o 3
        self.o3 = self.conv_leakyrelu_block(inputs=self.y3, filters=256, kernel_size=3)
        self.o3 = self.conv_base_block(inputs=self.o3, filters=(self.num_classes + 5) * self.num_anchors, kernel_size=1)

        # o 2
        self.y3_down = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(self.y3)
        self.y3_down = self.conv_leakyrelu_block(inputs=self.y3_down, filters=256, kernel_size=3, strides=(2, 2))
        self.y2 = keras.layers.Concatenate()([self.y3_down, self.y2])
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=512, kernel_size=3)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=512, kernel_size=3)
        self.y2 = self.conv_leakyrelu_block(inputs=self.y2, filters=256, kernel_size=1)
        self.o2 = self.conv_leakyrelu_block(inputs=self.y2, filters=512, kernel_size=3)
        self.o2 = self.conv_base_block(inputs=self.o2, filters=(self.num_classes + 5) * self.num_anchors, kernel_size=1)

        # o 1
        self.y2_down = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(self.y2)
        self.y2_down = self.conv_leakyrelu_block(inputs=self.y2_down, filters=512, kernel_size=3, strides=(2, 2))
        self.y1 = keras.layers.Concatenate()([self.y2_down, self.y1])
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=1024, kernel_size=3)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=1024, kernel_size=3)
        self.y1 = self.conv_leakyrelu_block(inputs=self.y1, filters=512, kernel_size=1)
        self.o1 = self.conv_leakyrelu_block(inputs=self.y1, filters=1024, kernel_size=3)
        self.o1 = self.conv_base_block(inputs=self.o1, filters=(self.num_classes + 5) * self.num_anchors, kernel_size=1)

        o1, o2, o3 = self.o1, self.o2, self.o3
        # [x // 32, xx/ 16, xx/8]
        return o1, o2, o3

    @staticmethod
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

        if calc_loss:
            anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
            grid_shape = K.shape(feats)[1:3]  # height, width
            grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                            [1, grid_shape[1], 1, 1])
            grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                            [grid_shape[0], 1, 1, 1])
            grid = K.concatenate([grid_x, grid_y])
            grid = K.cast(grid, K.floatx())
            feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
            box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
            box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
            return grid, feats, box_xy, box_wh

        else:
            anchors_tensor = np.reshape(np.array(anchors), [1, 1, 1, num_anchors, 2])
            grid_shape = np.asarray(feats.shape[1:3])  # height, width
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                             [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                             [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y], axis=-1)
            grid = grid.astype(feats.dtype)

            feats = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

            box_xy = (utils.sigmoid(feats[..., :2]) + grid) / grid_shape[..., ::-1].astype(feats.dtype)
            box_wh = np.exp(feats[..., 2:4]) * anchors_tensor / input_shape[..., ::-1].astype(feats.dtype)
            box_confidence = utils.sigmoid(feats[..., 4:5])
            box_class_probs = utils.sigmoid(feats[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def __call__(self, *args, **kwargs):
        return self.yolo

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


