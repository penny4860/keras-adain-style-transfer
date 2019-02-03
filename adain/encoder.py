# -*- coding: utf-8 -*-

import os
from adain import PROJECT_ROOT
from adain.utils import get_params, set_params

import tensorflow as tf
import keras


vgg_t7_file = os.path.join(PROJECT_ROOT, "pretrained", 'vgg_normalised.t7')


def vgg_encoder():
    vgg = vgg19(vgg_t7_file, [512,512,3])
    # Todo : hard-coding
    model = keras.models.Model(vgg.input, vgg.layers[-9].output)
    return model


class SpatialReflectionPadding(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), "REFLECT")
    
    
class VggPreprocess(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(VggPreprocess, self).__init__(**kwargs)

    def call(self, x):
        import numpy as np
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        return x


def vgg19(t7_file=vgg_t7_file, input_shape=[256,256,3]):
    
    def _build_model(input_shape):

        Input = keras.layers.Input
        Conv2D = keras.layers.Conv2D
        MaxPooling2D = keras.layers.MaxPooling2D
        Model = keras.models.Model
        
        x = Input(shape=input_shape)
        img_input = x
    
        # Block 1
        x = VggPreprocess()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        model = Model(img_input, x, name='vgg19')
        return model
    
    model = _build_model(input_shape)
    weights, biases = get_params(t7_file)
    set_params(model, weights, biases)
    return model


if __name__ == '__main__':
    encoder_model = vgg_encoder()
    encoder_model.summary()

    # 1. to frozen pb
    from adain.utils import freeze_session
    from keras import backend as K
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in encoder_model.outputs])
    tf.train.write_graph(frozen_graph, "tmp", "encoder.pb", as_text=False)
    # input_1 / block4_conv1/Relu
    for t in encoder_model.inputs + encoder_model.outputs:
        print("op name: {}, shape: {}".format(t.op.name, t.shape))

    # 2. optimize pb file
    # python -m tensorflow.python.tools.optimize_for_inference --input encoder.pb --output encoder_opt.pb --input_names=input_1 --output_names=block4_conv1/Relu

    # 3. Quantization (optional)

