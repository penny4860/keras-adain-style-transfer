# -*- coding: utf-8 -*-

import tensorflow as tf
import keras
import os

from adain import PROJECT_ROOT
from adain.encoder import SpatialReflectionPadding

t7_file = os.path.join(PROJECT_ROOT, "pretrained/decoder-content-similar.t7")

Input = keras.layers.Input
Conv2D = keras.layers.Conv2D
Model = keras.models.Model
UpSampling2D = keras.layers.UpSampling2D


class PostPreprocess(keras.layers.Layer):
 
    def __init__(self, **kwargs):
        super(PostPreprocess, self).__init__(**kwargs)
 
    def call(self, x):
        x = tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0)
        x = x * 255
        return x


def combine_and_decode_model(input_shape=[None,None,512], alpha=1.0, t7_file=t7_file):
    c_feat_input = Input(shape=input_shape, name="input_c")
    s_feat_input = Input(shape=input_shape, name="input_s")
    
    from adain.adain_layer import AdaIN
    x = AdaIN(alpha)([c_feat_input, s_feat_input])

    # Block 4
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block4_conv1_decode')(x)
    x = UpSampling2D()(x)

    # Block 3
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1_decode')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2_decode')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3_decode')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block3_conv4_decode')(x)
    x = UpSampling2D()(x)

    # Block 2
    x = SpatialReflectionPadding()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1_decode')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block2_conv2_decode')(x)
    x = UpSampling2D()(x)

    # Block 1
    x = SpatialReflectionPadding()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1_decode')(x)
    x = SpatialReflectionPadding()(x)
    x = Conv2D(3, (3, 3), activation=None, padding='valid', name='block1_conv2_decode')(x)
    x = PostPreprocess(name="output")(x)
    
    model = Model([c_feat_input, s_feat_input], x, name='decoder')
    
    from adain.utils import get_params, set_params
    weights, biases = get_params(t7_file)
    set_params(model, weights, biases)
    return model


if __name__ == '__main__':
    model = combine_and_decode_model(t7_file=t7_file)
    model.summary()

    # 1. to frozen pb
    from adain.utils import freeze_session
    from keras import backend as K
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "models", "decoder.pb", as_text=False)
    # input_c,input_s  / output/mul
    for t in model.inputs + model.outputs:
        print("op name: {}, shape: {}".format(t.op.name, t.shape))



