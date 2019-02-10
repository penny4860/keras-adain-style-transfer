# -*- coding: utf-8 -*-

import os
from adain import PROJECT_ROOT
from adain.utils import get_params, set_params

import tensorflow as tf
import keras


vgg_t7_file = os.path.join(PROJECT_ROOT, "pretrained", 'vgg_normalised.t7')


def vgg_encoder():
    vgg = vgg19(vgg_t7_file, [None,None,3])
    # Todo : hard-coding
    model = keras.models.Model(vgg.input, vgg.layers[-1].output)
    return model
    
    
class VggPreprocess(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(VggPreprocess, self).__init__(**kwargs)

    def call(self, x):
        import numpy as np
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        return x


class SpatialReflectionPadding(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        # Todo: mode="REFLECT"이 없어지면 안드로이드에서 모델 로딩이 안됨. 왜????
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")


def vgg19(t7_file=vgg_t7_file, input_shape=[256,256,3]):
    
    def _build_model(input_shape):

        Input = keras.layers.Input
        Conv2D = keras.layers.Conv2D
        MaxPooling2D = keras.layers.MaxPooling2D
        Model = keras.models.Model
        
        x = Input(shape=input_shape, name="input")
        img_input = x
    
        # Block 1
        x = VggPreprocess()(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name="output")(x)
        model = Model(img_input, x, name='vgg19')
        return model
    
    model = _build_model(input_shape)
    if t7_file:
        weights, biases = get_params(t7_file)
        set_params(model, weights, biases)
    return model


def vgg19_light(input_shape=[256,256,3]):

    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    DepthwiseConv2D = keras.layers.DepthwiseConv2D
    Model = keras.models.Model
    BatchNormalization = keras.layers.BatchNormalization
    Activateion = keras.layers.Activation
    MaxPooling2D = keras.layers.MaxPooling2D

    x = Input(shape=input_shape, name="input")
    img_input = x

    # Block 1
    x = VggPreprocess()(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    
    x = DepthwiseConv2D((3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
    # Block 2
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    
    # Block 4
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(512, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    model = Model(img_input, x, name='vgg19_light')
    return model


if __name__ == '__main__':
    model = vgg19(None)
    light_model = vgg19_light()

    model.summary()
    light_model.summary()

