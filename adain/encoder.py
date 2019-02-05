# -*- coding: utf-8 -*-

import os
from adain import PROJECT_ROOT
from adain.utils import get_params, set_params

import tensorflow as tf


vgg_t7_file = os.path.join(PROJECT_ROOT, "pretrained", 'vgg_normalised.t7')


def vgg_encoder():
    vgg = vgg19(vgg_t7_file, [None,None,3])
    # Todo : hard-coding
    model = tf.keras.models.Model(vgg.input, vgg.layers[-1].output)
    return model
    
    
class VggPreprocess(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(VggPreprocess, self).__init__(**kwargs)

    def call(self, x):
        import numpy as np
        x = tf.reverse(x, axis=[-1])
        x = x - tf.constant(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        return x


class SpatialReflectionPadding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        # Todo: mode="REFLECT"이 없어지면 안드로이드에서 모델 로딩이 안됨. 왜????
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), mode="REFLECT")


def vgg19(t7_file=vgg_t7_file, input_shape=[256,256,3]):
    
    def _build_model(input_shape):

        Input = tf.keras.layers.Input
        Conv2D = tf.keras.layers.Conv2D
        MaxPooling2D = tf.keras.layers.MaxPooling2D
        Model = tf.keras.models.Model
        
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
    weights, biases = get_params(t7_file)
    set_params(model, weights, biases)
    return model


def vgg19_light(input_shape=[256,256,3]):

    Input = tf.keras.layers.Input
    Conv2D = tf.keras.layers.Conv2D
    DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
    Model = tf.keras.models.Model
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activateion = tf.keras.layers.Activation
    
    x = Input(shape=input_shape, name="input")
    img_input = x

    # Block 1
    x = VggPreprocess()(x)
    x = Conv2D(32, (3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (112,112,32)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (112,112,64)

    x = DepthwiseConv2D((3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (56,56,64)
    x = Conv2D(128, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (56,56,128)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (56,56,128)

    x = DepthwiseConv2D((3, 3), strides=2, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (28,28,128)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (28,28,256)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    # (28,28,256)

    x = DepthwiseConv2D((3, 3), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = Conv2D(512, (1, 1), strides=1, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu", name="output")(x)
    # (28,28,512)
    
    model = Model(img_input, x, name='vgg19_light')
    return model


if __name__ == '__main__':
    model = vgg19()
    light_model = vgg19_light()
    mobilenet = tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3))
    print("======================================================")
    conv_params = []
    bn_params = []
    for layer in mobilenet.layers:
        params = layer.get_weights()
        if len(params) == 1:
            conv_params.append(params)
        if len(params) == 4:
            bn_params.append(params)
    print("======================================================")

    ci = 0
    bi = 0    
    for layer in light_model.layers:
        params = layer.get_weights()
        if len(params) == 1:
            print(layer.name, params[0].shape, conv_params[ci][0].shape)
            layer.set_weights(conv_params[ci])
            ci += 1
        if len(params) == 4:
            print(layer.name, params[0].shape, bn_params[bi][0].shape)
            layer.set_weights(bn_params[bi])
            bi += 1
    print(ci, bi)
    light_model.save_weights("mobile_init.h5")
    

