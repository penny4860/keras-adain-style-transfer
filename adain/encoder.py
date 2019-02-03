# -*- coding: utf-8 -*-

import os
from adain import PROJECT_ROOT
from adain.utils import get_params, set_params

import tensorflow as tf


vgg_t7_file = os.path.join(PROJECT_ROOT, "pretrained", 'vgg_normalised.t7')


def vgg_encoder():
    vgg = vgg19(vgg_t7_file, [None,None,3])
    # Todo : hard-coding
    model = tf.keras.models.Model(vgg.input, vgg.layers[-16].output)
    return model


class SpatialReflectionPadding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), "REFLECT")


def vgg19(t7_file=vgg_t7_file, input_shape=[256,256,3]):
    
    def _build_model(input_shape):

        Input = tf.keras.layers.Input
        Conv2D = tf.keras.layers.Conv2D
        MaxPooling2D = tf.keras.layers.MaxPooling2D
        Model = tf.keras.models.Model
        
        img_input = Input(shape=input_shape)
    
        # Block 1
        x = SpatialReflectionPadding()(img_input) # layer 1
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
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv4')(x)
        model = Model(img_input, x, name='vgg19')
        return model
    
    model = _build_model(input_shape)
    weights, biases = get_params(t7_file)
    set_params(model, weights, biases)
    return model


if __name__ == '__main__':
    model = vgg19()
    model.summary()
