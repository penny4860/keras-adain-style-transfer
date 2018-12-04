# -*- coding: utf-8 -*-

import tensorflow as tf

from adain.encoder import SpatialReflectionPadding


Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Model = tf.keras.models.Model
UpSampling2D = tf.keras.layers.UpSampling2D

def decoder(input_shape=[None,None,512]):
    img_input = Input(shape=input_shape)

    # Block 4
    x = SpatialReflectionPadding()(img_input) # layer 1
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
    
    model = Model(img_input, x, name='decoder')
    return model


if __name__ == '__main__':
    pass

