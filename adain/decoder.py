# -*- coding: utf-8 -*-

import tensorflow as tf
import keras

from adain import USE_TF_KERAS
from adain.layers import PostPreprocess, SpatialReflectionPadding, AdaIN


if USE_TF_KERAS:
    Input = tf.keras.layers.Input
    Conv2D = tf.keras.layers.Conv2D
    Model = tf.keras.models.Model
    UpSampling2D = tf.keras.layers.UpSampling2D
    Layer = tf.keras.layers.Layer
    SeparableConv2D = tf.keras.layers.SeparableConv2D
    DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
    BatchNormalization = tf.keras.layers.BatchNormalization
    Activateion = tf.keras.layers.Activation

else:
    Input = keras.layers.Input
    Conv2D = keras.layers.Conv2D
    Model = keras.models.Model
    UpSampling2D = keras.layers.UpSampling2D
    Layer = keras.layers.Layer
    SeparableConv2D = keras.layers.SeparableConv2D
    DepthwiseConv2D = keras.layers.DepthwiseConv2D
    BatchNormalization = keras.layers.BatchNormalization
    Activateion = keras.layers.Activation


def build_vgg_decoder(input_features):
    
    # (32,32,512)
    x = input_features
    
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
    return x


def build_mobile_decoder(input_features):
    
    # (32,32,512)
    x = input_features
    
    # Block 4
    # (32,32,512)
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = UpSampling2D()(x)
    # (64,64,256)

    # Block 3
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(256, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = UpSampling2D()(x)

    # Block 2
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(128, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)
    x = UpSampling2D()(x)

    # Block 1
    x = DepthwiseConv2D((3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization(fused=False)(x)
    x = Activateion("relu")(x)
    x = Conv2D(64, (1, 1), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activateion("relu")(x)

    x = Conv2D(3, (3, 3), activation=None, padding='same', name='block1_conv2_decode')(x)
    x = PostPreprocess(name="output")(x)
    return x


def combine_and_decode_model(input_shape=[None,None,512], alpha=1.0, model="vgg"):
    c_feat_input = Input(shape=input_shape, name="input_c")
    s_feat_input = Input(shape=input_shape, name="input_s")
    
    x = AdaIN(alpha)([c_feat_input, s_feat_input])
    if model == "vgg":
        x = build_vgg_decoder(x)
    elif model == "mobile":
        x = build_mobile_decoder(x)

    model = Model([c_feat_input, s_feat_input], x, name='decoder')
    return model


def vgg_decoder(input_shape=[None,None,512]):
    input_layer = Input(shape=input_shape, name="input")
    x = build_vgg_decoder(input_layer)
    model = Model(input_layer, x, name='vgg_decoder')
    return model

def mobile_decoder(input_shape=[None,None,512]):
    input_layer = Input(shape=input_shape, name="input")
    x = build_mobile_decoder(input_layer)
    model = Model(input_layer, x, name='mobile_decoder')
    return model


if __name__ == '__main__':
    # Total params: 3,505,219
    # Total params:   408,003
    model = combine_and_decode_model(input_shape=[32,32,512], model="mobile")
    model.summary()

