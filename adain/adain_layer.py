# -*- coding: utf-8 -*-

import tensorflow as tf
from adain.encoder import vgg_encoder


# def adain_combine_model(alpha):
#     content_input_tensor = keras.layers.Input((None, None, 3))
#     style_input_tensor = keras.layers.Input((None, None, 3))
#     
#     encoder = vgg_encoder()
#     content_feature_maps = encoder(content_input_tensor)
#     style_feature_maps = encoder(style_input_tensor)
#     combined_feature_maps = AdaIN(alpha)([content_feature_maps, style_feature_maps])
# 
#     model = keras.models.Model([content_input_tensor, style_input_tensor], combined_feature_maps)
#     return model


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, **kwargs):
        self.alpha = alpha
        super(AdaIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert input_shape[0] == input_shape[1]
        return input_shape[0]

    def call(self, x):
        assert isinstance(x, list)
        # Todo : args
        content_features, style_features = x[0], x[1]
        style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean, 
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = self.alpha * normalized_content_features + (1 - self.alpha) * content_features
        return normalized_content_features


if __name__ == '__main__':
    pass
    


