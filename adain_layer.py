

import os
from adain import PROJECT_ROOT

import tensorflow as tf
from AdaIN import image_from_file, graph_from_t7
from adain.encoder import vgg_encoder


decoder_t7 = os.path.join(PROJECT_ROOT, 'decoder.t7')
vgg_t7_file = os.path.join(PROJECT_ROOT, 'vgg_normalised.t7')
content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
style = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')


def adain_combine_model():
    content_input_tensor = tf.keras.layers.Input((None, None, 3))
    style_input_tensor = tf.keras.layers.Input((None, None, 3))
    
    encoder = vgg_encoder()
    content_feature_maps = encoder(content_input_tensor)
    style_feature_maps = encoder(style_input_tensor)
    combined_feature_maps = AdaIN()([content_feature_maps, style_feature_maps])

    model = tf.keras.models.Model([content_input_tensor, style_input_tensor], combined_feature_maps)
    return model


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0]

    def call(self, x):
        # Todo : args
        alpha = 1.0
        content_features, style_features = x[0], x[1]
        style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
        content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
        epsilon = 1e-5
        normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                content_variance, style_mean, 
                                                                tf.sqrt(style_variance), epsilon)
        normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
        return normalized_content_features


if __name__ == '__main__':
    def run_adain_layer_from_torch(content_fname, style_fname, resize=[224,224]):
        def AdaIN(content_features, style_features, alpha):
            '''
            Normalizes the `content_features` with scaling and offset from `style_features`.
            See "5. Adaptive Instance Normalization" in https://arxiv.org/abs/1703.06868 for details.
            '''
            style_mean, style_variance = tf.nn.moments(style_features, [1,2], keep_dims=True)
            content_mean, content_variance = tf.nn.moments(content_features, [1,2], keep_dims=True)
            epsilon = 1e-5
            normalized_content_features = tf.nn.batch_normalization(content_features, content_mean,
                                                                    content_variance, style_mean, 
                                                                    tf.sqrt(style_variance), epsilon)
            normalized_content_features = alpha * normalized_content_features + (1 - alpha) * content_features
            return normalized_content_features
        
        with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
            c, c_filename = image_from_file(g, 'content_image', size=resize)
            s, s_filename = image_from_file(g, 'style_image',size=resize)
            _, c_vgg = graph_from_t7(c, g, vgg_t7_file)
            _, s_vgg = graph_from_t7(s, g, vgg_t7_file)
            c_vgg = c_vgg[30]
            s_vgg = s_vgg[30]
            stylized_content = AdaIN(c_vgg, s_vgg, alpha=1.0)
            feed_dict = {c_filename: content_fname, s_filename: style_fname}
            stylized_content = sess.run(stylized_content, feed_dict=feed_dict)
        return stylized_content

    def calc_diff(img1, img2):
        diff = img1 - img2
        diff = abs(diff)
        return diff.max()
    
    # 1. from torch code
    features_torch = run_adain_layer_from_torch(content, style, [224,224])

    model = adain_combine_model()

    from adain.encoder import load_and_preprocess_img
    content_imgs = load_and_preprocess_img(content, [224,224])
    style_imgs = load_and_preprocess_img(style, [224,224])
    features_keras = model.predict([content_imgs, style_imgs])
    print(features_keras.shape)
    print("scores = ", calc_diff(features_torch, features_keras))
    


