
import os
from adain import PROJECT_ROOT
from adain.utils import set_params, get_params
import tensorflow as tf

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Model = tf.keras.models.Model
UpSampling2D = tf.keras.layers.UpSampling2D

decode_t7_file = os.path.join(PROJECT_ROOT, 'decoder.t7')
vgg_t7_file = os.path.join(PROJECT_ROOT, 'vgg_normalised.t7')
content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
style = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')


def adain_style_transfer():
    model = adain_combine_model()
    decoder_model = decoder()
    weights, biases = get_params(decode_t7_file)
    set_params(decoder_model, weights, biases)
    
    content_input_tensor = tf.keras.layers.Input((None, None, 3))
    style_input_tensor = tf.keras.layers.Input((None, None, 3))
    
    x = model([content_input_tensor, style_input_tensor])
    x = decoder_model(x)
    model = Model([content_input_tensor, style_input_tensor], x, name='style_transfer')
    return model


if __name__ == '__main__':
    from adain.adain_layer import adain_combine_model
    from adain.encoder import load_and_preprocess_img
    from adain.decoder import decoder

    model = adain_style_transfer()
    model.summary()

    content_imgs = load_and_preprocess_img(content, [224,224])
    style_imgs = load_and_preprocess_img(style, [224,224])
    stylized_imgs = model.predict([content_imgs, style_imgs])
    
#     import matplotlib.pyplot as plt
#     plt.imshow(stylized_imgs[0])
#     plt.show()

    from adain.utils import calc_diff
    from AdaIN import AdaIN
    from AdaIN import image_from_file, graph_from_t7, postprocess_image
    def run_from_torch(content, style, resize):
        with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
            c, c_filename = image_from_file(g, 'content_image', size=resize)
            s, s_filename = image_from_file(g, 'style_image',size=resize)
            _, c_vgg = graph_from_t7(c, g, vgg_t7_file)
            _, s_vgg = graph_from_t7(s, g, vgg_t7_file)
            c_vgg = c_vgg[30]
            s_vgg = s_vgg[30]
            stylized_content = AdaIN(c_vgg, s_vgg, 1.0)
            c_decoded, _ = graph_from_t7(stylized_content, g, decode_t7_file)
            c_decoded = postprocess_image(c_decoded)
            
            feed_dict = {c_filename: content, s_filename: style}
            images_torch = sess.run(c_decoded, feed_dict=feed_dict)
        return images_torch

    stylized_torch = run_from_torch(content, style, [224,224])
    print(calc_diff(stylized_imgs, stylized_torch))

