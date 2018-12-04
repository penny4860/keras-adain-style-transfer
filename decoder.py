

import os
from adain import PROJECT_ROOT

import tensorflow as tf
from AdaIN import image_from_file, graph_from_t7

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Model = tf.keras.models.Model


decode_t7_file = os.path.join(PROJECT_ROOT, 'decoder.t7')
vgg_t7_file = os.path.join(PROJECT_ROOT, 'vgg_normalised.t7')
content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
style = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')


from adain.encoder import SpatialReflectionPadding
def decoder(input_shape=[None,None,512]):
    img_input = Input(shape=input_shape)

    # Block 1
    x = SpatialReflectionPadding()(img_input) # layer 1
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block4_conv1_decode')(x)
    model = Model(img_input, x, name='decoder')
    return model



if __name__ == '__main__':
    from adain.utils import calc_diff
    from AdaIN import AdaIN
    def run_from_torch(content, style, resize):
        with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
            c, c_filename = image_from_file(g, 'content_image', size=resize)
            s, s_filename = image_from_file(g, 'style_image',size=resize)
            _, c_vgg = graph_from_t7(c, g, vgg_t7_file)
            _, s_vgg = graph_from_t7(s, g, vgg_t7_file)
            c_vgg = c_vgg[30]
            s_vgg = s_vgg[30]
            stylized_content = AdaIN(c_vgg, s_vgg, 1.0)
            c_decoded, decodes = graph_from_t7(stylized_content, g, decode_t7_file)
            
            feed_dict = {c_filename: content, s_filename: style}
            images_torch = sess.run(decodes[2], feed_dict=feed_dict)
        return images_torch

    features_torch = run_from_torch(content, style, [224,224])
#     import matplotlib.pyplot as plt
#     plt.imshow(imgs[0])
#     plt.show()
    from adain.adain_layer import adain_combine_model
    from adain.encoder import load_and_preprocess_img
    model = adain_combine_model()
    decoder_model = decoder()

    content_input_tensor = tf.keras.layers.Input((224, 224, 3))
    style_input_tensor = tf.keras.layers.Input((224, 224, 3))
    
    x = model([content_input_tensor, style_input_tensor])
    x = decoder_model(x)
    model = Model([content_input_tensor, style_input_tensor], x, name='decoder')
    model.summary()
    
    content_imgs = load_and_preprocess_img(content, [224,224])
    style_imgs = load_and_preprocess_img(style, [224,224])
    features_keras = model.predict([content_imgs, style_imgs])
    print(features_torch.shape, features_keras.shape)
    print("scores = ", calc_diff(features_torch, features_keras))

