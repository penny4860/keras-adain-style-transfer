
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
    
    import matplotlib.pyplot as plt
    plt.imshow(stylized_imgs[0])
    plt.show()
    
    
    

