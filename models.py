
import os
from adain import PROJECT_ROOT
from adain.utils import set_params, get_params
import tensorflow as tf

Input = tf.keras.layers.Input
Conv2D = tf.keras.layers.Conv2D
Model = tf.keras.models.Model
UpSampling2D = tf.keras.layers.UpSampling2D

decode_t7_file = os.path.join(PROJECT_ROOT, 'decoder.t7')


def adain_style_transfer():
    from adain.adain_layer import adain_combine_model
    from adain.decoder import decoder

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


    


    
    
    


