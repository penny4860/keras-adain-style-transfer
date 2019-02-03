# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf

preprocess_input = tf.keras.applications.vgg16.preprocess_input

def preprocess(image, img_size=(224,224)):
    """
    # Args
        image : rgb-ordered array
    """
    image = np.expand_dims(cv2.resize(image.astype(np.float32), img_size), axis=0)
    image = preprocess_input(image)
    return image


def postprocess(images):
    images[images < 0] = 0.0
    images[images > 1.0] = 1.0
    images *= 255
    return images[0].astype(np.uint8)


def get_params(t7_file):
    import torchfile
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    weights = []
    biases = []
    for idx, module in enumerate(t7.modules):
        weight = module.weight
        bias = module.bias
        if idx == 0:
            print(bias)
        elif weight is not None:
            weight = weight.transpose([2,3,1,0])
            weights.append(weight)
            biases.append(bias)
    return weights, biases


def set_params(model, weights, biases):
    i = 0
    for layer in model.layers:
        # assign params
        if len(layer.get_weights()) > 0:
            layer.set_weights([weights[i], biases[i]])
            i += 1


def print_t7_graph(t7_file):
    import torchfile
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    for idx, module in enumerate(t7.modules):
        print("{}, {}".format(idx, module._typename))
        
        weight = module.weight
        bias = module.bias
        if weight is not None:
            weight = weight.transpose([2,3,1,0])
            print(weight.shape, bias.shape)

