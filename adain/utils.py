# -*- coding: utf-8 -*-

import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input


def preprocess(image, img_size=(224,224)):
    """
    # Args
        image : bgr-ordered array
    """
    image = np.expand_dims(cv2.resize(image.astype(np.float32), img_size), axis=0)
    image = preprocess_input(image)
    return image


def postprocess(images):
    # bgr -> rgb
    images = images[:,:,:,::-1]
    images[images < 0] = 0.0
    images[images > 1.0] = 1.0
    images *= 255
    return images[0].astype(np.uint8)


def set_params(model, weights, biases):
    i = 0
    for layer in model.layers:
        # assign params
        if len(layer.get_weights()) > 0:
            layer.set_weights([weights[i], biases[i]])
            i += 1
