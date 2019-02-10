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
    return image


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

def plot(imgs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("content image")
    plt.imshow(imgs[0])
    plt.subplot(1, 3, 2)
    plt.axis('off')    
    plt.title("style image")
    plt.imshow(imgs[1])
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("stylized image")
    plt.imshow(imgs[2])
    plt.show()


def print_t7_graph(t7_file):
    import torchfile
    t7 = torchfile.load(t7_file, force_8bytes_long=True)
    for idx, module in enumerate(t7.modules):
        print("{}, {}".format(idx, module._typename))
        
        weight = module.weight
        bias = module.bias
        if weight is not None:
            weight = weight.transpose([2,3,1,0])
            print("    ", weight.shape, bias.shape)


if __name__ == '__main__':
    from adain import PROJECT_ROOT
    import os
    # print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "vgg_normalised.t7"))
    print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "decoder.t7"))

