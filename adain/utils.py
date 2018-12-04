# -*- coding: utf-8 -*-

import numpy as np
import torchfile

def postprocess(images):
    # scale range
    images = images * 256
    # bgr -> rgb
    images = images[:,:,:,::-1]
    return images.astype(np.uint8)[0]


def get_params(t7_file):
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
