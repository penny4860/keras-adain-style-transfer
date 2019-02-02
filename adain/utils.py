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
            print("    ", weight.shape, bias.shape)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__ == '__main__':
    from adain import PROJECT_ROOT
    import os
    # print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "vgg_normalised.t7"))
    print_t7_graph(os.path.join(PROJECT_ROOT, "pretrained", "decoder.t7"))

