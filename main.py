# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt

from adain.utils import preprocess

argparser = argparse.ArgumentParser(
    description='style transfer with Adaptive Instance Normalization')

argparser.add_argument(
    '-c',
    '--contents',
    default="input/content/chicago.jpg",
    help='content image file')

argparser.add_argument(
    '-s',
    '--style',
    default="input/style/asheville.jpg",
    help='style image file')

argparser.add_argument(
    '-a',
    '--alpha',
    default=1.0,
    type=float,
    help='style weight')

import tensorflow as tf
def load_graph(pb_file="adain/models/encoder_opt.pb"):
    sess = tf.Session()
    # load model from pb file
    with tf.gfile.GFile(pb_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    # for op in sess.graph.get_operations():
    #     print(op.name)
    return sess


if __name__ == '__main__':
    
    args = argparser.parse_args()
    
    content_fname = args.contents
    style_fname = args.style
    alpha = args.alpha
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 2. load input imgs
    c_img_prep = preprocess(c_img, (256,256))
    s_img_prep = preprocess(s_img, (256,256))
    
    # 3. encoding
    sess = load_graph("adain/models/encoder_opt.pb")
    tensor_input = sess.graph.get_tensor_by_name('import/input:0')
    tensor_output = sess.graph.get_tensor_by_name('import/output/Relu:0')
    c_feat = sess.run(tensor_output, {tensor_input: c_img_prep})
    s_feat = sess.run(tensor_output, {tensor_input: s_img_prep})

    # 4. mix & decoding
    sess = load_graph("adain/models/decoder_opt.pb")
    tensor_input_c = sess.graph.get_tensor_by_name('import_1/input_c:0')
    tensor_input_s = sess.graph.get_tensor_by_name('import_1/input_s:0')
    tensor_output = sess.graph.get_tensor_by_name('import_1/output/mul:0')
    stylized_imgs = sess.run(tensor_output, {tensor_input_c: c_feat, tensor_input_s: s_feat})
    stylized_img = stylized_imgs[0].astype(np.uint8)
   
    # 4. plot
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("content image")
    plt.imshow(c_img)
    plt.subplot(1, 3, 2)
    plt.axis('off')    
    plt.title("style image")
    plt.imshow(s_img)
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("stylized image")
    plt.imshow(stylized_img)
    plt.show()

