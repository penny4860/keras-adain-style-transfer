# -*- coding: utf-8 -*-

import numpy as np
import cv2
import argparse
import os

from adain.utils import preprocess, plot
from adain.graph import load_graph_from_pb
from adain import MODEL_ROOT

DEFAULT_ENCODER_PB = os.path.join(MODEL_ROOT, "mobile_encoder_opt.pb")
DEFAULT_DECODER_PB = os.path.join(MODEL_ROOT, "decoder_opt.pb")


argparser = argparse.ArgumentParser(
    description='style transfer with Adaptive Instance Normalization')

argparser.add_argument(
    '-e',
    '--encoder_pb',
    default=DEFAULT_ENCODER_PB,
    help='encoder pb file')

argparser.add_argument(
    '-d',
    '--decoder_pb',
    default=DEFAULT_DECODER_PB,
    help='decoder pb file')

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
    sess = load_graph_from_pb(args.encoder_pb)
    tensor_input = sess.graph.get_tensor_by_name('import/input:0')
    tensor_output = sess.graph.get_tensor_by_name('import/output/Relu:0')
    c_feat = sess.run(tensor_output, {tensor_input: c_img_prep})
    s_feat = sess.run(tensor_output, {tensor_input: s_img_prep})

    # 4. mix & decoding
    sess = load_graph_from_pb(args.decoder_pb)
    tensor_input_c = sess.graph.get_tensor_by_name('import_1/input_c:0')
    tensor_input_s = sess.graph.get_tensor_by_name('import_1/input_s:0')
    tensor_output = sess.graph.get_tensor_by_name('import_1/output/mul:0')
    stylized_imgs = sess.run(tensor_output, {tensor_input_c: c_feat, tensor_input_s: s_feat})
    stylized_img = stylized_imgs[0].astype(np.uint8)
   
    # 5. plot
    plot([c_img, s_img, stylized_img])
