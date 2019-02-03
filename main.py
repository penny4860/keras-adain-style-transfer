# -*- coding: utf-8 -*-

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


if __name__ == '__main__':
    
    args = argparser.parse_args()
    
    content_fname = args.contents
    style_fname = args.style
    alpha = args.alpha
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 3. run style transfer
    c_img_prep = preprocess(c_img, (512,512))
    s_img_prep = preprocess(s_img, (512,512))
    
    # 1) encode images
    from adain.encoder import vgg_encoder
    encoder_model = vgg_encoder()
    c_features = encoder_model.predict(c_img_prep)
    s_features = encoder_model.predict(s_img_prep)
    print(c_features.shape, s_features.shape)
    
    # 2) combine & decode
    from adain.decoder import combine_and_decode_model
    decoder_model = combine_and_decode_model(alpha=1.0)
    stylized_imgs = decoder_model.predict([c_features, s_features])
    print(stylized_imgs.shape)
    
    import numpy as np
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

