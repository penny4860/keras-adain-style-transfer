# -*- coding: utf-8 -*-

import cv2
import argparse
import matplotlib.pyplot as plt

from adain.models import adain_style_transfer
from adain.utils import preprocess, postprocess

argparser = argparse.ArgumentParser(
    description='style transfer with Adaptive Instance Normalization')

argparser.add_argument(
    '-c',
    '--contents',
    default="input/content/brad_pitt.jpg",
    help='content image file')

argparser.add_argument(
    '-s',
    '--style',
    default="input/style/sketch.png",
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
    c_img = cv2.imread(content_fname)
    s_img = cv2.imread(style_fname)

    # 2. get model
    model = adain_style_transfer(alpha=alpha)
    model.load_weights("adain.h5")
    
    # 3. run style transfer
    c_img_prep = preprocess(c_img, (512,512))
    s_img_prep = preprocess(s_img, (512,512))
    stylized_imgs = model.predict([c_img_prep, s_img_prep])
    stylized_img = postprocess(stylized_imgs)

    # 4. plot
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title("content image")
    plt.imshow(c_img[:,:,::-1])
    plt.subplot(1, 3, 2)
    plt.axis('off')    
    plt.title("style image")
    plt.imshow(s_img[:,:,::-1])
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title("stylized image")
    plt.imshow(stylized_img)
    plt.show()

