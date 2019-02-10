# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

from adain.utils import preprocess, plot
from adain import MODEL_ROOT

DEFAULT_ENCODER_H5 = os.path.join(MODEL_ROOT, "h5", "mobile_encoder.h5")
DEFAULT_DECODER_H5 = os.path.join(MODEL_ROOT, "h5", "vgg_decoder.h5")

content_fname="../input/content/chicago.jpg"
style_fname="../input/style/asheville.jpg"
alpha = 1.0


if __name__ == '__main__':
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)[:,:,::-1]
    s_img = cv2.imread(style_fname)[:,:,::-1]

    # 2. load input imgs
    c_img_prep = preprocess(c_img, (256,256))
    s_img_prep = preprocess(s_img, (256,256))
    
    # 3. encoding
    from adain.encoder import vgg19_light
    from adain.decoder import combine_and_decode_model
    encoder = vgg19_light()
    decoder = combine_and_decode_model()
    encoder.load_weights(DEFAULT_ENCODER_H5)
    decoder.load_weights(DEFAULT_DECODER_H5)

    c_features = encoder.predict(c_img_prep)
    s_features = encoder.predict(s_img_prep)

    stylized_imgs = decoder.predict([c_features, s_features])
    stylized_img = stylized_imgs[0].astype(np.uint8)
   
    # 5. plot
    plot([c_img, s_img, stylized_img])
