# -*- coding: utf-8 -*-

import os
import cv2

from adain import PROJECT_ROOT
from adain.models import adain_style_transfer
from adain.utils import preprocess, postprocess
    

if __name__ == '__main__':
    content_fname = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
    style_fname = os.path.join(PROJECT_ROOT, 'input/style/sketch.png')
    
    # 1. contents / style images
    c_img = cv2.imread(content_fname)
    s_img = cv2.imread(style_fname)

    # 2. get model    
    model = adain_style_transfer()
    
    # 3. run
    c_img_prep = preprocess(c_img, (512,512))
    s_img_prep = preprocess(s_img, (512,512))
    stylized_imgs = model.predict([c_img_prep, s_img_prep])
    stylized_img = postprocess(stylized_imgs)

    # 4. plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.subplot(1, 3, 1)
    plt.imshow(c_img[:,:,::-1])
    plt.subplot(1, 3, 2)
    plt.imshow(s_img[:,:,::-1])
    plt.subplot(1, 3, 3)
    plt.imshow(stylized_img)
    plt.show()

