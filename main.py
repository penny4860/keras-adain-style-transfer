# -*- coding: utf-8 -*-

import os
from adain import PROJECT_ROOT
from adain.models import adain_style_transfer
from adain.encoder import load_and_preprocess_img
from adain.utils import postprocess
    

if __name__ == '__main__':
    content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
    style = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')
    
    model = adain_style_transfer()

    content_imgs = load_and_preprocess_img(content, (512,512))
    style_imgs = load_and_preprocess_img(style, (512,512))
    stylized_imgs = model.predict([content_imgs, style_imgs])
    stylized_img = postprocess(stylized_imgs)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
#     plt.subplot(1, 3, 1)
#     plt.imshow(content_imgs)
#     plt.subplot(1, 3, 2)
#     plt.imshow(style_imgs)
    plt.subplot(1, 3, 3)
    plt.imshow(stylized_img)
    plt.show()

