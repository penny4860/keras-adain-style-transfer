# -*- coding: utf-8 -*-

from adain import PROJECT_ROOT
from adain.encoder import load_and_preprocess_img
from adain.models import adain_style_transfer
import numpy as np
import os

def test_style_transfer():
    
    content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
    style = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')
    
    true_transfer_img = np.load("stylized_imgs_truth.npy")

    model = adain_style_transfer()
    model.summary()

    content_imgs = load_and_preprocess_img(content, (512,512))
    style_imgs = load_and_preprocess_img(style, (512,512))
    stylized_imgs = model.predict([content_imgs, style_imgs])
    stylized_imgs = stylized_imgs * 256
    stylized_imgs = stylized_imgs[:,:,:,::-1]

    assert np.allclose(stylized_imgs, true_transfer_img)

import pytest
if __name__ == '__main__':
    pytest.main()
