# -*- coding: utf-8 -*-

from adain import PROJECT_ROOT
from adain.models import adain_style_transfer
from adain.utils import preprocess, postprocess
import numpy as np
import os
import cv2


def test_style_transfer():
    
    
    content_fname = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')
    style_fname = os.path.join(PROJECT_ROOT, 'input/style/goeritz.jpg')

    c_img = cv2.imread(content_fname)
    s_img = cv2.imread(style_fname)
    
    true_transfer_img = np.load("stylized_imgs_truth.npy")

    model = adain_style_transfer(alpha=1.0)
    model.load_weights(os.path.join(PROJECT_ROOT, "adain.h5"))

    content_imgs = preprocess(c_img, (512,512))
    style_imgs = preprocess(s_img, (512,512))
    stylized_imgs = model.predict([content_imgs, style_imgs])
    stylized_img = postprocess(stylized_imgs)

    assert np.allclose(stylized_img, true_transfer_img)

import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])

