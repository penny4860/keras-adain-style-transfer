
# import modules
import numpy as np
import os
np.random.seed(1337)

IMG_ROOT = "experiments/imgs"


if __name__ == '__main__':
    import glob
    import cv2
    fnames = glob.glob(IMG_ROOT+"/*.jpg")
    print(len(fnames))
    for fname in fnames:
        img = cv2.imread(fname)
        print(img.shape)

