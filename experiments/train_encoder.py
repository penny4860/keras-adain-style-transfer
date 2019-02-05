

from adain.encoder import vgg19, vgg19_light
import numpy as np
np.random.seed(1337)
from keras.utils import Sequence

IMG_ROOT = "C://Users//penny//git//dataset//raccoon//imgs"


import cv2
class BatchGenerator(Sequence):
    def __init__(self, fnames, batch_size, shuffle, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.truth_encoder = vgg19(input_shape=[input_size,input_size,3])
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.fnames) /self._batch_size)

    def __getitem__(self, idx):
        """
        # Args
            idx : batch index
        """
        batch_fnames = self.fnames[idx*self.batch_size: (idx+1)*self.batch_size]
        xs = [cv2.imread(fname)[:,:,::-1] for fname in batch_fnames]
        xs = np.array([cv2.resize(img, (self.input_size,self.input_size)) for img in xs])
        ys = self.truth_encoder.predict(xs)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)


if __name__ == '__main__':
#     light_model = vgg19_light(input_shape=[256,256,3])
#     light_model.summary()
    import glob
    fnames = glob.glob(IMG_ROOT+"/*.jpg")
    train_generator = BatchGenerator(fnames, batch_size=4, shuffle=True)
    xs, ys = train_generator[0]

