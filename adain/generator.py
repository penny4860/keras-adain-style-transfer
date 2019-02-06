# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np
from adain.encoder import vgg19

def _create_callbacks(saved_weights_name="mobile_encoder.h5"):
    # Make a few callbacks
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    checkpoint = ModelCheckpoint(saved_weights_name, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 period=1)
    callbacks = [checkpoint]
    return callbacks


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, batch_size, shuffle, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.truth_encoder = vgg19(input_shape=[input_size,input_size,3])
        self.truth_encoder.predict(np.zeros((1,input_size,input_size,3)))
        self.on_epoch_end()

    def __len__(self):
        return int(len(self.fnames) /self.batch_size)

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



