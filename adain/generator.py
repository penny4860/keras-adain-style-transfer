# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2
import numpy as np

def create_callbacks(saved_weights_name="mobile_encoder.h5"):
    # Make a few callbacks
    # from tf.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_weights_name, 
                                                    monitor='val_loss', 
                                                    verbose=1, 
                                                    save_best_only=True, 
                                                    mode='min', 
                                                    period=1)
    callbacks = [checkpoint]
    return callbacks


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, fnames, batch_size, shuffle, truth_model, input_size=256):
        self.fnames = fnames
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_size = input_size
        self.truth_model = truth_model        
        self.truth_model.predict(np.zeros((1,input_size,input_size,3)))
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
        ys = np.zeros((len(xs), 32, 32, 512))
        # ys = self.truth_model.predict(xs)
        return xs, ys

    def on_epoch_end(self):
        np.random.shuffle(self.fnames)



