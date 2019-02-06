
# import modules
import numpy as np
import tensorflow as tf
import os
np.random.seed(1337)
from adain.encoder import vgg19_light, vgg19
from adain.generator import BatchGenerator, create_callbacks


IMG_ROOT = "experiments/imgs"


if __name__ == '__main__':
    input_size = 256
    
    model = vgg19_light(input_shape=[input_size,input_size,3])
    model.load_weights("experiments/mobile_encoder.h5", by_name=True)
    truth_encoder = vgg19(t7_file=None, input_shape=[input_size,input_size,3])
    truth_encoder.load_weights(os.path.join("adain", "models", "vgg_encoder.h5"))
    
    import glob
    fnames = glob.glob(IMG_ROOT+"/*.jpg")
    train_generator = BatchGenerator(fnames, batch_size=8, truth_model=truth_encoder, shuffle=False)
    # valid_generator = BatchGenerator(fnames[160:], batch_size=4, shuffle=False)
    
    # 2. create loss function
    model.compile(loss="mean_squared_error",
                  optimizer=tf.keras.optimizers.Adam(lr=1e-2))
    model.fit_generator(train_generator,
                        steps_per_epoch=len(train_generator),
                        callbacks=create_callbacks(),
                        validation_data  = train_generator,
                        validation_steps = len(train_generator),
                        epochs=1000)



