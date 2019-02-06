
# import modules
import numpy as np
import tensorflow as tf
np.random.seed(1337)
from adain.encoder import vgg19_light
from adain.generator import BatchGenerator, create_callbacks


IMG_ROOT = "C://Users//penny//git//dataset//raccoon//imgs"


if __name__ == '__main__':
    model = vgg19_light(input_shape=[256,256,3])
    model.load_weights("mobile_encoder.h5", by_name=True)
#     for l in model.layers[:-6]:
#         l.trainable = False

#     light_model.summary()
    import glob
    fnames = glob.glob(IMG_ROOT+"/*.jpg")
    train_generator = BatchGenerator(fnames[:2], batch_size=2, shuffle=False)
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



