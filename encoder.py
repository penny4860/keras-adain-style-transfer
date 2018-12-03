

import os
from adain import PROJECT_ROOT

import tensorflow as tf
from AdaIN import image_from_file, graph_from_t7


decoder_t7 = os.path.join(PROJECT_ROOT, 'decoder.t7')
vgg_t7_file = os.path.join(PROJECT_ROOT, 'vgg_normalised.t7')
content = os.path.join(PROJECT_ROOT, 'input/content/modern.jpg')


def load_and_preprocess_img(img_fname, img_size=[224,224]):
    from keras.applications.vgg16 import preprocess_input
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
        c, c_filename = image_from_file(g, 'content_image', size=img_size)
         
        # BGR-ordered image [0, 1]-ranged
        images = sess.run(c, feed_dict = {c_filename: img_fname})
        print(images.shape)

    images_keras = preprocess_input(images * 255)
    return images_keras


class SpatialReflectionPadding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(SpatialReflectionPadding, self).__init__(**kwargs)

    def call(self, x):
        return tf.pad(x, tf.constant([[0,0], [1,1], [1,1], [0,0]]), "REFLECT")


def vgg19(t7_file, input_shape=[224,224,3]):
    
    def _get_params(t7_file):
        import torchfile
        t7 = torchfile.load(t7_file, force_8bytes_long=True)
        weights = []
        biases = []
        for idx, module in enumerate(t7.modules):
            weight = module.weight
            bias = module.bias
            if idx == 0:
                print(bias)
            elif weight is not None:
                weight = weight.transpose([2,3,1,0])
                weights.append(weight)
                biases.append(bias)
        return weights, biases
    
    def _set_params(model, weights, biases):
        i = 0
        for layer in model.layers:
            # assign params
            if len(layer.get_weights()) > 0:
                layer.set_weights([weights[i], biases[i]])
                i += 1

    def _build_model(input_shape):

        Input = tf.keras.layers.Input
        Conv2D = tf.keras.layers.Conv2D
        MaxPooling2D = tf.keras.layers.MaxPooling2D
        Model = tf.keras.models.Model
        
        img_input = Input(shape=input_shape)
    
        # Block 1
        x = SpatialReflectionPadding()(img_input) # layer 1
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv1')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv2')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv3')(x)
        x = SpatialReflectionPadding()(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block5_conv4')(x)
        model = Model(img_input, x, name='vgg19')
        return model
    
    model = _build_model(input_shape)
    weights, biases = _get_params(t7_file)
    _set_params(model, weights, biases)
    # model.load_weights("vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=True)
    return model


if __name__ == '__main__':
    def run_from_torch(img_fname, layer_idx, resize):
        with tf.Graph().as_default() as g, tf.Session(graph=g) as sess:
            c, c_filename = image_from_file(g, 'content_image', size=resize)
            net, c_vgg = graph_from_t7(c, g, vgg_t7_file)
            images_torch = sess.run(c_vgg[layer_idx], feed_dict = {c_filename: img_fname})
        return images_torch

    def calc_diff(img1, img2):
        diff = img1 - img2
        diff = abs(diff)
        return diff.max()

    def run_from_keras(img_fname, layer_idx, resize):
        vgg = vgg19(vgg_t7_file, [resize[0],resize[1],3])
        images_keras = load_and_preprocess_img(img_fname, resize)

        if layer_idx == 0:
            return images_keras
        
        model = tf.keras.models.Model(vgg.input, vgg.layers[layer_idx].output)
        return model.predict(images_keras)

    # Todo : get layer name : block4_conv1
    images_torch = run_from_torch(content, layer_idx=30, resize=[224,224])
    images_keras = run_from_keras(content, layer_idx=-16, resize=[224,224])
    print(images_torch.shape, images_keras.shape)
    print("difference = ", calc_diff(images_torch, images_keras))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(3):
        plt.subplot(1, 2, 1)
        plt.imshow(images_torch[0, :, :, i])
        plt.subplot(1, 2, 2)
        plt.imshow(images_keras[0, :, :, i])
        plt.show()

