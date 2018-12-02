

import os
from adain import PROJECT_ROOT

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
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        
        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        model = Model(img_input, x, name='vgg19')
        return model
    
    model = _build_model(input_shape)
    weights, biases = _get_params(t7_file)
    _set_params(model, weights, biases)
    
    return model


if __name__ == '__main__':

    import tensorflow as tf
    from AdaIN import image_from_file, graph_from_t7
    resize = [224,224]
    with tf.Graph().as_default() as g, tf.Session(graph=g) as sess, tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
        c, c_filename = image_from_file(g, 'content_image', size=resize)
         
        net, c_vgg = graph_from_t7(c, g, vgg_t7_file)
        # preds = sess.run(net, feed_dict = {c_filename: content})
        
        preprocess_layer = c_vgg[3]
        images_torch = sess.run(net, feed_dict = {c_filename: content})

    print(images_torch.shape)

    images_keras = load_and_preprocess_img(content, [224,224])
    vgg = vgg19(vgg_t7_file, [224,224,3])
    
    # model = tf.keras.models.Model(vgg.input, vgg.layers[1].output)
    images_keras = vgg.predict(images_keras)
          
    import matplotlib.pyplot as plt
    plt.imshow(images_torch[0, :, :, 100])
    plt.show()
    plt.imshow(images_keras[0, :, :, 100])
    plt.show()
    

