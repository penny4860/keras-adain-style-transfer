
import tensorflow as tf

if __name__ == '__main__':
    from adain.encoder import vgg19_light
    # encoder_model = vgg_encoder()
    model = vgg19_light()
    model.load_weights("../mobile_encoder.h5", by_name=False)
    model.summary()
        
    # 1. to frozen pb
    from adain.utils import freeze_session
    K = tf.keras.backend
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "models", "encoder.pb", as_text=False)
    # input_c,input_s  / output/mul
    for t in model.inputs + model.outputs:
        print("op name: {}, shape: {}".format(t.op.name, t.shape))

    # python -m tensorflow.python.tools.optimize_for_inference --input encoder.pb --output mobile_encoder_opt.pb --input_names=input --output_names=output/Relu

