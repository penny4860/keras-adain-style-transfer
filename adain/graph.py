# -*- coding: utf-8 -*-

import tensorflow as tf
def load_graph_from_pb(pb_file="adain/models/encoder_opt.pb", print_op_name=False):
    sess = tf.Session()
    # load model from pb file
    with tf.gfile.GFile(pb_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
    if print_op_name:
        for op in sess.graph.get_operations():
            print(op.name)
    return sess




