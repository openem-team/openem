import os
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def keras_to_tensorflow(model, pred_node_names, out_path):
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')
    num_output = len(pred_node_names)
    pred = [tf.identity(model.outputs[i], name=p) 
        for i, p in enumerate(pred_node_names)]
    sess = K.get_session()
    out_dir, out_file = os.path.split(out_path)
    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), pred_node_names)    
    graph_io.write_graph(constant_graph, out_dir, out_file, as_text=False)

