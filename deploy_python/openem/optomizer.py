""" 
Module to define optomizing a graph prior to loading into a session

Defaults to using TensorRT, if not available, will revert to passing through
the original graph. This allows for code to run on platforms without tensorrt.
"""

def optomizeGraph(graph_def, output_nodes, user_trt_args=None):
    try:
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        tensor_rt_args={input_graph_def=graph_def,
                        nodes_blacklist=output_nodes,
                        precision_mode=trt.TrtPrecisionMode.FP16,
                        is_dynamic_op=True,
                        maximum_cached_engines=10,
                        max_batch_size=4}
        if user_trt_args:
            tensor_rt_args.update(user_trt_args)
        converter = trt.TrtGraphConverter(**tensor_rt_args)
        return converter.convert()
    except:
        print("WARNING: Unable to optomize graph.")
        return graph_def
    
