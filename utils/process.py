import tensorflow as tf # type: ignore
import tensorflow_addons as tfa # type: ignore
from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph # type: ignore
import logging
from data.dataset import Dataset
from constants.const import globalVar
import os
import numpy as np # type: ignore
from functools import lru_cache
import sys
from searchspace.graph import Graph, OperatorDesc # type: ignore
from searchspace.cnnarchitecture import Conv2D, DWConv2D, Add, Pool, Dense # type: ignore

def with_probability(p, true, false):
    return tf.cond(tf.random.uniform([]) < p, true, false)


def random_rotate(x, rads=0.3):
    angle = tf.random.uniform([], minval=-rads, maxval=rads)
    return tfa.image.rotate(x, angle)

def random_shift(x, h_pixels, w_pixels):
    orig = x.shape
    x = tf.pad(x, mode="SYMMETRIC",
               paddings=tf.constant([[w_pixels, w_pixels], [h_pixels, h_pixels], [0, 0]]))
    return tf.image.random_crop(x, size=orig)


def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    if hasattr(model, 'inputs'):
        # try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            dense_flops = 0
            depth_flops = 0
            maxpool_flops = 0
            BS_flops = 0
            conv_flops = 0
            for layer in flops.children:
                if 'dense' in layer.name:
                    dense_flops += layer.total_float_ops
                elif 'depthwise' in layer.name:
                    depth_flops += layer.total_float_ops
                elif 'max_pooling' in layer.name:
                    maxpool_flops += layer.total_float_ops
                elif 'batch_normalization' in layer.name:
                    BS_flops += layer.total_float_ops
                elif 'conv2d' in layer.name:
                    conv_flops += layer.total_float_ops
                # else:
                #     print('layer {0}, layer_num {1}'.format(layer.name, layer.total_float_ops),file=open('Layer_flops_NE_unlocking2.txt', 'a'))
            # print("dense flops {0}; depth {1}; maxpool {2}; bs {3}; conv {4}; macs {5}".format(dense_flops,depth_flops,maxpool_flops,BS_flops,conv_flops, flops.total_float_ops),file=open('Layer_flops_NE_digits2.txt', 'a'))

            return flops.total_float_ops, dense_flops,depth_flops,maxpool_flops,BS_flops,conv_flops
        # except Exception as e:  # pylint: disable=broad-except
        #     logging.info(
        #         'Failed to count model FLOPs with error %s, because the build() '
        #         'methods in keras layers were not called. This is probably because '
        #         'the model was not feed any input, e.g., the max train step already '
        #         'reached before this run.', e)
        #     print("failed to count model FLOPs with error")
        #     return None
    else:
      print("models do not contain inputs")
    return None

def quantised_accuracy(model: tf.keras.Model, dataset: Dataset,
                       batch_size: int, num_representative_batches=5,
                       num_eval_workers=4, output_file=None):
    log = logging.getLogger("Quantiser")
    log.info("Computing quantised test accuracy...")

    if output_file is not None and batch_size != 1:
        print("Model output is requested, so the batch_size will be set to 1.")
        num_representative_batches = num_representative_batches * batch_size
        batch_size = 1
    def representative_dataset_gen_solar():
        input_shape = model.input_shape  
        if input_shape[0] is None:
            input_shape = (1,) + input_shape[1:] 
        print(f"Model input shape: {input_shape}")
        for _ in range(100):  
            data = np.random.rand(*input_shape).astype(np.float32)
            yield [data]  

    def representative_dataset_gen():
        data = dataset.validation_dataset().batch(batch_size, drop_remainder=True)
        if num_representative_batches:
            data = data.take(num_representative_batches)
        for sample, _ in data:
            yield [sample]
    model.inputs[0].set_shape((batch_size, ) + dataset.input_shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if globalVar.appName == 'speech':
        converter.representative_dataset = representative_dataset_gen
    elif globalVar.appName == 'solar':
        converter.representative_dataset = representative_dataset_gen_solar
    elif globalVar.appName == 'cifar10':
        input_shape = (32,32,3)
        converter.representative_dataset = \
                lambda: [[np.random.random((1,) + input_shape).astype("float32")] for _ in range(5)]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8 # change to int
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    
    with open( "model_aux.tflite","wb") as f:
            f.write(model_bytes)
    quantized_model_size = os.path.getsize("model_aux.tflite")

    return quantized_model_size



def peak_memory_usage(g: Graph, exclude_weights=True, exclude_inputs=True):
    def occupies_memory(x):
        is_input = (not x.is_constant) and x.producer is None
        is_weight = x.is_constant
        return not ((exclude_inputs and is_input) or (exclude_weights and is_weight))

    def sum_of_tensor_sizes(tensors):
        return sum(x.size for x in tensors if occupies_memory(x))

    @lru_cache(maxsize=None)
    def mem(tensors):
        # Computes the peak memory usage of a runtime system that computes all tensors in a set `tensors`.
        constants = [t for t in tensors if t.producer is None]
        if constants:
            upstream_mem_use, op_order = mem(frozenset(t for t in tensors if t.producer is not None))
            return sum_of_tensor_sizes(constants) + upstream_mem_use, op_order
        if not tensors:
            return 0, []

        min_use = sys.maxsize  # A reasonably large integer
        op_order = []
        # For each of tensors in our working set, we try to unapply the operator that produced it
        for t in tensors:
            rest = tensors - {t}
            # We constrain the search to never consider evaluating an operator (`t.producer`) more than once ---
            # so we prevent cases where we consider unapplying `t.producer` but it's actually necessary for other
            # tensors in the working set.
            if any(t in r.predecessors for r in rest):
                continue
            inputs = frozenset(t.producer.inputs)
            new_set = rest | inputs
            upstream_mem_use, operators = mem(new_set)

            def last_use_point(i):
                return all(o in operators for o in i.consumers if o != t.producer)

            if isinstance(t.producer, Add) and any(i.shape == t.shape and last_use_point(i) for i in inputs):
                # When evaluating Add, instead of creating a separate output buffer, we can accumulate into one
                # of its inputs, provided it's no longer used anywhere else (either not consumed elsewhere or its
                # other consumers have already been evaluated).
                current_mem_use = sum_of_tensor_sizes(new_set)
            else:
                current_mem_use = sum_of_tensor_sizes(new_set | {t})

            mem_use = max(upstream_mem_use, current_mem_use)
            if mem_use < min_use:
                min_use = mem_use
                op_order = operators + [t.producer]
        return min_use, op_order

    mem.cache_clear()
    if len(g.outputs) == 0:
        raise ValueError("Provided graph has no outputs. Did you call `g.add_output(...)`?.")
    peak_usage, _ = mem(frozenset(g.outputs))
    return peak_usage


def model_size(g: Graph, sparse=False):
    return sum(x.size if not sparse else (x.sparse_size or x.size)
               for x in g.tensors.values() if x.is_constant)


def macs(g: Union[Graph, OperatorDesc, List[OperatorDesc]]):
    return inference_latency(g, mem_access_weight=0, compute_weight=1)


def inference_latency(g: Union[Graph, OperatorDesc, List[OperatorDesc]],
                      mem_access_weight=0, compute_weight=1):
    if isinstance(g, Graph):
        ops = g.operators.values() # type: ignore
    elif isinstance(g, OperatorDesc):
        ops = [g]
    else:
        ops = g

    latency = 0
    for op in ops:
        loads, compute = 0, 0
        if isinstance(op, Conv2D):
            k_h, k_w, i_c, o_c = op.inputs[1].shape # type: ignore
            n, o_h, o_w, _ = op.output.shape # type: ignore
            work = n * o_h * o_w * o_c * k_h * k_w * i_c
            loads, compute = 2 * work, work
            if op.use_bias: # type: ignore
                loads += n * o_h * o_w * o_c
        if isinstance(op, DWConv2D):
            k_h, k_w, c, _ = op.inputs[1].shape # type: ignore
            n, o_h, o_w, _ = op.output.shape # type: ignore
            work = n * c * o_h * o_w * k_h * k_w
            loads, compute = 2 * work, work
            if op.use_bias: # type: ignore
                loads += n * c * o_h * o_w
        if isinstance(op, Pool):
            n, o_h, o_w, c = op.output.shape # type: ignore
            pool_h, pool_w = op.pool_size # type: ignore
            work = n * o_h * o_w * c * pool_h * pool_w
            loads, compute = work, work
        if isinstance(op, Dense):
            n, _ = op.output.shape # type: ignore
            in_dim, out_dim = op.inputs[1].shape # type: ignore
            work = n * in_dim * out_dim
            loads, compute = 2 * work, work
            if op.use_bias: # type: ignore
                loads += n * out_dim
        if isinstance(op, Add):
            # TODO: not precise when inputs are of different shapes
            num_terms = len(op.inputs) # type: ignore
            elems_per_term = np.prod(op.output.shape) # type: ignore
            loads = num_terms * elems_per_term
            compute = (num_terms - 1) * elems_per_term
        latency += mem_access_weight * loads + compute_weight * compute
    return latency

