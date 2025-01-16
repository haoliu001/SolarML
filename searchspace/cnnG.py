from .schema import get_schema # type: ignore
import random
from .cnnarchitecture import CnnArchitecture # type: ignore
import numpy as np # type: ignore
from constants.const import globalVar

def random_conv_layer_type(block_idx, layer_idx, layer_type, relu_prob=0.9, pre_pool_prob=0.25):
    i, j = block_idx, layer_idx
    schema = get_schema()
    layer = {"type": layer_type}
    if layer_type == "Conv2D":
        layer["ker_size"] = schema[f"conv{i}-l{j}-ker-size"].uniform_random_value()
        layer["filters"] = schema[f"conv{i}-l{j}-filters"].uniform_random_value()
        layer["2x_stride"] = schema[f"conv{i}-l{j}-2x-stride"].uniform_random_value()
    elif layer_type == "1x1Conv2D":
        layer["filters"] = schema[f"conv{i}-l{j}-filters"].uniform_random_value()
    elif layer_type == "DWConv2D":
        layer["ker_size"] = schema[f"conv{i}-l{j}-ker-size"].uniform_random_value()
        layer["2x_stride"] = schema[f"conv{i}-l{j}-2x-stride"].uniform_random_value()
    else:
        raise ValueError(f"Unknown conv layer type: {layer_type}")
    layer["has_bn"] = schema[f"conv{i}-l{j}-has-bn"].uniform_random_value()
    layer["has_relu"] = (np.random.random_sample() < relu_prob)
    layer["has_prepool"] = (np.random.random_sample() < pre_pool_prob)
    return layer


def random_conv_layer(block_idx, layer_idx):
    schema = get_schema()
    layer_type = schema[f"conv{block_idx}-l{layer_idx}-type"].uniform_random_value()
    return random_conv_layer_type(block_idx, layer_idx, layer_type)


def random_conv_block(block_idx, num_layers=None):
    schema = get_schema()

    i = block_idx
    block = {
        "is_branch": False if block_idx == 0 else schema[f"conv{i}-is-branch"].uniform_random_value(),
        "layers": []
    }
    num_layers = num_layers or schema[f"conv{i}-num-layers"].uniform_random_value()
    for j in range(num_layers):
        layer = random_conv_layer(i, j)
        block["layers"].append(layer)

    return block


def random_pooling():
    schema = get_schema()
    return {
        "type": "avg" if schema["pool-is-avg"].uniform_random_value() else "max",
        "pool_size": schema["pool-size"].uniform_random_value()
    }


def random_dense_block(block_idx):
    schema = get_schema()
    i = block_idx
    return {
        "units": schema[f"dense{i}-units"].uniform_random_value(),
        "activation": "relu"
    }


def random_arch(pooling_prob=0.9):
    """
    Generates a valid architecture by sampling free variables uniformly at random.
    :param pooling_prob: Probability of pooling
    :return: The architecture
    """
    schema = get_schema()
    arch = {
        "sample":[],
        "conv_blocks": [],
        "pooling": None,
        "dense_blocks": []
    }

    # line 8
    def generate_random_numbers(count, start, end):
        cnt = 0
        clist = []
        while (cnt < count):
          ele = random.randint(start, end)
          if ele not in clist:
            clist.append(ele)
          cnt += 1
        return clist

    # add line 2
    sense1 = schema["sense1"].uniform_random_value()
    arch["sample"].append(sense1)
    # add line 2.1
    sense2 = schema["sense2"].uniform_random_value()
    arch["sample"].append(sense2)
    sense3_num = schema["sense3"].uniform_random_value()

    # line 2.2 since solar have different channels, so we use  a list,.   Later I will change the name
    if globalVar.appName == 'solar':
      sense3 = generate_random_numbers(sense3_num, 0, 8)
    elif globalVar.appName == 'speech':
      sense3 = sense3_num
    elif globalVar.appName == 'cifar10':
      sense3 = sense3_num
    else: 
      sense3 = None
      print('no sample channel correspnds to appName')

    arch["sample"].append(sense3)

    num_conv_blocks = schema["num-conv-blocks"].uniform_random_value()
    for i in range(num_conv_blocks):
        block = random_conv_block(i)
        arch["conv_blocks"].append(block)

    if np.random.random_sample() < pooling_prob:
        arch["pooling"] = random_pooling()

    num_dense_blocks = schema["num-dense-blocks"].uniform_random_value()
    for i in range(num_dense_blocks):
        block = random_dense_block(i)
        arch["dense_blocks"].append(block)

    return CnnArchitecture(arch)
