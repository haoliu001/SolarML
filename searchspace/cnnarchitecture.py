import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from .architecture import Architecture # type: ignore
from math import ceil
from typing import List, Optional, Tuple, Union
from .graph import _get_current_graph, OperatorDesc, TensorDesc, Graph # type: ignore

def Input(shape, name=None):
    graph = _get_current_graph()
    i = graph.add_tensor("input" if name is None else name, shape)
    graph.add_input(i)
    return i

class Conv2D(OperatorDesc):
    """
    2D Convolution
    """
    def __init__(self, filters: int, kernel_size: int, stride: int = 1, use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None, padding: str = "valid",
                 sparse_kernel_size: Optional[int] = None, name: str = None): # type: ignore
        assert padding in ["valid", "same"]
        super().__init__("conv2d" if name is None else name)
        self.num_filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        self._add_weight(shape=(self.kernel_size, self.kernel_size, in_channels, self.num_filters),
                         suffix="weight", sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(self.num_filters, ), suffix="bias")
        h_ = h if self.padding == "same" else h - self.kernel_size + 1
        w_ = w if self.padding == "same" else w - self.kernel_size + 1
        output_shape = (batch_size, ceil(h_ / self.stride), ceil(w_ / self.stride), self.num_filters)
        return self._produce_output(shape=output_shape)


class DWConv2D(OperatorDesc):
    """
    Depthwise 2D Convolution (nb. does not include the 1x1 convolution that typically follows d/wise convolution).
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: str = "same", use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None,
                 sparse_kernel_size: Optional[int] = None, name: str = None): # type: ignore
        assert padding in ["valid", "same"]
        super().__init__("dw_conv2d" if name is None else name)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        self._add_weight(shape=(self.kernel_size, self.kernel_size, in_channels, 1),
                         suffix="weight", sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(in_channels, ), suffix="bias")
        h_ = h if self.padding == "same" else h - self.kernel_size + 1
        w_ = w if self.padding == "same" else w - self.kernel_size + 1
        output_shape = (batch_size, ceil(h_ / self.stride), ceil(w_ / self.stride), in_channels)
        return self._produce_output(shape=output_shape)


class Dense(OperatorDesc):
    def __init__(self, units: int, preflatten_input: bool = False, use_bias: bool = True,
                 batch_norm: bool = False, activation: Optional[str] = None,
                 sparse_kernel_size: Optional[int] = None, name: str = None): # type: ignore
        super().__init__("dense" if name is None else name)
        self.units = units
        self.flatten = preflatten_input
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.activation = activation
        self.sparse_kernel_size = sparse_kernel_size

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 2 or (self.flatten and len(x.shape) > 2)
        batch_size, input_dim = x.shape[0], np.prod(x.shape[1:])
        self._add_weight(shape=(input_dim, self.units), suffix="weight",
                         sparse_size=self.sparse_kernel_size)
        if self.use_bias:
            self._add_weight(shape=(self.units, ), suffix="bias")
        return self._produce_output(shape=(batch_size, self.units))


class Pool(OperatorDesc):
    def __init__(self, pool_size: Union[int, Tuple[int, int]], type: str, name: str = None): # type: ignore
        super().__init__("pool" if name is None else name)
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.type = type

    def __call__(self, x: TensorDesc):
        super().__call__(x)
        assert len(x.shape) == 4
        batch_size, h, w, in_channels = x.shape
        pool_h, pool_w = self.pool_size
        assert pool_h <= h and pool_w <= w, f"Can't apply {self.pool_size} pooling to {x.shape}"
        output_shape = (batch_size, ceil(h / pool_h), ceil(w / pool_w), in_channels)
        return self._produce_output(shape=output_shape)


class Add(OperatorDesc):
    def __init__(self, all_equal_shape: bool = True, name: str = None): # type: ignore
        super().__init__("add" if name is None else name)
        # if `all_equal_shape` is False, inputs can be of different sizes (but same dimensionality) and will be padded
        self.all_equal_shape = all_equal_shape

    def __call__(self, xs: List[TensorDesc]):
        super().__call__(xs)

        def all_equal(l):
            return l[1:] == l[:-1]

        assert len(xs) >= 2

        if self.all_equal_shape:
            assert all_equal([x.shape for x in xs])
            output_shape = xs[0].shape
        else:
            assert all_equal([len(x.shape) for x in xs])
            output_shape = [1, ] * len(xs[0].shape)
            for i in range(len(output_shape)):
                output_shape[i] = max(x.shape[i] for x in xs)
            output_shape = tuple(output_shape)
        return self._produce_output(shape=output_shape)


class CnnArchitecture(Architecture):
    """
    A candidate architecture in the search process.
    Internally, the architecture is represented as a set of nested dictionaries, describing each block and layer.
    The representation can be converted to:
    * A feature vector according to the schema (e.g. for the GP modelling) using `.to_feature_vector()`.
    * A Keras model for training using `.to_keras_model()`.
    * A resource graph model for computing resource usages using `.to_resource_graph()`.
    A random architecture can be generated using `.random_arch(...)`.
    """
    def __init__(self, architecture_dict):
        # self.architecture is a nested dictionary. If a key starts with underscore,
        # its value is predetermined by other parameters, but we still keep it as an
        # entry for convenient referencing or passing information to morphs.
        self.architecture = architecture_dict

    def _assemble_a_network(self, input, num_classes, conv_layer,
                            pooling_layer, dense_layer, add_layer, flatten_layer):
        """
        Assembles a network architecture, starting at `input`, using factory functions for each
        layer type.
        :returns Output tensor of the network
        """
        def tie_up_pending_outputs(outputs):
            if len(outputs) == 1:
                return outputs[0]
            smallest_h = min(o.shape[1] for o in outputs)
            smallest_w = min(o.shape[2] for o in outputs)
            xs = []
            for o in outputs:
                downsampling_h = int(round(o.shape[1] / smallest_h))
                downsampling_w = int(round(o.shape[2] / smallest_w))
                if downsampling_h > 1 or downsampling_w > 1:
                    o = pooling_layer(o, {
                        "pool_size": (downsampling_h, downsampling_w),
                        "type": "max"
                    })
                xs.append(o)
            return add_layer(xs)

        li = None  # Last seen input tensor a conv block

        xs = [input]
        for conv_block in self.architecture["conv_blocks"]:
            if conv_block["is_branch"]:
                assert li is not None, "The first block can't be a branch block."
                previous_channels = xs[0].shape[-1]

                x = li
                follow_up_with_1x1 = False
                for j, l in enumerate(conv_block["layers"]):
                    if j == len(conv_block["layers"]) - 1:
                        # Last layer in a block must produce the shame shape as
                        # the output of the previous block
                        if l["type"] == "DWConv2D":
                            # Depthwise convolution can't change the number of channels,
                            # so we'll need to follow up with an extra 1x1 convolution
                            follow_up_with_1x1 = True
                        else:
                            l["filters"] = previous_channels
                    x = conv_layer(x, l)

                if follow_up_with_1x1:
                    x = conv_layer(x, {
                        "type": "1x1Conv2D",
                        "filters": previous_channels,
                        "has_bn": False,
                        "has_relu": False,
                        "has_prepool": False,
                    })
                xs.append(x)

            else:
                x = tie_up_pending_outputs(xs)
                li = x
                for l in conv_block["layers"]:
                    x = conv_layer(x, l)
                xs = [x]

        x = tie_up_pending_outputs(xs)
        pooling = self.architecture["pooling"]
        if pooling:
            x = pooling_layer(x, pooling)

        x = flatten_layer(x)

        for l in self.architecture["dense_blocks"]:
            x = dense_layer(x, l)

        final_dense = self.architecture.get("_final_dense", {})
        final_dense.update({
            "units": 1 if num_classes == 2 else num_classes,
            "activation": None
        })
        x = dense_layer(x, final_dense)
        self.architecture["_final_dense"] = final_dense
        return x

    def to_keras_model(self, input_shape, num_classes, dropout=0.0, **kwargs):
        """
        Creates a Keras model for the candidate architecture.
        """
        from tensorflow.keras import Model # type: ignore
        from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Dense, Input, Add, Flatten, MaxPool2D, AvgPool2D, ZeroPadding2D, Dropout # type: ignore

         # add line 4
        sample_params = self.architecture["sample"]
        print("input shape in to_keras _model" ,input_shape)
        i = Input(shape=input_shape)

        def conv_layer(x, l):
            if l["has_prepool"] and (x.shape[1] > 1 or x.shape[2] > 1):
                pool_size = (min(2, x.shape[1]), min(2, x.shape[2]))
                x = MaxPool2D(pool_size=pool_size)(x)

            kernel_size = 1 if l["type"] == "1x1Conv2D" else \
                min(l["ker_size"], x.shape[1], x.shape[2])
            stride = 1 if l["type"] == "1x1Conv2D" or not l["2x_stride"] else 2
            if l["type"] in ["Conv2D", "1x1Conv2D"]:
                conv = Conv2D(filters=l["filters"],
                              kernel_size=kernel_size,
                              strides=stride,
                              padding="valid")
                x = conv(x)
            else:
                assert l["type"] == "DWConv2D"
                conv = DepthwiseConv2D(kernel_size=kernel_size,
                                       strides=stride,
                                       padding="valid")
                x = conv(x)
            if l["has_bn"]:
                bn = BatchNormalization()
                x = bn(x)
            if l["has_relu"]:
                x = ReLU()(x)
            l["_weights"] = [w.name for w in conv.trainable_weights]
            return x

        def pooling_layer(x, l):
            if isinstance(l["pool_size"], int):
                pool_h, pool_w = (l["pool_size"], l["pool_size"])
            else:
                pool_h, pool_w = l["pool_size"]
            pool_size = (min(pool_h, x.shape[1]),
                         min(pool_w, x.shape[2]))
            if l["type"] == "avg":
                return AvgPool2D(pool_size)(x)
            else:
                assert l["type"] == "max"
                return MaxPool2D(pool_size)(x)

        def dense_layer(x, l):
            if l["activation"] is not None and dropout > 0.0:
                x = Dropout(dropout)(x)
            dense = Dense(units=l["units"],
                          activation=l["activation"])
            x = dense(x)
            l["_weights"] = [w.name for w in dense.trainable_weights]
            return x

        def add_layer(xs):
            max_height = max(x.shape[1] for x in xs)
            max_width = max(x.shape[2] for x in xs)
            os = []
            for x in xs:
                h_diff = max_height - x.shape[1]
                w_diff = max_width - x.shape[2]
                if w_diff > 0 or h_diff > 0:
                    x = ZeroPadding2D(padding=((0, h_diff), (0, w_diff)))(x)
                os.append(x)
            return Add()(os)

        def flatten_layer(x):
            return Flatten()(x)

        o = self._assemble_a_network(i, num_classes, conv_layer, pooling_layer, dense_layer, add_layer, flatten_layer)
        return Model(inputs=i, outputs=o)

    def to_resource_graph(self, input_shape, num_classes, element_type=np.uint8, batch_size=1,
                          pruned_weights=None):
        """
        Assembles a resource graph for the model, which can be used to compute runtime properties.
        """

        pruned_weights = {w.name: w for w in pruned_weights} if pruned_weights else {}

        def process_pruned_weights(l):
            """ If pruned weights are available, this will extract the correct number of
                channels / units from the weight matrix, and the number of non-zero entries from
                the surviving channels / units. """
            if "_weights" not in l:
                return None, None

            name = l["_weights"][0]  # Kernel matrix is the first entry
            if name not in pruned_weights:
                return None, None

            w = pruned_weights[name]
            channel_counts = tf.math.count_nonzero(tf.reshape(w, (-1, w.shape[-1])), axis=0)
            units = int(tf.math.count_nonzero(channel_counts).numpy())
            sparse_size = sum(channel_counts.numpy())
            return max(units, 1), sparse_size

        def conv_layer(x, l):
            if l["has_prepool"] and (x.shape[1] > 1 or x.shape[2] > 1):
                pool_size = (min(2, x.shape[1]), min(2, x.shape[2]))
                x = Pool(type="max", pool_size=pool_size)(x)

            kernel_size = 1 if l["type"] == "1x1Conv2D" else \
                min(l["ker_size"], x.shape[1], x.shape[2])
            stride = 1 if l["type"] == "1x1Conv2D" or not l["2x_stride"] else 2
            if l["type"] in ["Conv2D", "1x1Conv2D"]:
                filters, sparse_kernel_length = process_pruned_weights(l)
                x = Conv2D(filters=filters or l["filters"],
                           kernel_size=kernel_size, stride=stride, padding="valid",
                           batch_norm=l["has_bn"], activation="relu" if l["has_relu"] else None,
                           sparse_kernel_size=sparse_kernel_length)(x)
            else:
                assert l["type"] == "DWConv2D"
                _, sparse_kernel_length = process_pruned_weights(l)
                x = DWConv2D(kernel_size=kernel_size, stride=stride,
                             padding="valid", batch_norm=l["has_bn"],
                             activation="relu" if l["has_relu"] else None,
                             sparse_kernel_size=sparse_kernel_length)(x)
            return x

        def pooling_layer(x, l):
            if isinstance(l["pool_size"], int):
                pool_h, pool_w = (l["pool_size"], l["pool_size"])
            else:
                pool_h, pool_w = l["pool_size"]
            pool_size = (min(pool_h, x.shape[1]),
                         min(pool_w, x.shape[2]))
            return Pool(pool_size=pool_size, type=l["type"])(x)

        def dense_layer(x, l):
            units, sparse_kernel_size = process_pruned_weights(l)
            return Dense(units=units or l["units"],
                         preflatten_input=True, activation=l["activation"],
                         sparse_kernel_size=sparse_kernel_size)(x)

        def add_layer(xs):
            return Add(all_equal_shape=False)(xs)

        def flatten_layer(x):
            return x

        g = Graph(element_type)
        with g.as_current():
            i = Input(shape=(batch_size,) + input_shape)
            o = self._assemble_a_network(i, num_classes, conv_layer, pooling_layer, dense_layer, add_layer, flatten_layer)
            g.add_output(o)

        return g
