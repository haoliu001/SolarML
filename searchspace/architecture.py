from tensorflow.keras.models import Model # type: ignore
from value.valuetype import ValueType # type: ignore
from .graph import Graph # type: ignore
from abc import ABC, abstractmethod


class Architecture(ABC):
    @abstractmethod
    def to_keras_model(self, input_shape, num_classes, inherit_weights_from=None, **kwargs) -> Model:
        pass
    @abstractmethod
    def to_resource_graph(self, input_shape, num_classes, **kwargs) -> Graph:
        pass