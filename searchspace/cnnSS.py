from typing import List
from searchspace.cnnM import produce_all_morphs # type: ignore
from .searchspace import SearchSpace # type: ignore
from .architecture import Architecture # type: ignore
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar
from value.valuetype import ValueType # type: ignore
from constants.const import globalVar
from .schema import get_schema # type: ignore
from .cnnG import random_arch # type: ignore

ArchType = TypeVar('ArchType', bound=Architecture)
SchemaType = Dict[str, ValueType]

class CnnSearchSpace(SearchSpace):
    input_shape = None # type: ignore
    num_classes = None # type: ignore

    def __init__(self, dropout=0.0):
        self.dropout = dropout

    @property
    def schema(self) -> SchemaType:
        return get_schema()

    def random_architecture(self) -> ArchType:
        return random_arch() # type: ignore

    def produce_morphs(self, arch: ArchType) -> List[ArchType]:
        return produce_all_morphs(arch) # type: ignore

    def to_keras_model(self, arch: ArchType, input_shape=None, num_classes=None, **kwargs):
        input_shape = input_shape or self.input_shape
        return arch.to_keras_model(input_shape=input_shape or self.input_shape,
                                   num_classes=num_classes or self.num_classes,
                                   dropout=self.dropout,
                                   **kwargs)
