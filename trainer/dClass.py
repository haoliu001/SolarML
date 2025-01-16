import logging
import pickle
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from pathlib import Path
from dataclasses import dataclass
from typing import List, Union, Optional
from searchspace.architecture import Architecture # type: ignore

@dataclass
class ArchitecturePoint:
    arch: Architecture
    sparsity: Optional[float] = None


@dataclass
class EvaluatedPoint:
    point: ArchitecturePoint
    val_error: float
    test_error: float
    resource_features: List[Union[int, float]]