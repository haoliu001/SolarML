from typing import Any, Callable, Dict, List, Optional, Union
from constants.const import globalVar
from value.valuetype import ValueType, Discrete, Boolean, Categorical # type: ignore


def build_schema() -> Dict[str, ValueType]:
    keys = []
    
    if globalVar.appName == 'solar':
      keys.append(Discrete("sense1",bounds=(10,globalVar.MAX_SENSE1)))
      keys.append(Discrete("sense2",bounds=(4,globalVar.MAX_SENSE2)))
      keys.append(Discrete("sense3",bounds=(3,globalVar.MAX_SENSE3)))
    elif globalVar.appName == 'speech':
      keys.append(Discrete("sense1",bounds=(10,globalVar.MAX_SENSE1)))
      keys.append(Discrete("sense2",bounds=(18,globalVar.MAX_SENSE2)))
      keys.append(Discrete("sense3",bounds=(10,globalVar.MAX_SENSE3)))
    elif globalVar.appName == 'cifar10':
      keys.append(Discrete("sense1",bounds=(10,globalVar.MAX_SENSE1)))
      keys.append(Discrete("sense2",bounds=(4,globalVar.MAX_SENSE2)))
      keys.append(Discrete("sense3",bounds=(3,globalVar.MAX_SENSE3)))

    keys.append(Discrete("num-conv-blocks", bounds=(1, globalVar.MAX_CONV_BLOCKS)))
    for c in range(globalVar.MAX_CONV_BLOCKS): 
        keys.append(Boolean(f"conv{c}-is-branch", can_be_optional=(c > 0)))
        keys.append(Discrete(f"conv{c}-num-layers", bounds=(1, globalVar.MAX_LAYERS_PER_CONV_BLOCK), can_be_optional=True))
        for i in range(globalVar.MAX_LAYERS_PER_CONV_BLOCK): 
            keys.extend([
                Categorical(f"conv{c}-l{i}-type", values=["Conv2D", "1x1Conv2D", "DWConv2D"], can_be_optional=True),
                Discrete(f"conv{c}-l{i}-ker-size", bounds=(3, 7), increment=2, can_be_optional=True),
                Discrete(f"conv{c}-l{i}-filters", bounds=(1, 128), can_be_optional=True),
                Boolean(f"conv{c}-l{i}-2x-stride", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-pre-pool", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-bn", can_be_optional=True),
                Boolean(f"conv{c}-l{i}-has-relu", can_be_optional=True),
            ])
    keys.extend([
        Boolean("pool-is-avg", can_be_optional=True),
        Discrete("pool-size", bounds=(2, 6), increment=2, can_be_optional=True)
    ])
    keys.append(Discrete("num-dense-blocks", bounds=(1, globalVar.MAX_DENSE_BLOCKS)))
    for d in range(globalVar.MAX_DENSE_BLOCKS): # type: ignore
        keys.extend([
            Discrete(f"dense{d}-units", bounds=(10, 256), can_be_optional=(d > 0)),
        ])
    return {k.name: k for k in keys}

def get_schema():
    if globalVar._SCHEMA is None: 
        globalVar._SCHEMA = build_schema() # type: ignore
    return globalVar._SCHEMA # type: ignore

def compute_search_space_size():
    schema = get_schema()
    options = 1
    for v in schema.values(): # type: ignore
        size = 1
        if isinstance(v, Boolean):
            size = 2
        if isinstance(v, Discrete):
            min, max = v.bounds
            size = max - min + 1
        if isinstance(v, Categorical):
            size = len(v.values)
        options *= size
    return options