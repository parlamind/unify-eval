from typing import Union, List

import numpy as np
from tensorflow import Tensor as TF_Tensor
from torch import Tensor as PT_Tensor

"""
Type aliases for input data and tensors
"""

Tensor = Union[np.ndarray, PT_Tensor, TF_Tensor, float]
Label = Union[str, int]
ListOfRawTexts = Union[List[str], np.ndarray]
ListOfTokenizedTexts = Union[List[List[str]], np.ndarray]
