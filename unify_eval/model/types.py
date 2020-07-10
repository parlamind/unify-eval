from typing import Union

import numpy as np
from fastai.text import transform
from tensorflow import Tensor as TF_Tensor
from torch import Tensor as PT_Tensor

"""
Type aliases for input data and tensors
"""

Tensor = Union[np.ndarray, PT_Tensor, TF_Tensor, float]
Label = Union[str, int]
ListOfRawTexts = Union[transform.Collection[str], np.ndarray]
ListOfTokenizedTexts = Union[transform.Collection[transform.Collection[str]], np.ndarray]
